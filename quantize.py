"""Sequential AWQ(+SNC) quantizer: one decoder layer at a time.

Standard GPTQ/AWQ flow — a single calibration forward, memory-bounded (only one
layer resident), error-propagation aware (layer i+1 sees quantized layer i).
Works for Llama / Llama-3.x / Mistral / Qwen2.5 (model.model.layers).
"""
import argparse, gc, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import awq_quantize_linear
from data import get_calib
import snc_core as C

MAX_STAT_TOKENS = 4096   # subsample per linear for stats/grid (memory bound)


def _linears(module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


def _is_tied_lm_head(model):
    return (hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens")
            and model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())


def _infer_head_dim(model):
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "head_dim", None) is not None:
        return cfg.head_dim
    if cfg is not None and getattr(cfg, "hidden_size", None) and getattr(cfg, "num_attention_heads", None):
        return cfg.hidden_size // cfg.num_attention_heads
    return 128


def _qk_proj_names(lins):
    q_name = k_name = None
    for name in lins:
        low = name.lower()
        if "q_proj" in low or "query" in low:
            q_name = name
        elif "k_proj" in low or "key" in low:
            k_name = name
    return q_name, k_name


@torch.no_grad()
def _qk_bilinear_alphas(model, lins, X_by_name, device):
    q_name, k_name = _qk_proj_names(lins)
    if q_name is None or k_name is None:
        return {}
    if q_name not in X_by_name:
        return {}

    Wq = lins[q_name].weight.data.float().to(device)
    Wk = lins[k_name].weight.data.float().to(device)
    X = X_by_name[q_name].float().to(device)
    mu, Sigma, _, _ = C.block_stats(X)

    q_out, hidden = Wq.shape
    k_out = Wk.shape[0]
    head_dim = _infer_head_dim(model)
    if q_out % head_dim != 0 or k_out % head_dim != 0:
        return {}
    n_q_heads = q_out // head_dim
    n_k_heads = k_out // head_dim
    if n_k_heads == 0 or n_q_heads % n_k_heads != 0:
        return {}
    gqa_ratio = n_q_heads // n_k_heads

    Wq_h = Wq.view(n_q_heads, head_dim, hidden)
    Wk_h = Wk.view(n_k_heads, head_dim, hidden)

    # Q flips are scored by the K head read by that Q head.
    v_k = torch.einsum("ghd,d->gh", Wk_h, mu)
    alpha_sig_q = v_k.pow(2).repeat_interleave(gqa_ratio, dim=0)
    sig_wk_t = torch.einsum("ij,gdj->gdi", Sigma, Wk_h)
    alpha_noi_q = (Wk_h * sig_wk_t).sum(dim=2).clamp(min=0.0).repeat_interleave(gqa_ratio, dim=0)

    # K flips aggregate importance from all Q heads sharing the K head.
    v_q = torch.einsum("hjd,d->hj", Wq_h, mu)
    alpha_sig_per_q = v_q.pow(2)
    sig_wq_t = torch.einsum("ij,hdj->hdi", Sigma, Wq_h)
    alpha_noi_per_q = (Wq_h * sig_wq_t).sum(dim=2).clamp(min=0.0)
    alpha_sig_k = alpha_sig_per_q.view(n_k_heads, gqa_ratio, head_dim).sum(dim=1)
    alpha_noi_k = alpha_noi_per_q.view(n_k_heads, gqa_ratio, head_dim).sum(dim=1)

    print(f"    QK-SNC: q_heads={n_q_heads} k_heads={n_k_heads} ratio={gqa_ratio} head_dim={head_dim}",
          flush=True)
    return {
        q_name: (alpha_sig_q.reshape(q_out).cpu(), alpha_noi_q.reshape(q_out).cpu()),
        k_name: (alpha_sig_k.reshape(k_out).cpu(), alpha_noi_k.reshape(k_out).cpu()),
    }


class _Catcher(nn.Module):
    def __init__(self, layer): super().__init__(); self.layer = layer; self.inps = []; self.kwargs = None
    def forward(self, x, **kw):
        self.inps.append(x.detach()); self.kwargs = kw
        raise StopIteration
    def __getattr__(self, name):
        try: return super().__getattr__(name)
        except AttributeError: return getattr(super().__getattr__("layer"), name)


@torch.no_grad()
def quantize_model(model, calib, bits, group_size, use_snc, p, device, seed=42,
                   lam=1.0, beta=1.0, snc_guard=True, include_lm_head=False,
                   qk_snc=False):
    # AWQ token subsampling uses torch.randperm in this function and in awq.py.
    # Reset once per quantization run so independently launched AWQ/SNC runs
    # see the same calibration subsets.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    layers = model.model.layers
    accepted = rejected = 0
    model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb.to(device)

    # catch inputs + kwargs entering layer 0
    layers[0] = _Catcher(layers[0])
    for ids in calib:
        try: model.model(ids.to(device))
        except StopIteration: pass
    cap = layers[0]; layers[0] = cap.layer
    inps, kwargs = cap.inps, cap.kwargs
    kwargs.pop("past_key_values", None)   # don't accumulate KV cache across calib samples
    kwargs["use_cache"] = False
    model.model.embed_tokens.cpu()
    inps = [x.cpu() for x in inps]   # keep calibration inputs on CPU

    for li, layer in enumerate(layers):
        layer.to(device)
        lins = _linears(layer)
        caught = {n: [] for n in lins}
        hooks = [m.register_forward_hook(   # store captured activations on CPU
            lambda _m, i, _o, n=n: caught[n].append(i[0].detach().reshape(-1, i[0].shape[-1]).cpu()))
            for n, m in lins.items()]
        for x in inps:
            layer(x.to(device), **kwargs)
        for h in hooks: h.remove()

        X_by_name = {}
        for n in lins:
            X = torch.cat(caught[n], 0)
            if X.shape[0] > MAX_STAT_TOKENS:
                X = X[torch.randperm(X.shape[0])[:MAX_STAT_TOKENS]]
            X_by_name[n] = X

        if use_snc and qk_snc:
            q_name, k_name = _qk_proj_names(lins)
            if q_name in X_by_name and k_name in X_by_name:
                X_by_name[k_name] = X_by_name[q_name]
        qk_alphas = _qk_bilinear_alphas(model, lins, X_by_name, device) if (use_snc and qk_snc) else {}

        for n, m in lins.items():
            X = X_by_name[n]
            alpha_sig = alpha_noi = None
            if n in qk_alphas:
                alpha_sig, alpha_noi = qk_alphas[n]
            Wq, info = awq_quantize_linear(m.weight.data, X.to(device), bits, group_size,
                                           use_snc=use_snc, p=p, lam=lam, beta=beta,
                                           snc_guard=snc_guard,
                                           alpha_sig=alpha_sig, alpha_noi=alpha_noi)
            if info["snc_accepted"] is not None:
                accepted += int(info["snc_accepted"])
                rejected += int(not info["snc_accepted"])
            m.weight.data = Wq.to(m.weight.dtype)
            caught[n].clear(); del X
        caught.clear(); X_by_name.clear()

        def _fwd(x):
            o = layer(x.to(device), **kwargs)
            o = o[0] if isinstance(o, tuple) else o
            return o.cpu()

        inps = [_fwd(x) for x in inps]   # outputs feed next layer
        layer.cpu(); torch.cuda.empty_cache(); gc.collect()
        print(f"  layer {li+1}/{len(layers)} done", flush=True)
    model.cpu(); torch.cuda.empty_cache(); gc.collect()   # uniform device for save/eval
    if use_snc:
        print(f"  SNC guard: accepted={accepted} rejected={rejected}", flush=True)
    if include_lm_head and hasattr(model, "lm_head"):
        if _is_tied_lm_head(model):
            print("  lm_head skipped: tied to embed_tokens", flush=True)
        else:
            if hasattr(model.model, "norm") and model.model.norm is not None:
                model.model.norm.to(device)
                head_inps = [model.model.norm(x.to(device)).reshape(-1, x.shape[-1]).cpu()
                             for x in inps]
                model.model.norm.cpu()
            else:
                head_inps = [x.reshape(-1, x.shape[-1]).cpu() for x in inps]
            X = torch.cat(head_inps, 0)
            if X.shape[0] > MAX_STAT_TOKENS:
                X = X[torch.randperm(X.shape[0])[:MAX_STAT_TOKENS]]
            model.lm_head.to(device)
            Wq, info = awq_quantize_linear(model.lm_head.weight.data, X.to(device),
                                           bits, group_size, use_snc=use_snc, p=p,
                                           lam=lam, beta=beta, snc_guard=snc_guard)
            model.lm_head.weight.data = Wq.to(model.lm_head.weight.dtype)
            model.lm_head.cpu(); del X, head_inps
            torch.cuda.empty_cache(); gc.collect()
            if use_snc:
                print(f"  lm_head SNC accepted={info['snc_accepted']} flips={info['n_flips']}",
                      flush=True)
            else:
                print("  lm_head quantized", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=["awq", "snc"], default="snc")
    ap.add_argument("--bits", type=int, default=4, choices=[3, 4])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-dataset", default="c4", choices=["c4", "wikitext2"])
    ap.add_argument("--p", type=float, default=0.05, help="SNC budget fraction")
    ap.add_argument("--lam", type=float, default=1.0, help="SNC SNR lambda")
    ap.add_argument("--beta", type=float, default=1.0, help="SNC SNR beta")
    ap.add_argument("--no-snc-guard", action="store_true",
                    help="apply SNC even when calibration reconstruction MSE increases")
    ap.add_argument("--include-lm-head", action="store_true",
                    help="also quantize lm_head when it is not tied to embeddings")
    ap.add_argument("--qk-snc", action="store_true",
                    help="use bilinear GQA alpha for q_proj/k_proj SNC")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    calib = get_calib(tok, args.n_calib, args.seqlen, args.seed, args.calib_dataset)
    print(f"[{args.method}] bits={args.bits} gs={args.group_size} "
          f"calib={len(calib)} p={args.p}")
    quantize_model(model, calib, args.bits, args.group_size,
                   use_snc=(args.method == "snc"), p=args.p, device=device,
                   seed=args.seed, lam=args.lam, beta=args.beta,
                   snc_guard=not args.no_snc_guard,
                   include_lm_head=args.include_lm_head,
                   qk_snc=args.qk_snc)
    model.save_pretrained(args.output_dir); tok.save_pretrained(args.output_dir)
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
