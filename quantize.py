"""Sequential AWQ(+SNC) quantizer: one decoder layer at a time.

Standard GPTQ/AWQ flow — a single calibration forward, memory-bounded (only one
layer resident), error-propagation aware (layer i+1 sees quantized layer i).
Works for Llama / Llama-3.x / Mistral / Qwen2.5 (model.model.layers).
"""
import argparse, gc, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import awq_quantize_linear
from data import get_calib

MAX_STAT_TOKENS = 4096   # subsample per linear for stats/grid (memory bound)


def _linears(module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


class _Catcher(nn.Module):
    def __init__(self, layer): super().__init__(); self.layer = layer; self.inps = []; self.kwargs = None
    def forward(self, x, **kw):
        self.inps.append(x.detach()); self.kwargs = kw
        raise StopIteration
    def __getattr__(self, name):
        try: return super().__getattr__(name)
        except AttributeError: return getattr(super().__getattr__("layer"), name)


@torch.no_grad()
def quantize_model(model, calib, bits, group_size, use_snc, p, device):
    layers = model.model.layers
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

        for n, m in lins.items():
            X = torch.cat(caught[n], 0)
            if X.shape[0] > MAX_STAT_TOKENS:
                X = X[torch.randperm(X.shape[0])[:MAX_STAT_TOKENS]]
            Wq, info = awq_quantize_linear(m.weight.data, X.to(device), bits, group_size,
                                           use_snc=use_snc, p=p)
            m.weight.data = Wq.to(m.weight.dtype)
            caught[n].clear(); del X
        caught.clear()

        def _fwd(x):
            o = layer(x.to(device), **kwargs)
            o = o[0] if isinstance(o, tuple) else o
            return o.cpu()

        inps = [_fwd(x) for x in inps]   # outputs feed next layer
        layer.cpu(); torch.cuda.empty_cache(); gc.collect()
        print(f"  layer {li+1}/{len(layers)} done", flush=True)
    model.cpu(); torch.cuda.empty_cache(); gc.collect()   # uniform device for save/eval
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
                   use_snc=(args.method == "snc"), p=args.p, device=device)
    model.save_pretrained(args.output_dir); tok.save_pretrained(args.output_dir)
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
