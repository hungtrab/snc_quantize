"""Sequential AWQ(+SNC) quantizer: one decoder layer at a time.

Standard GPTQ/AWQ flow — a single calibration forward, memory-bounded (only one
layer resident), error-propagation aware (layer i+1 sees quantized layer i).
Works for Llama / Llama-3.x / Mistral / Qwen2.5 (model.model.layers).
"""
import argparse, gc, itertools, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import awq_apply_prepared, awq_prepare_linear, awq_quantize_linear
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


def _attn_proj_names(lins):
    out = {}
    for name in lins:
        low = name.lower()
        if "q_proj" in low or "query" in low:
            out["q"] = name
        elif "k_proj" in low or "key" in low:
            out["k"] = name
        elif "v_proj" in low or "value" in low:
            out["v"] = name
        elif "o_proj" in low or "out_proj" in low:
            out["o"] = name
    return out


@torch.no_grad()
def _qk_bilinear_alphas(model, lins, preps, device):
    q_name, k_name = _qk_proj_names(lins)
    if q_name is None or k_name is None:
        return {}
    if q_name not in preps or k_name not in preps:
        return {}

    q_p = preps[q_name]
    k_p = preps[k_name]
    q_scales = q_p["scales"].to(device)
    k_scales = k_p["scales"].to(device)
    Wq = lins[q_name].weight.data.float().to(device)
    Wk = lins[k_name].weight.data.float().to(device)
    Wq_for_k = Wq * k_scales[None, :]
    Wk_for_q = Wk * q_scales[None, :]

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

    Wq_h = Wq_for_k.view(n_q_heads, head_dim, hidden)
    Wk_h = Wk_for_q.view(n_k_heads, head_dim, hidden)

    # Q flips live in the AWQ-scaled Q coordinate, so K is represented in that
    # coordinate too: X Wk^T == (X / s_q) (Wk * s_q)^T.
    v_k = torch.einsum("ghd,d->gh", Wk_h, q_p["mu_s"])
    alpha_sig_q = v_k.pow(2).repeat_interleave(gqa_ratio, dim=0)
    sig_wk_t = torch.einsum("ij,gdj->gdi", q_p["Sigma_s"], Wk_h)
    alpha_noi_q = (Wk_h * sig_wk_t).sum(dim=2).clamp(min=0.0).repeat_interleave(gqa_ratio, dim=0)

    # K flips live in the AWQ-scaled K coordinate and aggregate all Q heads
    # sharing that K head under GQA.
    v_q = torch.einsum("hjd,d->hj", Wq_h, k_p["mu_s"])
    alpha_sig_per_q = v_q.pow(2)
    sig_wq_t = torch.einsum("ij,hdj->hdi", k_p["Sigma_s"], Wq_h)
    alpha_noi_per_q = (Wq_h * sig_wq_t).sum(dim=2).clamp(min=0.0)
    alpha_sig_k = alpha_sig_per_q.view(n_k_heads, gqa_ratio, head_dim).sum(dim=1)
    alpha_noi_k = alpha_noi_per_q.view(n_k_heads, gqa_ratio, head_dim).sum(dim=1)

    print(f"    QK-SNC: q_heads={n_q_heads} k_heads={n_k_heads} ratio={gqa_ratio} head_dim={head_dim}",
          flush=True)
    return {
        q_name: (alpha_sig_q.reshape(q_out).cpu(), alpha_noi_q.reshape(q_out).cpu()),
        k_name: (alpha_sig_k.reshape(k_out).cpu(), alpha_noi_k.reshape(k_out).cpu()),
    }


@torch.no_grad()
def _qk_logits_mse(X, Wq, Wk, Wq_ref, Wk_ref, head_dim, device,
                   max_tokens=256):
    """Approximate attention-logit reconstruction loss for a Q/K pair.

    This intentionally guards the bilinear dot-product that QK-SNC is trying to
    protect, instead of guarding q_proj/k_proj as independent linears.
    """
    if X.shape[0] > max_tokens:
        X = X[torch.randperm(X.shape[0])[:max_tokens]]
    X = X.float().to(device)
    Wq = Wq.float().to(device)
    Wk = Wk.float().to(device)
    Wq_ref = Wq_ref.float().to(device)
    Wk_ref = Wk_ref.float().to(device)

    q = X @ Wq.t()
    k = X @ Wk.t()
    q_ref = X @ Wq_ref.t()
    k_ref = X @ Wk_ref.t()
    n_q_heads = q.shape[1] // head_dim
    n_k_heads = k.shape[1] // head_dim
    if n_k_heads == 0 or n_q_heads % n_k_heads != 0:
        return float("inf")
    gqa_ratio = n_q_heads // n_k_heads
    q = q.view(X.shape[0], n_q_heads, head_dim)
    k = k.view(X.shape[0], n_k_heads, head_dim)
    q_ref = q_ref.view(X.shape[0], n_q_heads, head_dim)
    k_ref = k_ref.view(X.shape[0], n_k_heads, head_dim)

    total = q.new_zeros(())
    n = 0
    scale = head_dim ** -0.5
    for h in range(n_q_heads):
        g = h // gqa_ratio
        logits = (q[:, h] @ k[:, g].t()) * scale
        ref = (q_ref[:, h] @ k_ref[:, g].t()) * scale
        total += (logits - ref).pow(2).mean()
        n += 1
    return (total / max(n, 1)).item()


@torch.no_grad()
def _rope_qk_logits_mse(model, layer, lins, inps, kwargs, Wq, Wk, Wq_ref, Wk_ref,
                        device, max_samples=1, max_seq=256):
    head_dim = _infer_head_dim(model)
    Wq = Wq.float().to(device)
    Wk = Wk.float().to(device)
    Wq_ref = Wq_ref.float().to(device)
    Wk_ref = Wk_ref.float().to(device)
    total = None
    n = 0
    for x in inps[:max_samples]:
        hs = x.to(device)
        if hs.shape[1] > max_seq:
            hs = hs[:, :max_seq, :]
        B, T, _ = hs.shape
        if hasattr(layer, "input_layernorm") and layer.input_layernorm is not None:
            attn_in = layer.input_layernorm(hs)
        else:
            attn_in = hs
        pos_ids = kwargs.get("position_ids")
        if pos_ids is None:
            pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        else:
            pos_ids = pos_ids[:, :T].to(device)
        cos, sin = model.model.rotary_emb(attn_in, pos_ids)

        def _project(Wq_cur, Wk_cur):
            q = attn_in.float() @ Wq_cur.t()
            k = attn_in.float() @ Wk_cur.t()
            n_q_heads = q.shape[-1] // head_dim
            n_k_heads = k.shape[-1] // head_dim
            if n_k_heads == 0 or n_q_heads % n_k_heads != 0:
                return None, None
            q = q.view(B, T, n_q_heads, head_dim).transpose(1, 2)
            k = k.view(B, T, n_k_heads, head_dim).transpose(1, 2)
            cos_u = cos.unsqueeze(1).float()
            sin_u = sin.unsqueeze(1).float()
            q_rot = (q * cos_u) + (_rotate_half(q) * sin_u)
            k_rot = (k * cos_u) + (_rotate_half(k) * sin_u)
            k_rot = k_rot.repeat_interleave(n_q_heads // n_k_heads, dim=1)
            return q_rot, k_rot

        q, k = _project(Wq, Wk)
        q_ref, k_ref = _project(Wq_ref, Wk_ref)
        if q is None:
            return float("inf")
        logits = torch.einsum("bhtd,bhsd->bhts", q, k) * (head_dim ** -0.5)
        ref = torch.einsum("bhtd,bhsd->bhts", q_ref, k_ref) * (head_dim ** -0.5)
        valid = torch.ones((T, T), device=device, dtype=torch.bool).tril()
        attn_mask = _crop_mask(kwargs.get("attention_mask"), T, device)
        if attn_mask is not None and attn_mask.dim() >= 4:
            valid_b = valid[None, None, :, :] & torch.isfinite(attn_mask)
        else:
            valid_b = valid[None, None, :, :]
        err = (logits - ref).pow(2)
        cur = err.masked_select(valid_b.expand_as(err)).mean()
        total = cur if total is None else total + cur
        n += 1
    if total is None:
        return float("inf")
    return (total / max(n, 1)).item()


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.no_grad()
def _select_qk_pair(model, lins, preps, X_by_name, qk_alphas, bits, group_size,
                    p, lam, beta, device):
    q_name, k_name = _qk_proj_names(lins)
    if q_name not in preps or k_name not in preps:
        return {}, {}
    if q_name not in qk_alphas or k_name not in qk_alphas:
        return {}, {}

    q_p = preps[q_name]
    k_p = preps[k_name]
    q_base, _ = awq_apply_prepared(q_p, use_snc=False)
    k_base, _ = awq_apply_prepared(k_p, use_snc=False)
    q_snc, q_info = awq_apply_prepared(
        q_p, use_snc=True, p=p, lam=lam, beta=beta, snc_guard=False,
        alpha_sig=qk_alphas[q_name][0], alpha_noi=qk_alphas[q_name][1])
    k_snc, k_info = awq_apply_prepared(
        k_p, use_snc=True, p=p, lam=lam, beta=beta, snc_guard=False,
        alpha_sig=qk_alphas[k_name][0], alpha_noi=qk_alphas[k_name][1])

    Wq_ref = lins[q_name].weight.data
    Wk_ref = lins[k_name].weight.data
    head_dim = _infer_head_dim(model)
    X = X_by_name[q_name]
    candidates = [
        ("base/base", q_base, k_base, False, False),
        ("snc/base", q_snc, k_base, True, False),
        ("base/snc", q_base, k_snc, False, True),
        ("snc/snc", q_snc, k_snc, True, True),
    ]
    scored = []
    for label, Wq, Wk, use_q, use_k in candidates:
        loss = _qk_logits_mse(X, Wq, Wk, Wq_ref, Wk_ref, head_dim, device)
        scored.append((loss, label, Wq, Wk, use_q, use_k))
    loss, label, Wq, Wk, use_q, use_k = min(scored, key=lambda x: x[0])
    score_msg = " ".join(f"{label_i}={loss_i:.4e}" for loss_i, label_i, *_ in scored)
    print(f"    QK guard: choose={label} {score_msg}", flush=True)

    def _info(src, accepted):
        out = dict(src)
        out["snc_accepted"] = accepted
        if not accepted:
            out["n_flips"] = 0
            out["G"] = 0.0
        out["qk_guard"] = label
        out["qk_guard_loss"] = loss
        return out

    return {
        q_name: Wq,
        k_name: Wk,
    }, {
        q_name: _info(q_info, use_q),
        k_name: _info(k_info, use_k),
    }


@torch.no_grad()
def _select_rope_qk_pair(model, layer, lins, preps, inps, kwargs, qk_alphas,
                         bits, group_size, p, lam, beta, device):
    q_name, k_name = _qk_proj_names(lins)
    if q_name not in preps or k_name not in preps:
        return {}, {}
    if q_name not in qk_alphas or k_name not in qk_alphas:
        return {}, {}

    q_p = preps[q_name]
    k_p = preps[k_name]
    q_base, _ = awq_apply_prepared(q_p, use_snc=False)
    k_base, _ = awq_apply_prepared(k_p, use_snc=False)
    q_snc, q_info = awq_apply_prepared(
        q_p, use_snc=True, p=p, lam=lam, beta=beta, snc_guard=False,
        alpha_sig=qk_alphas[q_name][0], alpha_noi=qk_alphas[q_name][1])
    k_snc, k_info = awq_apply_prepared(
        k_p, use_snc=True, p=p, lam=lam, beta=beta, snc_guard=False,
        alpha_sig=qk_alphas[k_name][0], alpha_noi=qk_alphas[k_name][1])

    Wq_ref = lins[q_name].weight.data
    Wk_ref = lins[k_name].weight.data
    candidates = [
        ("base/base", q_base, k_base, False, False),
        ("snc/base", q_snc, k_base, True, False),
        ("base/snc", q_base, k_snc, False, True),
        ("snc/snc", q_snc, k_snc, True, True),
    ]
    scored = []
    for label, Wq, Wk, use_q, use_k in candidates:
        loss = _rope_qk_logits_mse(model, layer, lins, inps, kwargs, Wq, Wk,
                                   Wq_ref, Wk_ref, device)
        scored.append((loss, label, Wq, Wk, use_q, use_k))
    loss, label, Wq, Wk, use_q, use_k = min(scored, key=lambda x: x[0])
    score_msg = " ".join(f"{label_i}={loss_i:.4e}" for loss_i, label_i, *_ in scored)
    print(f"    RoPE QK guard: choose={label} {score_msg}", flush=True)

    def _info(src, accepted):
        out = dict(src)
        out["snc_accepted"] = accepted
        if not accepted:
            out["n_flips"] = 0
            out["G"] = 0.0
        out["rope_qk_guard"] = label
        out["rope_qk_guard_loss"] = loss
        return out

    return {
        q_name: Wq,
        k_name: Wk,
    }, {
        q_name: _info(q_info, use_q),
        k_name: _info(k_info, use_k),
    }


def _crop_mask(mask, T, device):
    if mask is None:
        return None
    mask = mask.to(device)
    if mask.dim() >= 4:
        return mask[..., :T, :T]
    if mask.dim() >= 2:
        return mask[..., :T]
    return mask


def _crop_cache_position(cache_position, T, device):
    if cache_position is None:
        return None
    return cache_position[:T].to(device)


@torch.no_grad()
def _attn_outputs(model, layer, lins, inps, kwargs, weights, device,
                  max_samples=1, max_seq=256):
    old = {}
    for name, W in weights.items():
        old[name] = lins[name].weight.data
        lins[name].weight.data = W.to(device=device, dtype=lins[name].weight.dtype)

    outs = []
    try:
        for x in inps[:max_samples]:
            hs = x.to(device)
            if hs.shape[1] > max_seq:
                hs = hs[:, :max_seq, :]
            T = hs.shape[1]
            if hasattr(layer, "input_layernorm") and layer.input_layernorm is not None:
                attn_in = layer.input_layernorm(hs)
            else:
                attn_in = hs
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = torch.arange(T, device=device).unsqueeze(0)
            else:
                pos_ids = pos_ids[:, :T].to(device)
            position_embeddings = model.model.rotary_emb(attn_in, pos_ids)
            attn_mask = _crop_mask(kwargs.get("attention_mask"), T, device)
            cache_position = _crop_cache_position(kwargs.get("cache_position"), T, device)
            out = layer.self_attn(
                attn_in,
                position_embeddings=position_embeddings,
                attention_mask=attn_mask,
                past_key_values=None,
                cache_position=cache_position,
            )
            out = out[0] if isinstance(out, tuple) else out
            outs.append(out.float().detach())
    finally:
        for name, W in old.items():
            lins[name].weight.data = W
    return outs


@torch.no_grad()
def _attention_output_mse(model, layer, lins, inps, kwargs, weights, ref_outs, device):
    outs = _attn_outputs(model, layer, lins, inps, kwargs, weights, device)
    if not outs or len(outs) != len(ref_outs):
        return float("inf")
    total = outs[0].new_zeros(())
    for out, ref in zip(outs, ref_outs):
        total += (out - ref).pow(2).mean()
    return (total / len(outs)).item()


@torch.no_grad()
def _select_attention_block(model, layer, lins, preps, inps, kwargs, qk_alphas,
                            p, lam, beta, device):
    names = _attn_proj_names(lins)
    ordered = [names[k] for k in ("q", "k", "v", "o") if k in names and names[k] in preps]
    if len(ordered) < 2:
        return {}, {}

    candidates = {}
    infos = {}
    ref_weights = {name: lins[name].weight.data.float().to(device) for name in ordered}
    for name in ordered:
        base, _ = awq_apply_prepared(preps[name], use_snc=False)
        alpha_sig = alpha_noi = None
        if name in qk_alphas:
            alpha_sig, alpha_noi = qk_alphas[name]
        snc, info = awq_apply_prepared(
            preps[name], use_snc=True, p=p, lam=lam, beta=beta,
            snc_guard=False, alpha_sig=alpha_sig, alpha_noi=alpha_noi)
        candidates[name] = (base, snc)
        infos[name] = info

    ref_outs = _attn_outputs(model, layer, lins, inps, kwargs, ref_weights, device)
    scored = []
    for mask in itertools.product((0, 1), repeat=len(ordered)):
        weights = {name: candidates[name][use_snc] for name, use_snc in zip(ordered, mask)}
        loss = _attention_output_mse(model, layer, lins, inps, kwargs, weights, ref_outs, device)
        label = "/".join(f"{name.split('.')[-1]}:{'snc' if use_snc else 'base'}"
                         for name, use_snc in zip(ordered, mask))
        scored.append((loss, label, weights, mask))

    loss, label, weights, mask = min(scored, key=lambda x: x[0])
    base_loss = scored[0][0]
    print(f"    attn guard: choose={label} loss={loss:.4e} base={base_loss:.4e}",
          flush=True)

    out_infos = {}
    for name, use_snc in zip(ordered, mask):
        src = dict(infos[name])
        src["snc_accepted"] = bool(use_snc)
        if not use_snc:
            src["n_flips"] = 0
            src["G"] = 0.0
        src["attn_guard"] = label
        src["attn_guard_loss"] = loss
        out_infos[name] = src
    return weights, out_infos


@torch.no_grad()
def _select_hybrid_attention_block(model, layer, lins, preps, X_by_name, inps,
                                   kwargs, qk_alphas, bits, group_size, p, lam,
                                   beta, device, rope_qk_guard=False):
    """Use QK-logit guard for q/k, then true attention-output guard for v/o.

    The full attention guard is a good local metric but can move q/k away from
    the bilinear objective that correlated with PPL. This hybrid keeps the
    stronger q/k decision and only uses the full attention path for value/output
    mixing.
    """
    if rope_qk_guard:
        qk_weights, qk_infos = _select_rope_qk_pair(
            model, layer, lins, preps, inps, kwargs, qk_alphas,
            bits, group_size, p, lam, beta, device)
    else:
        qk_weights, qk_infos = _select_qk_pair(
            model, lins, preps, X_by_name, qk_alphas, bits, group_size,
            p, lam, beta, device)
    names = _attn_proj_names(lins)
    vo_names = [names[k] for k in ("v", "o") if k in names and names[k] in preps]
    if not vo_names:
        return qk_weights, qk_infos

    weights_fixed = dict(qk_weights)
    candidates = {}
    infos = {}
    for name in vo_names:
        base, _ = awq_apply_prepared(preps[name], use_snc=False)
        snc, info = awq_apply_prepared(
            preps[name], use_snc=True, p=p, lam=lam, beta=beta,
            snc_guard=False)
        candidates[name] = (base, snc)
        infos[name] = info

    ref_names = set(weights_fixed) | set(vo_names)
    ref_weights = {name: lins[name].weight.data.float().to(device) for name in ref_names}
    ref_outs = _attn_outputs(model, layer, lins, inps, kwargs, ref_weights, device)

    scored = []
    for mask in itertools.product((0, 1), repeat=len(vo_names)):
        weights = dict(weights_fixed)
        for name, use_snc in zip(vo_names, mask):
            weights[name] = candidates[name][use_snc]
        loss = _attention_output_mse(model, layer, lins, inps, kwargs, weights, ref_outs, device)
        label = "/".join(f"{name.split('.')[-1]}:{'snc' if use_snc else 'base'}"
                         for name, use_snc in zip(vo_names, mask))
        scored.append((loss, label, weights, mask))

    loss, label, weights, mask = min(scored, key=lambda x: x[0])
    base_loss = scored[0][0]
    print(f"    hybrid attn guard: choose={label} loss={loss:.4e} base={base_loss:.4e}",
          flush=True)

    out_infos = dict(qk_infos)
    for name, use_snc in zip(vo_names, mask):
        src = dict(infos[name])
        src["snc_accepted"] = bool(use_snc)
        if not use_snc:
            src["n_flips"] = 0
            src["G"] = 0.0
        src["hybrid_attn_guard"] = label
        src["hybrid_attn_guard_loss"] = loss
        out_infos[name] = src
    return weights, out_infos


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
                   qk_snc=False, attn_guard=False, hybrid_guard=False,
                   rope_qk_guard=False):
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
        preps = {}
        if use_snc and qk_snc:
            for n, m in lins.items():
                preps[n] = awq_prepare_linear(m.weight.data, X_by_name[n].to(device),
                                              bits, group_size)
        qk_alphas = _qk_bilinear_alphas(model, lins, preps, device) if (use_snc and qk_snc) else {}
        qk_weights, qk_infos = ({}, {})
        if use_snc and qk_snc and hybrid_guard:
            qk_weights, qk_infos = _select_hybrid_attention_block(
                model, layer, lins, preps, X_by_name, inps, kwargs, qk_alphas,
                bits, group_size, p, lam, beta, device,
                rope_qk_guard=rope_qk_guard)
        elif use_snc and qk_snc and attn_guard:
            qk_weights, qk_infos = _select_attention_block(
                model, layer, lins, preps, inps, kwargs, qk_alphas,
                p, lam, beta, device)
        elif use_snc and qk_snc and rope_qk_guard:
            qk_weights, qk_infos = _select_rope_qk_pair(
                model, layer, lins, preps, inps, kwargs, qk_alphas,
                bits, group_size, p, lam, beta, device)
        elif use_snc and qk_snc:
            qk_weights, qk_infos = _select_qk_pair(
                model, lins, preps, X_by_name, qk_alphas, bits, group_size,
                p, lam, beta, device)

        for n, m in lins.items():
            X = X_by_name[n]
            alpha_sig = alpha_noi = None
            if n in qk_alphas:
                alpha_sig, alpha_noi = qk_alphas[n]
            if n in qk_weights:
                Wq, info = qk_weights[n], qk_infos[n]
            elif n in preps:
                Wq, info = awq_apply_prepared(preps[n], use_snc=use_snc, p=p,
                                              lam=lam, beta=beta,
                                              snc_guard=snc_guard,
                                              alpha_sig=alpha_sig, alpha_noi=alpha_noi)
            else:
                Wq, info = awq_quantize_linear(m.weight.data, X.to(device), bits, group_size,
                                               use_snc=use_snc, p=p, lam=lam, beta=beta,
                                               snc_guard=snc_guard,
                                               alpha_sig=alpha_sig, alpha_noi=alpha_noi)
            if info["snc_accepted"] is not None:
                accepted += int(info["snc_accepted"])
                rejected += int(not info["snc_accepted"])
            m.weight.data = Wq.to(m.weight.dtype)
            caught[n].clear(); del X
        caught.clear(); X_by_name.clear(); preps.clear()

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
    ap.add_argument("--attn-guard", action="store_true",
                    help="choose q/k/v/o candidates by true self-attention output loss")
    ap.add_argument("--hybrid-guard", action="store_true",
                    help="choose q/k by QK logits, then v/o by attention output loss")
    ap.add_argument("--rope-qk-guard", action="store_true",
                    help="choose q/k by RoPE-aware causal QK logit loss")
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
                   qk_snc=args.qk_snc,
                   attn_guard=args.attn_guard,
                   hybrid_guard=args.hybrid_guard,
                   rope_qk_guard=args.rope_qk_guard)
    model.save_pretrained(args.output_dir); tok.save_pretrained(args.output_dir)
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
