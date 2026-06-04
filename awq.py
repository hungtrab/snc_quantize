"""AWQ (standard AutoAWQ recipe) with optional SNC.

Per linear:
  1. scale search: s = (x_mean^r / w_mean^(1-r)), normalized, r grid-searched
     on output MSE  (x_mean = E|X|, w_mean = mean of group-normalized |W|).
  2. weight clip search: per (out-channel, group) shrink of |W| range,
     chosen by output MSE  (AutoAWQ _search_best_clip).
  3. quantize the scaled+clipped weight; if use_snc, run the verified SNC
     correction in the scaled space, else plain group-wise asym RTN.
"""
import torch
import snc_core as C


@torch.no_grad()
def _reconstruction_mse(X, W_error, token_chunk=128):
    """Mean output reconstruction error without materializing all token outputs."""
    total = X.new_zeros(())
    n = 0
    for i in range(0, X.shape[0], token_chunk):
        out_error = X[i:i + token_chunk] @ W_error.t()
        total += out_error.pow(2).sum()
        n += out_error.numel()
    return (total / n).item()


@torch.no_grad()
def _q_group(Wg, bits):
    """Group-wise asym quant-dequant; Wg: (..., group_size)."""
    mx = Wg.amax(-1, keepdim=True); mn = Wg.amin(-1, keepdim=True)
    maxi = 2 ** bits - 1
    sc = ((mx - mn) / maxi).clamp(min=1e-8)
    zp = torch.round(-mn / sc).clamp(0, maxi)
    return (torch.round(Wg / sc + zp).clamp(0, maxi) - zp) * sc


@torch.no_grad()
def _scale_search(W, X, bits, group_size, n_grid, n_mse=512):
    out_f, in_f = W.shape
    g = in_f // group_size
    x_mean = X.abs().mean(0)                                   # E|X|  [in]
    Wg = W.view(out_f, g, group_size)
    w_scale = Wg.abs() / (Wg.abs().amax(-1, keepdim=True) + 1e-6)
    w_mean = w_scale.view(out_f, in_f).mean(0)                 # [in]
    idx = torch.randperm(X.shape[0])[:n_mse]
    Xs = X[idx]; Y = Xs @ W.t()
    best = (float("inf"), None, None)
    for gi in range(n_grid + 1):
        r = gi / n_grid
        s = (x_mean.pow(r) / (w_mean.pow(1 - r) + 1e-4)).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        Wq = C.dequant(*C.rtn_quantize(W * s, bits, group_size)) / s
        err = (Y - Xs @ Wq.t()).pow(2).mean().item()
        if err < best[0]:
            best = (err, r, s.clone())
    return best[2], best[1]


@torch.no_grad()
def _clip_search(Ws, X_scaled, bits, group_size, n_grid=20, max_shrink=0.5,
                 n_tok=128, out_chunk=128):
    """Search per-(out,group) clip on the scaled weight Ws by output MSE.
    Returns clamped Ws (same shape)."""
    out_f, in_f = Ws.shape
    g = in_f // group_size
    idx = torch.randperm(X_scaled.shape[0])[:n_tok]
    inp = X_scaled[idx].view(-1, g, group_size)               # [t, g, gs]
    Wsg = Ws.view(out_f, g, group_size)
    org_max = Wsg.abs().amax(-1, keepdim=True)                # [out, g, 1]
    best_max = org_max.clone()
    for ci in range(0, out_f, out_chunk):
        w = Wsg[ci:ci + out_chunk]                            # [oc, g, gs]
        omax = org_max[ci:ci + out_chunk]
        org_out = torch.einsum('tgk,ogk->otg', inp, w)        # [oc, t, g]
        best_err = torch.full((w.shape[0], g), float("inf"), device=Ws.device)
        bmax = omax.clone()
        steps = max(1, int(n_grid * max_shrink))
        for i in range(steps):
            mv = omax * (1 - i / n_grid)                      # [oc, g, 1]
            q = _q_group(torch.clamp(w, -mv, mv), bits)
            cur = torch.einsum('tgk,ogk->otg', inp, q)
            err = (cur - org_out).pow(2).mean(1)              # [oc, g]
            upd = err < best_err
            best_err = torch.where(upd, err, best_err)
            bmax = torch.where(upd.unsqueeze(-1), mv, bmax)
        best_max[ci:ci + out_chunk] = bmax
    return torch.clamp(Wsg, -best_max, best_max).view(out_f, in_f)


@torch.no_grad()
def awq_prepare_linear(W, X, bits=4, group_size=128, n_grid=20, clip=True):
    """Run AWQ scale/clip search and precompute the scaled SNC statistics."""
    dev = W.device
    W = W.float(); X = X.float().to(dev)
    mu, Sigma, _, _ = C.block_stats(X)

    scales, alpha = _scale_search(W, X, bits, group_size, n_grid)
    Ws = W * scales
    W_ref_s = Ws
    if clip:
        Ws = _clip_search(Ws, X / scales, bits, group_size, n_grid)

    inv = 1.0 / scales
    Sigma_s = Sigma * inv[None, :] * inv[:, None]
    A = Sigma_s.abs()
    return {
        "W": W,
        "X": X,
        "bits": bits,
        "group_size": group_size,
        "scales": scales,
        "alpha": alpha,
        "Ws": Ws,
        "W_ref_s": W_ref_s,
        "mu": mu,
        "Sigma": Sigma,
        "mu_s": mu / scales,
        "Sigma_s": Sigma_s,
        "sd_s": torch.diagonal(Sigma_s).clamp(min=0.0),
        "r_s": A.sum(1) - torch.diagonal(A),
    }


@torch.no_grad()
def awq_apply_prepared(prep, use_snc=True, p=0.05, lam=1.0, beta=1.0,
                       snc_guard=True, alpha_sig=None, alpha_noi=None):
    """Quantize a prepared AWQ linear, optionally applying SNC in scaled space."""
    W = prep["W"]
    X = prep["X"]
    bits = prep["bits"]
    group_size = prep["group_size"]
    scales = prep["scales"]
    Ws = prep["Ws"]
    W_ref_s = prep["W_ref_s"]
    dev = W.device
    out_f = W.shape[0]
    if use_snc:
        if alpha_sig is None or alpha_noi is None:
            asig, anoi = C.alpha_standalone(out_f, dev, torch.float32)
        else:
            asig = alpha_sig.to(device=dev, dtype=torch.float32)
            anoi = alpha_noi.to(device=dev, dtype=torch.float32)
        spec = C.MatrixSpec(Ws, bits, group_size, prep["mu_s"], prep["Sigma_s"],
                            prep["sd_s"], prep["r_s"], asig, anoi)
        res = C.snc_correct_block([spec], p=p, lam=lam, beta=beta)[0]
        W_base_s = C.dequant(*C.rtn_quantize(Ws, bits, group_size))
        W_snc_s = res["W_corrected"]
        accepted = True
        base_mse = snc_mse = None
        if snc_guard:
            X_scaled = X / scales
            base_mse = _reconstruction_mse(X_scaled, W_base_s - W_ref_s)
            snc_mse = _reconstruction_mse(X_scaled, W_snc_s - W_ref_s)
            accepted = snc_mse <= base_mse
        Wq = (W_snc_s if accepted else W_base_s) / scales
        info = {"alpha": prep["alpha"], "n_flips": res["n_flips"] if accepted else 0,
                "G": res["G"] if accepted else 0.0, "snc_accepted": accepted,
                "base_mse": base_mse, "snc_mse": snc_mse}
    else:
        Wq = C.dequant(*C.rtn_quantize(Ws, bits, group_size)) / scales
        info = {"alpha": prep["alpha"], "n_flips": 0, "G": 0.0, "snc_accepted": None,
                "base_mse": None, "snc_mse": None}
    return Wq, info


@torch.no_grad()
def awq_quantize_linear(W, X, bits=4, group_size=128, n_grid=20,
                        use_snc=True, p=0.05, lam=1.0, beta=1.0, clip=True,
                        snc_guard=True, alpha_sig=None, alpha_noi=None):
    """W: (out, in) fp. X: (N, in) calibration input. Returns fake-quant W, info."""
    prep = awq_prepare_linear(W, X, bits, group_size, n_grid, clip)
    return awq_apply_prepared(prep, use_snc=use_snc, p=p, lam=lam, beta=beta,
                              snc_guard=snc_guard,
                              alpha_sig=alpha_sig, alpha_noi=alpha_noi)
