"""
SNC core (PyTorch) — one pure function per line of Algorithm 1 of the SNC paper.

Every function is stateless and takes plain tensors so each paper equation can
be unit-tested in isolation. Naming follows the paper:
  mu  = mu_hat (James-Stein mean)      Sigma = activation covariance
  e   = w_q - w (residual)             b = mu^T e_j      s = lattice step s_{j,i}
  alpha_sig / alpha_noi = output-side importances (Eq. 3)

Assumptions kept minimal for trackability:
  * group_size divides in_features (true for all 4 target models at gs=128).
  * adjacent-level flips only: each applied step is +-s_{j,i}.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import torch

Tensor = torch.Tensor


# --- Eq. 13: James-Stein shrinkage of the activation mean -------------------
def james_stein(x_bar: Tensor, variance_estimate: Tensor | None = None) -> Tensor:
    d = x_bar.shape[0]
    if d < 3:
        return x_bar
    grand = x_bar.mean()
    dev = x_bar - grand
    ss = dev.pow(2).sum()
    if ss < 1e-12:
        return x_bar
    if variance_estimate is None:
        variance_estimate = dev.pow(2).mean()
    var = variance_estimate.clamp(min=0.0)       # sigma_hat^2 of the sample mean
    c = ((d - 2) * var / ss).clamp(0.0, 1.0)
    return grand + (1.0 - c) * dev               # mu_hat


# --- Algorithm 1, lines 1-2: block statistics from calibration X ------------
def block_stats(X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """X: (N, d). Returns mu_hat, Sigma, sigma_diag, r (r_i = sum_{k!=i}|Sigma_ik|)."""
    X = X.to(torch.float64) if X.dtype == torch.float64 else X.float()
    N = X.shape[0]
    mu = X.mean(0)
    Sigma = (X.t() @ X) / N - torch.outer(mu, mu)
    # James-Stein observes x_bar, so its noise is Var[x_bar] = Var[x] / N.
    # Using variance across x_bar coordinates would force c ~= (d - 2) / d
    # regardless of calibration quality and collapse almost the entire mean.
    mu = james_stein(mu, torch.diagonal(Sigma).clamp(min=0.0).mean() / N)
    sigma_diag = torch.diagonal(Sigma).clamp(min=0.0)
    A = Sigma.abs()
    r = A.sum(1) - torch.diagonal(A)
    return mu, Sigma, sigma_diag, r


# --- group-wise asymmetric RTN base (the projection SNC corrects) -----------
def rtn_quantize(W: Tensor, bits: int, group_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns W_int, scale_flat, zp_flat (all (out, in)). in must be divisible by gs."""
    out_f, in_f = W.shape
    assert in_f % group_size == 0, "in_features must be divisible by group_size"
    g = in_f // group_size
    Wg = W.reshape(out_f, g, group_size)
    w_min = Wg.min(2, keepdim=True)[0]
    w_max = Wg.max(2, keepdim=True)[0]
    max_int = 2 ** bits - 1
    scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
    zp = torch.round(-w_min / scale).clamp(0, max_int)
    W_int = torch.round(Wg / scale + zp).clamp(0, max_int)
    rep = lambda t: t.repeat(1, 1, group_size).reshape(out_f, in_f)
    return W_int.reshape(out_f, in_f), rep(scale), rep(zp)


def dequant(W_int: Tensor, scale: Tensor, zp: Tensor) -> Tensor:
    return (W_int - zp) * scale


# --- Eq. 3 / Sec. 3.4.1: output-side importances ----------------------------
def alpha_standalone(out_f: int, device, dtype) -> Tuple[Tensor, Tensor]:
    """M = I: alpha_sig = alpha_noi = 1."""
    one = torch.ones(out_f, device=device, dtype=dtype)
    return one, one.clone()


def alpha_from_partner(W_partner: Tensor, mu: Tensor, Sigma: Tensor) -> Tuple[Tensor, Tensor]:
    """Per partner-row j: alpha_sig_j = ((W_p mu)_j)^2, alpha_noi_j = (W_p Sigma W_p^T)_jj.
    GQA aggregation (summing over the Q-heads sharing a K) is left to the caller."""
    Wp = W_partner.float()
    alpha_sig = (Wp @ mu.float()).pow(2)
    alpha_noi = ((Wp @ Sigma.float()) * Wp).sum(1).clamp(min=0.0)
    return alpha_sig, alpha_noi


# --- Eq. 7 gradient indicator, Eq. 8 flip direction -------------------------
def gradient_g(e: Tensor, b: Tensor, mu: Tensor, Sigma: Tensor,
               alpha_sig: Tensor, alpha_noi: Tensor) -> Tensor:
    Sigma_e = e.float() @ Sigma.float()                       # (Sigma e_j)_i
    g = (2.0 * alpha_noi[:, None] * Sigma_e
         + 2.0 * alpha_sig[:, None] * b[:, None] * mu[None, :])
    return g


def flip_direction(g: Tensor) -> Tensor:
    d = -torch.sign(g)
    d[d == 0] = 1.0
    return d


# --- Eq. 5 self-coupling h, benefit filter (Alg. lines 11-12) ---------------
def self_coupling_h(mu: Tensor, sigma_diag: Tensor,
                    alpha_sig: Tensor, alpha_noi: Tensor) -> Tensor:
    return alpha_noi[:, None] * sigma_diag[None, :] + alpha_sig[:, None] * mu[None, :].pow(2)


def benefit_filter(g: Tensor, d: Tensor, h: Tensor, s: Tensor) -> Tensor:
    """Keep candidates with first-order gain over quadratic self-cost: -d g s - h s^2 > 0."""
    return (-d * g * s - h * s.pow(2)) > 0


# --- Eq. 9: signal-to-noise score -------------------------------------------
def snr_score(mu: Tensor, s: Tensor, sigma_diag: Tensor, r: Tensor,
              alpha_sig: Tensor, alpha_noi: Tensor, lam: float, beta: float) -> Tensor:
    num = alpha_sig.sqrt()[:, None] * mu.abs()[None, :] * s.sqrt()
    denom = (mu[None, :].pow(2)
             + lam * alpha_noi[:, None] * (sigma_diag[None, :] + beta * r[None, :]))
    return num / denom.clamp(min=1e-12).sqrt()


# --- Sec. 3.4.2 value-greedy + Prop. 1 (the bug-fixed step) -----------------
def value_greedy(b: Tensor, mu: Tensor, d: Tensor, s: Tensor,
                 snr: Tensor, eligible: Tensor) -> Tensor:
    """Per channel j, accept flips in SNR order minimizing the TRUE new bias
    |b'_j| = |b_j + sum v|, v_{j,i} = mu_i d_{j,i} s_{j,i}. k=0 feasible => |b'|<=|b|
    => G>=0 (Prop. 1). `eligible` already encodes the global top-B budget.
    Returns a boolean flip mask (out, in)."""
    out_f, in_f = d.shape
    v = mu[None, :] * d * s
    order = torch.argsort(torch.where(eligible, snr, snr.new_full((), -math.inf)),
                          dim=1, descending=True)
    elig_s = torch.gather(eligible, 1, order)
    v_s = torch.gather(v, 1, order) * elig_s.to(v.dtype)
    cum = torch.cumsum(v_s, dim=1)
    cand = torch.cat([b.abs()[:, None], (b[:, None] + cum).abs()], dim=1)  # k = 0..in
    k_star = torch.argmin(cand, dim=1)
    pos = torch.arange(in_f, device=d.device)[None, :]
    flip_s = (pos < k_star[:, None]) & elig_s
    mask = torch.zeros_like(eligible)
    mask.scatter_(1, order, flip_s)
    return mask


@dataclass
class MatrixSpec:
    W: Tensor                      # weight in the (AWQ-scaled) space where s applies
    bits: int
    group_size: int
    mu: Tensor
    Sigma: Tensor
    sigma_diag: Tensor
    r: Tensor
    alpha_sig: Tensor
    alpha_noi: Tensor


# --- Algorithm 1, lines 4-23: pool across matrices, global top-B, apply -----
def snc_correct_block(mats: List[MatrixSpec], p: float,
                      lam: float = 1.0, beta: float = 1.0) -> List[dict]:
    """Corrects every matrix in a block jointly. Returns per-matrix dicts with
    'W_corrected', 'n_flips', 'G' (signal-side gain, must be >=0)."""
    per = []
    snr_pool = []
    for m in mats:
        W_int, scale, zp = rtn_quantize(m.W, m.bits, m.group_size)
        e = dequant(W_int, scale, zp) - m.W
        b = (e * m.mu[None, :]).sum(1)
        g = gradient_g(e, b, m.mu, m.Sigma, m.alpha_sig, m.alpha_noi)
        d = flip_direction(g)
        h = self_coupling_h(m.mu, m.sigma_diag, m.alpha_sig, m.alpha_noi)
        max_int = 2 ** m.bits - 1
        in_range = ((W_int + d) >= 0) & ((W_int + d) <= max_int)
        feasible = benefit_filter(g, d, h, scale) & in_range
        snr = snr_score(m.mu, scale, m.sigma_diag, m.r,
                        m.alpha_sig, m.alpha_noi, lam, beta)
        per.append(dict(W_int=W_int, scale=scale, zp=zp, e=e, b=b, d=d,
                        feasible=feasible, snr=snr))
        snr_pool.append(snr[feasible])

    # global candidate pool C and budget B = ceil(p |C|) (Alg. lines 18-19)
    pool = torch.cat(snr_pool) if snr_pool else torch.empty(0)
    thr = -math.inf
    if pool.numel() > 0:
        B = max(1, math.ceil(p * pool.numel()))
        B = min(B, pool.numel())
        thr = torch.sort(pool, descending=True).values[B - 1].item()

    out = []
    for m, s in zip(mats, per):
        eligible = s["feasible"] & (s["snr"] >= thr)
        mask = value_greedy(s["b"], m.mu, s["d"], s["scale"], s["snr"], eligible)
        W_int_new = (s["W_int"] + s["d"] * mask).clamp(0, 2 ** m.bits - 1)
        e_new = dequant(W_int_new, s["scale"], s["zp"]) - m.W
        b_new = (e_new * m.mu[None, :]).sum(1)
        G = (m.alpha_sig * (s["b"].pow(2) - b_new.pow(2))).sum()
        out.append(dict(W_corrected=dequant(W_int_new, s["scale"], s["zp"]),
                        n_flips=int(mask.sum().item()), G=float(G.item())))
    return out
