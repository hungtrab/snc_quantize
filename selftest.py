"""
Self-test for the SNC core.

  (1) Proposition 1: the value-greedy step never increases the signal-side bias,
      so G >= 0 and |b'_j| <= |b_j| for every channel. This is the guarantee the
      old code violated (it minimized |b - cum| instead of |b + cum|).
  (2) SNC decreases the empirical standalone reconstruction error on synthetic
      blocks. This catches over-shrinking mu_hat in the James-Stein estimator.
  (3) torch <-> JAX numeric parity on the same synthetic block (skipped if JAX
      is not installed).

Run:  python selftest.py
"""
import math
import numpy as np
import torch

import snc_core as ct


def make_block(seed=0, out_f=64, in_f=256, bits=4, gs=128):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out_f, in_f, generator=g)
    X = torch.randn(4096, in_f, generator=g)
    X[:, torch.arange(0, in_f, 17)] += 3.0          # non-zero mean -> real bias
    mu, Sigma, sd, r = ct.block_stats(X)
    asig, anoi = ct.alpha_standalone(out_f, W.device, W.dtype)
    return ct.MatrixSpec(W, bits, gs, mu, Sigma, sd, r, asig, anoi)


def test_prop1():
    print("[1] Proposition 1 (G >= 0, |b'| <= |b|)")
    ok = True
    for seed in range(5):
        m = make_block(seed)
        # recompute base bias b for the assertion
        W_int, scale, zp = ct.rtn_quantize(m.W, m.bits, m.group_size)
        e = ct.dequant(W_int, scale, zp) - m.W
        b = (e * m.mu[None, :]).sum(1)
        res = ct.snc_correct_block([m], p=0.1)[0]
        # |b'| <= |b| per channel
        e2 = res["W_corrected"] - m.W
        # b' uses the SAME mu; reconstruct from corrected weight on the lattice
        W_int2 = torch.round((res["W_corrected"] / scale) + zp).clamp(0, 2 ** m.bits - 1)
        e2 = ct.dequant(W_int2, scale, zp) - m.W
        b2 = (e2 * m.mu[None, :]).sum(1)
        viol = int((b2.abs() > b.abs() + 1e-5).sum())
        G_ok = res["G"] >= -1e-6
        print(f"  seed={seed}: flips={res['n_flips']:4d}  G={res['G']:+.4e}  "
              f"|b'|>|b| violations={viol}")
        ok = ok and viol == 0 and G_ok
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


def test_reconstruction_descent():
    print("[2] standalone reconstruction descent")
    ok = True
    for seed in range(5):
        m = make_block(seed)
        W_int, scale, zp = ct.rtn_quantize(m.W, m.bits, m.group_size)
        W_base = ct.dequant(W_int, scale, zp)
        W_new = ct.snc_correct_block([m], p=0.05)[0]["W_corrected"]
        g = torch.Generator().manual_seed(seed)
        _ = torch.randn(m.W.shape, generator=g)
        X = torch.randn(4096, m.W.shape[1], generator=g)
        X[:, torch.arange(0, m.W.shape[1], 17)] += 3.0
        R_base = (X @ (W_base - m.W).t()).pow(2).mean()
        R_new = (X @ (W_new - m.W).t()).pow(2).mean()
        descent = R_new <= R_base
        print(f"  seed={seed}: R={R_base:.6f} -> {R_new:.6f}")
        ok = ok and bool(descent)
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


def test_parity():
    print("[3] torch <-> JAX parity")
    try:
        import jax.numpy as jnp
        import core_jax as cj
    except Exception as ex:
        print(f"  SKIP (JAX unavailable: {ex})")
        return True
    m = make_block(0)
    rt = ct.snc_correct_block([m], p=0.1)[0]
    mj = cj.MatrixSpec(jnp.asarray(m.W.numpy()), m.bits, m.group_size,
                       jnp.asarray(m.mu.numpy()), jnp.asarray(m.Sigma.numpy()),
                       jnp.asarray(m.sigma_diag.numpy()), jnp.asarray(m.r.numpy()),
                       jnp.asarray(m.alpha_sig.numpy()), jnp.asarray(m.alpha_noi.numpy()))
    rj = cj.snc_correct_block([mj], p=0.1)[0]
    dW = float(np.abs(rt["W_corrected"].numpy() - np.asarray(rj["W_corrected"])).max())
    print(f"  n_flips torch={rt['n_flips']} jax={rj['n_flips']}  "
          f"max|dW|={dW:.2e}  G torch={rt['G']:+.3e} jax={rj['G']:+.3e}")
    ok = dW < 1e-4 and rt["n_flips"] == rj["n_flips"]
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


if __name__ == "__main__":
    a = test_prop1()
    b = test_reconstruction_descent()
    c = test_parity()
    raise SystemExit(0 if (a and b and c) else 1)
