# C5 (E2): freeze-out spectrum of ES noise for p > 1 (accelerating side).
# Modes l(x) = lam x^2/2 start at x=0; step eta = eta0 / (t + t0)^p.  Rate per e-fold  eta t lam  falls,
# so each mode stops tracking the noise and freezes (analogue of horizon exit).
# Exact scaling (t0 -> 0):  V_inf = G eta0^2 ∫ t^{-2p} exp(-2 lam eta0 t^{1-p}/(p-1)) dt  ∝  G lam^{-2-1/(p-1)}.
# ES noise at x≈0: G = 15 lam^2 sigma^2 / (4M)  =>  V_inf ∝ lam^{-1/(p-1)}.
# PRE-REGISTERED: slope d ln V / d ln lam = -2 (p=1.5), -1.5 (p=5/3), -1 (p=2).
#   Scale invariant in 3D (V ∝ k^-3, lam = k^2) at p = 5/3.
# usage: python c5_freeze_spectrum.py p eta0 t0 T
import sys
import numpy as np

p, eta0, t0, T = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(int(p * 100))
sigma, M, R = 0.1, 8, 400
lams = np.array([0.4, 0.6, 0.9, 1.3, 2.0, 3.0])
x = np.zeros((R, len(lams)))
half = None
for t in range(1, T + 1):
    eta = eta0 / (t + t0) ** p
    eps = rng.standard_normal((R, len(lams), M))
    xp = x[..., None] + sigma * eps
    g = ((0.5 * lams[None, :, None] * (xp ** 2 - x[..., None] ** 2)) * eps).mean(-1) / sigma
    x = x - eta * g
    if t == T // 2:
        half = (x ** 2).mean(0)
V = (x ** 2).mean(0)
err = V * np.sqrt(2 / R)
tk = (2 * eta0 * lams / (p - 1)) ** (1 / (p - 1))
slope, icpt = np.polyfit(np.log(lams), np.log(V), 1)
print(f"p={p} eta0={eta0} t0={t0} T={T}  eta(0)*lam_max={eta0 / t0**p * lams.max():.2f}")
for l, v, h, e, k in zip(lams, V, half, err, tk):
    print(f"  lam={l:4.1f}: V={v:.3e} ±{e:.1e}  V(T/2)/V(T)={h / v:.3f}  freeze time ~{k:9.0f}")
print(f"  fitted slope = {slope:.3f}, predicted = {-1 / (p - 1):.3f}")
