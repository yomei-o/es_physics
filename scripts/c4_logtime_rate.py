# C4 (E1): the c*t = const schedule seen in log time s = ln t.
# ES (Stein estimator, no derivatives) on independent quadratic modes l(x) = lam x^2 / 2,
# step eta = eta0 / (t + t0)  (c * t = const).  In s = ln t: d x/ds = -eta0 lam x + noise whose power falls as e^{-s}.
# Two exponential rates compete: relaxation 2 eta0 lam per e-fold vs cooling 1 per e-fold.
# PRE-REGISTERED:
#   E[x^2] ∝ t^{-min(1, 2 eta0 lam)}: sharp switch at eta0 lam = 1/2.
#   eta0 lam > 1/2 ("thermal"): t E[x^2] -> eta0^2 G / (2 eta0 lam - 1),  G = 15 lam^2 sigma^2 / (4 M)  (ES noise at x=0)
#   eta0 lam < 1/2 ("relic"):   E[x^2] ≈ x0^2 (t/t0)^{-2 eta0 lam}  -> remembers the initial condition.
import numpy as np

rng = np.random.default_rng(1)
eta0, t0, sigma, M, R, T = 1.0, 10.0, 0.1, 8, 400, 100_000
lams = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 2.0, 4.0])
x = np.ones((R, len(lams)))
checks = {1_000: None, 10_000: None, 100_000: None}
for t in range(1, T + 1):
    eta = eta0 / (t + t0)
    eps = rng.standard_normal((R, len(lams), M))
    xp = x[..., None] + sigma * eps
    g = ((0.5 * lams[None, :, None] * (xp ** 2 - x[..., None] ** 2)) * eps).mean(-1) / sigma
    x = x - eta * g
    if t in checks:
        checks[t] = (x ** 2).mean(0)

E3, E5 = checks[1_000], checks[100_000]
slope = np.log(E5 / E3) / np.log((1e5 + t0) / (1e3 + t0))
print(f"eta0={eta0} sigma={sigma} M={M} runs={R}")
print(f"{'eta0*lam':>8} {'slope':>8} {'pred':>6}   {'t*E[x2] @1e5':>13} {'pred(thermal)':>13}   {'E[x2]@1e5':>10} {'pred(relic)':>11}")
for i, lam in enumerate(lams):
    r = eta0 * lam
    pred = -min(1.0, 2 * r)
    G = 15 * lam ** 2 * sigma ** 2 / (4 * M)
    th = eta0 ** 2 * G / (2 * r - 1) if r > 0.5 else float('nan')
    relic = ((1e5 + t0) / t0) ** (-2 * r)
    print(f"{r:8.2f} {slope[i]:8.3f} {pred:6.2f}   {(1e5 + t0) * E5[i]:13.3e} {th:13.3e}   {E5[i]:10.3e} {relic:11.3e}")
