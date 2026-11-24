# C2: "c * t = const" as a step-size schedule of ES agents.
# A chain of agents; agent 0 is pinned to 1 (a "signal"), the rest start at 0.
# Each agent does ES (Stein estimator, no derivatives) on its local loss
#   l_i(x) = 1/2[(x - th_{i-1})^2 + (x - th_{i+1})^2]
# with step eta_t = eta0 / (1 + t/t0)^p.  eta_t is "how far information moves per tick" = c_t.
# p = 1  <=>  c_t * t = const.
# BP run: same schedule, exact gradient (the N->inf limit of ES = BP).
# PRE-REGISTERED:
#   reach^2 ∝ sum eta:  p=0 ~ sqrt(t), p=0.5 ~ t^1/4, p=1 ~ sqrt(ln t), p=2 saturates.
#   p = 1 is the fastest decay for which the reach still diverges (critical case).
#   final ES noise (rms ES-BP) decreases with p.
import numpy as np

N, T, eta0, t0, sigma, M = 600, 100_000, 0.4, 10.0, 0.1, 4
checks = [100, 1_000, 10_000, 100_000]


def run(p, es, seed=0):
    rng = np.random.default_rng(seed)
    th = np.zeros(N); th[0] = 1.0
    out = {}
    for t in range(1, T + 1):
        eta = eta0 / (1 + t / t0) ** p
        left, right = th[:-2], th[2:]
        x = th[1:-1]
        g_true = 2 * x - left - right
        if es:
            eps = rng.standard_normal((N - 2, M))
            xp = x[:, None] + sigma * eps
            lp = 0.5 * ((xp - left[:, None]) ** 2 + (xp - right[:, None]) ** 2)
            l0 = 0.5 * ((x - left) ** 2 + (x - right) ** 2)
            g = ((lp - l0[:, None]) * eps).mean(1) / sigma
        else:
            g = g_true
        th[1:-1] = x - eta * g
        th[-1] = th[-2]
        if t in checks:
            out[t] = th.copy()
    return out


def reach(th, thr=0.05):
    idx = np.where(th > thr)[0]
    return idx.max() if len(idx) else 0


print(f"N={N} eta0={eta0} t0={t0} sigma={sigma} M={M}")
print(f"{'p':>4} | " + " | ".join(f"t={c:>6}: reach  noise" for c in checks) + " | sum eta")
for p in [0.0, 0.5, 1.0, 1.5, 2.0]:
    bp, es = run(p, False), run(p, True)
    s = np.sum(eta0 / (1 + np.arange(1, T + 1) / t0) ** p)
    cells = []
    for c in checks:
        r = reach(bp[c])
        noise = np.sqrt(np.mean((es[c][: max(r, 1)] - bp[c][: max(r, 1)]) ** 2))
        cells.append(f"{r:>13d} {noise:6.4f}")
    print(f"{p:>4} | " + " | ".join(cells) + f" | {s:9.1f}")

# reach growth law check: reach^2 / sum(eta) should be roughly constant across t and p
print("\nreach^2 / sum_eta(t)  (diffusive law check):")
for p in [0.0, 0.5, 1.0, 2.0]:
    bp = run(p, False)
    vals = []
    for c in checks:
        s = np.sum(eta0 / (1 + np.arange(1, c + 1) / t0) ** p)
        vals.append(reach(bp[c]) ** 2 / s)
    print(f"  p={p}: " + "  ".join(f"{v:6.2f}" for v in vals))
