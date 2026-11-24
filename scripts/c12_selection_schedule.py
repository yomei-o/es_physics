# C12 (H11): which "schedule" does natural selection run?
# Wright-Fisher population, Gaussian stabilizing selection w(x) = exp(-s x^2/2), optimum 0, start mean 1, variance V0 = 1.
# Mean moves by  dmean = -s V/(1+sV) * mean  (Gaussian selection = Bayesian update: 1/V' = 1/V + s).
# Effective step size eta_t = s V_t/(1 + s V_t)  (the "c" of the population).
# PRE-REGISTERED:
#   (a) no mutation, N large: V_t = V0/(1 + s V0 t) -> eta ∝ 1/t : p = 1  (c t = const)
#   (b) no mutation, finite N: extra drift loss V -> V(1 - 1/N) per generation -> for t >~ N exponential freezing (p -> inf)
#   (c) mutation variance m per generation: V* solves V* = V*/(1+sV*) + m  -> V* ≈ sqrt(m/s) : p -> 0
import numpy as np

rng = np.random.default_rng(12)
s, T = 0.05, 3000


def run(N, m):
    x = 1 + rng.standard_normal(N)
    Vs = []
    for t in range(1, T + 1):
        w = np.exp(-0.5 * s * x ** 2)
        par = rng.choice(N, size=N, p=w / w.sum())
        x = x[par] + (np.sqrt(m) * rng.standard_normal(N) if m > 0 else 0)
        Vs.append(x.var())
    return np.array(Vs)


def local_p(V, t1, t2):
    eta = s * V / (1 + s * V)
    return -np.log(eta[t2 - 1] / eta[t1 - 1]) / np.log(t2 / t1)


for N, m, label in [(200_000, 0.0, "(a) N=2e5, no mutation"), (2_000, 0.0, "(b) N=2e3, no mutation"), (200_000, 1e-4, "(c) N=2e5, mutation m=1e-4")]:
    V = run(N, m)
    pred_a = 1 / (1 + s * np.arange(1, T + 1))
    print(label)
    for t in [10, 100, 300, 1000, 3000]:
        print(f"   t={t:5d}: V={V[t-1]:.3e}  (selection-only prediction {pred_a[t-1]:.3e})")
    print(f"   local p: [30,300] {local_p(V,30,300):.3f}   [300,3000] {local_p(V,300,3000):.3f}")
    if m > 0:
        print(f"   V* predicted sqrt(m/s) ~ {np.sqrt(m/s):.3e}  (exact root {(m + np.sqrt(m*m + 4*m/s))/2:.3e})")
