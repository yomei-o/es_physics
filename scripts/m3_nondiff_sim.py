# M3: gradients of a non-differentiable Monte Carlo (1D rod-model particle transport through a slab).
# Particles enter at x=0 moving +x. Free path ~ Exp(mu). At each collision: absorbed with prob pa, else direction = +-1.
# Detector counts transmission (x > d): an indicator -> pathwise derivative is 0 almost everywhere.
# PRE-REGISTERED (es_methods_notes.html M0, M3).
import numpy as np

rng = np.random.default_rng(5)
mu = 1.0


def transmit(d, n, pa, rng):
    x = np.zeros(n); dirn = np.ones(n); alive = np.ones(n, bool); out = np.zeros(n)
    while alive.any():
        step = rng.exponential(1 / mu, n)
        x = np.where(alive, x + dirn * step, x)
        passed = alive & (x > d); refl = alive & (x < 0)
        out[passed] = 1; alive &= ~(passed | refl)
        absorb = alive & (rng.random(n) < pa)
        alive &= ~absorb
        dirn = np.where(alive, np.where(rng.random(n) < 0.5, 1.0, -1.0), dirn)
    return out


d0 = 1.5
print("=== pure absorption (pa=1): analytic P = exp(-mu d), dP/dd = -mu exp(-mu d) ===")
true = -mu * np.exp(-mu * d0)
print(f"analytic gradient {true:.5f}")
print("pathwise derivative of the indicator: 0 for every sampled path (step function) -> estimate 0.00000  (wrong)")
budget = 200000
for name in ["FD (no CRN)", "FD (CRN)", "ES sigma=0.1 (antithetic)", "ES sigma=0.3 (antithetic)"]:
    ests = []
    for rep in range(20):
        if name.startswith("FD"):
            h = 0.05; n = budget // 2
            if "no CRN" in name:
                g = (transmit(d0 + h, n, 1.0, rng).mean() - transmit(d0 - h, n, 1.0, rng).mean()) / (2 * h)
            else:
                s = rng.integers(1 << 30)
                g = (transmit(d0 + h, n, 1.0, np.random.default_rng(s)).mean() - transmit(d0 - h, n, 1.0, np.random.default_rng(s)).mean()) / (2 * h)
        else:
            sig = 0.1 if "0.1" in name else 0.3
            n = budget // 2
            eps = rng.normal(size=n)
            # one simulated particle per perturbed thickness (vectorised per-particle thickness)
            def tr_vec(dv):
                x = np.zeros(n); dirn = np.ones(n); alive = np.ones(n, bool); out = np.zeros(n)
                while alive.any():
                    x = np.where(alive, x + dirn * rng.exponential(1 / mu, n), x)
                    passed = alive & (x > dv); refl = alive & (x < 0)
                    out[passed] = 1; alive &= ~(passed | refl)
                    alive &= ~(alive & (rng.random(n) < 1.0))
                return out
            g = np.mean((tr_vec(d0 + sig * eps) - tr_vec(d0 - sig * eps)) * eps) / (2 * sig)
        ests.append(g)
    smoothed = true * np.exp(0.5 * mu**2 * (0.1 if "0.1" in name else 0.3) ** 2) if name.startswith("ES") else true
    print(f"{name:26s}: mean {np.mean(ests):+.5f}, spread {np.std(ests):.5f} | target {smoothed:+.5f} "
          f"({'smoothed objective' if name.startswith('ES') else 'exact'})")

print("\n=== high-dimensional design: 20 layers with thicknesses d_k (total transmission), gradient vector ===")
K = 20; dk = np.full(K, 0.075); mus = np.linspace(0.5, 1.5, K)    # absorption only, layer k has mu_k
true_g = -mus * np.exp(-np.sum(mus * dk))
P = lambda dv: np.exp(-np.sum(mus * dv, axis=-1))                    # exact objective (for reference only)


def sim_P(dmat, rng):  # one particle per row of thickness matrix: survive all layers
    return (rng.random(dmat.shape) < np.exp(-mus * dmat)).all(axis=1).astype(float)


for name, budget in [("FD per coordinate (2K sims)", 40000), ("ES all coordinates at once", 40000)]:
    errs = []
    for rep in range(20):
        if name.startswith("FD"):
            n = budget // (2 * K); g = np.zeros(K); h = 0.02
            for k in range(K):
                e = np.zeros(K); e[k] = h
                g[k] = (sim_P(np.tile(dk + e, (n, 1)), rng).mean() - sim_P(np.tile(dk - e, (n, 1)), rng).mean()) / (2 * h)
        else:
            n = budget // 2; sig = 0.02
            E = rng.normal(size=(n, K))
            g = ((sim_P(dk + sig * E, rng) - sim_P(dk - sig * E, rng))[:, None] * E).mean(0) / (2 * sig)
        errs.append(np.linalg.norm(g - true_g) / np.linalg.norm(true_g))
    print(f"{name:30s}: relative error of gradient vector {np.mean(errs):.3f} (same budget {budget} particle histories)")
