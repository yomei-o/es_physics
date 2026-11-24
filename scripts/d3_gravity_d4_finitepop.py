# D3: "slow clock attracts?"  Couplings J(x) vary in space (local clock rate). Dimerisation ratio m fixed,
#     so local light speed c ~ J(x) and local rest frequency M = 2 m J(x) both scale with the clock.
#     Pre-registered prediction (Hamilton eqs with E = J(x) sqrt(p^2 + (2m)^2)):  packet at rest accelerates
#     toward LOWER J with a = -J J'.   Both positive- and negative-frequency packets should fall the same way.
# D4: finite ES population.  Agent j's loss is linear in its own x_j, so the antithetic ES estimate is
#     exact gradient * (chi^2_pop / pop): a random per-agent time step with variance 2/pop ("clock jitter").
#     Pre-registered law: norm drift after time T  ~  eta * T * <|Ax|^2>/|x|^2 * (1 + 2/pop)   (vanishes as eta->0)
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(7)


def chain_A(Js, m):
    N = len(Js)
    A = np.zeros((N, N))
    for j in range(N - 1):
        h = Js[j] * (1 + m if j % 2 == 0 else 1 - m)
        A[j, j + 1] += h
        A[j + 1, j] -= h
    return A


print("=== D3: does a packet at rest fall toward the slow-clock region? ===")
N, m = 800, 0.15
pos = np.arange(N)
for g in [+4e-4, -4e-4]:
    Js = 1 + g * (pos - N / 2)                      # clock rate gradient
    A = chain_A(Js, m)
    lam, V = np.linalg.eig(A)
    x0 = np.exp(-(pos - N / 2) ** 2 / (2 * 30.0**2)) * np.ones(N)   # at rest: with antisymmetric couplings the gap (rest mass) sits at cell momentum K=0
    coef = np.linalg.solve(V, x0.astype(complex))
    for branch, sel in [("positive freq", lam.imag > 0), ("negative freq", lam.imag < 0)]:
        c = np.where(sel, coef, 0)
        cents = []
        ts = [0, 20, 40, 60]
        for t in ts:
            psi = V @ (c * np.exp(lam * t))
            p = np.abs(psi) ** 2
            cents.append((pos * p).sum() / p.sum())
        d = np.array(cents) - cents[0]
        a_fit = 2 * np.polyfit(ts, d, 2)[0]
        print(f"J' = {g:+.0e}: {branch}: centroid shift at t=20,40,60 = {d[1]:+.2f},{d[2]:+.2f},{d[3]:+.2f} sites; "
              f"accel {a_fit:+.2e}, predicted -J J' = {-g:+.2e}  -> moves toward {'LOWER' if np.sign(d[3]) == -np.sign(g) else 'HIGHER'} J")

print("\n=== D4: finite population -> norm drift law ===")
N, m, T = 60, 0.2, 10.0
Js = np.ones(N)
A = chain_A(Js, m); A[N - 1, 0] = 1 - m; A[0, N - 1] = -(1 - m)   # ring
x_init = rng.standard_normal(N); x_init /= np.linalg.norm(x_init)
w2 = np.mean([np.linalg.norm(expm(A * t) @ x_init @ A.T) ** 2 for t in np.linspace(0, T, 30)])
for eta in [0.01, 0.002]:
    for pop in [4, 16, 64, 10**9]:
        runs = []
        for r in range(8):
            x = x_init.copy()
            for _ in range(int(T / eta)):
                k = rng.chisquare(pop, N) / pop if pop < 10**9 else np.ones(N)   # exact ES estimate factor
                x = x + eta * k * (A @ x)
            runs.append(x @ x - 1)
        pred = eta * T * w2 * (1 + 2 / pop)
        print(f"eta={eta:<6} pop={pop if pop < 10**9 else 'inf':>4}: |x|^2-1 = {np.mean(runs):.4f} +- {np.std(runs):.4f}, "
              f"law eta*T*<|Ax|^2>*(1+2/pop) = {pred:.4f}")
