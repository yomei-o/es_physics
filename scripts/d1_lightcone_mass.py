# D1/D2: local zero-sum network on a 1D ring.  Agent j controls x_j; coupling only to neighbours.
#   x_j' = J_R(j) x_{j+1} - J_L(j) x_{j-1}       (antisymmetric: A_{j,j+1} = -A_{j+1,j})
# Pre-registered success criteria (written before running):
#  (a) a localised disturbance spreads with a finite maximal speed (light cone) set by J * spacing
#  (b) with alternating ("dimerised") couplings 1+-m, the spectrum opens a gap and near it
#      omega^2 = c^2 k^2 + M^2  (relativistic dispersion), with c and M predicted from J, m
#  (c) the same holds when each agent only uses ES on its own loss (no gradients)
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(5)


def ring_A(N, m=0.0, J=1.0):
    A = np.zeros((N, N))
    for j in range(N):
        h = J * (1 + m if j % 2 == 0 else 1 - m)
        A[j, (j + 1) % N] += h
        A[(j + 1) % N, j] -= h
    return A


# ---- (a) light cone ------------------------------------------------------------
N = 400
A = ring_A(N, 0.0)
x0 = np.zeros(N); x0[N // 2] = 1.0
print("=== (a) light cone: front position of a point disturbance, J=1 ===")
for t in [20, 40, 80]:
    x = expm(A * t) @ x0
    amp = np.abs(x)
    inside = np.where(amp > 1e-6)[0]
    front = max(abs(inside - N // 2))
    beyond = amp[np.abs(np.arange(N) - N // 2) > 1.2 * 2 * t].max(initial=0)  # amplitude beyond 1.2*c*t
    print(f"t={t:3d}: front at {front:4d} sites, front/t = {front/t:.3f}; max amplitude beyond 1.2*c*t (c=2J): {beyond:.1e}")
print("prediction: dispersion omega = 2J sin k  ->  max group velocity c = 2J (sites per unit time)")
for J in [0.5, 1.0, 2.0]:
    x = expm(ring_A(N, 0.0, J) * 40) @ x0
    inside = np.where(np.abs(x) > 1e-6)[0]
    print(f"   J={J}: measured speed {max(abs(inside - N//2))/40:.3f} vs 2J = {2*J:.3f}")

# ---- (b) mass gap + relativistic dispersion --------------------------------------
print("\n=== (b) dimerised couplings 1+-m: gap and dispersion ===")
N = 2000
for m in [0.05, 0.1, 0.2]:
    w = np.sort(np.abs(np.linalg.eigvals(ring_A(N, m)).imag))
    gap = w[0]
    # analytic: unit cell of 2 sites; omega(k) = |(1+m) - (1-m) e^{-ik}| -> at k=pi: 2m ; near k=pi: omega^2 = (2m)^2 + (1-m^2) q^2
    q = np.linspace(-0.3, 0.3, 7)
    omega = np.abs((1 + m) + (1 - m) * np.exp(-1j * (np.pi + q)))
    fitM2, fitc2 = np.polyfit(q**2, omega**2, 1)[::-1]
    print(f"m={m}: numerical gap (min |omega|) = {gap:.4f}, predicted M = 2m = {2*m:.4f}; "
          f"omega^2 = M^2 + c^2 q^2 with M^2={fitM2:.4f} (pred {(2*m)**2:.4f}), c^2={fitc2:.4f} (pred 1-m^2={1-m*m:.4f}), "
          f"residual {np.abs(np.polyval([fitc2, fitM2], q**2) - omega**2).max():.1e}")

# ---- (c) ES agents --------------------------------------------------------------
print("\n=== (c) same network run by ES agents (each perturbs only its own x_j) ===")
N, m, eta, T = 60, 0.2, 0.01, 15.0
A = ring_A(N, m)
x = np.exp(-(np.arange(N) - 30) ** 2 / 8.0) * np.cos(np.pi * np.arange(N))  # packet near k=pi
x /= np.linalg.norm(x); x_init = x.copy()
pop, sig = 64, 1e-2
for step in range(int(T / eta)):
    g = np.zeros(N)
    for j in range(N):
        # agent j's own loss: L_j = -x_j * (A x)_j   (A_jj = 0) ; gradient wrt x_j = -(A x)_j
        eps = rng.standard_normal(pop)
        f = lambda v: -v * (A[j] @ x)
        d = f(x[j] + sig * eps) - f(x[j] - sig * eps)
        g[j] = (d * eps).mean() / (2 * sig)
    x = x - eta * g          # explicit Euler of x' = A x  (simultaneous moves)
exact = expm(A * T) @ x_init
print(f"overlap with exact evolution: {exact @ x / np.linalg.norm(x) / np.linalg.norm(exact):.4f}, |x|^2 = {x @ x:.3f}")
