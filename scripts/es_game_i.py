# Where does "i" come from?  Two ES populations q, p (psi = q + i p) each minimise THEIR OWN loss.
#   L_q = q.H.p + (a/2) q.H.q        L_p = -q.H.p + (a/2) p.H.p
# competition term (q.H.p, zero-sum)  +  cooperation term (a/2)(q.H.q + p.H.p, shared energy)
# Exact gradient flow:  psi' = (i - a) H psi   ->  a=0: Schroedinger (real time), a->inf: imaginary time.
# Here gradients are NOT used: each population only evaluates its own loss at random perturbations (ES / Stein).
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(1)
N = 16
H = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)          # particle on a chain (discrete Laplacian)
evals, evecs = np.linalg.eigh(H)
j = np.arange(N)
psi0 = np.exp(-(j - 5) ** 2 / 6.0) * np.exp(1j * 1.2 * j)      # moving wave packet
psi0 /= np.linalg.norm(psi0)


def Lq(q, p, a): return q @ H @ p + 0.5 * a * q @ H @ q
def Lp(q, p, a): return -q @ H @ p + 0.5 * a * p @ H @ p


def es_grad(f, x, pop, sigma=1e-2):
    eps = rng.standard_normal((pop, N))
    d = np.array([f(x + sigma * e) - f(x - sigma * e) for e in eps])
    return (d[:, None] * eps).mean(0) / (2 * sigma)


def run(a, pop, comp=1.0, T=12.0, eta=0.005):
    q, p = psi0.real.copy(), psi0.imag.copy()
    lq = lambda q, p, a: comp * q @ H @ p + 0.5 * a * q @ H @ q
    lp = lambda q, p, a: -comp * q @ H @ p + 0.5 * a * p @ H @ p
    for _ in range(int(T / eta)):                                  # players move in turn (alternating)
        q = q - eta * es_grad(lambda x: lq(x, p, a), q, pop)
        p = p - eta * es_grad(lambda x: lp(q, x, a), p, pop)
    return q + 1j * p


T = 12.0
print("=== 1. pure competition (a=0): does ES produce Schroedinger evolution exp(iHt)? ===")
exact = expm(1j * H * T) @ psi0
for pop in [8, 32, 128, 512]:
    psi = run(0.0, pop, T=T)
    fid = abs(np.vdot(exact, psi)) / np.linalg.norm(psi)
    print(f"pop={pop:4d}: |psi|^2 = {np.linalg.norm(psi)**2:.3f} (exact 1), fidelity with exp(iHT)psi0 = {fid:.4f}")

print("\n=== 2. pure cooperation (no game term): imaginary time -> ground state ===")
psi = run(1.0, 128, comp=0.0, T=T)
print(f"|psi|^2 = {np.linalg.norm(psi)**2:.2e} (decays), overlap with ground state = "
      f"{abs(np.vdot(evecs[:,0], psi))/np.linalg.norm(psi):.4f}  (initial {abs(np.vdot(evecs[:,0], psi0)):.4f})")

print("\n=== 3. mixed: measured Wick-rotation angle vs predicted arctan(a) ===")
for a in [0.0, 0.25, 1.0, 4.0]:
    Ts = 3.0
    psi = run(a, 512, T=Ts)
    # per-eigenmode complex growth rate  c_k = log(psi_k(T)/psi_k(0))/T  should equal (i - a) * E_k
    k = 3
    c = np.log(np.vdot(evecs[:, k], psi) / np.vdot(evecs[:, k], psi0)) / Ts
    ang = np.degrees(np.arctan2(-c.real, c.imag))
    print(f"a={a:4.2f}: mode {k}: decay rate {-c.real:.3f} (pred {a*evals[k]:.3f}), "
          f"frequency {c.imag:.3f} (pred {evals[k]:.3f}), angle {ang:5.1f} deg (pred {np.degrees(np.arctan(a)):5.1f})")
