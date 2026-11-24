# Does ES (Stein / Gaussian smoothing) give a least-action principle?
# Test system: harmonic oscillator path x(t), t in [0,T], x(0)=0, x(T)=1, discretised.
import numpy as np

rng = np.random.default_rng(0)
N, T = 49, 1.0
dt = T / (N + 1)
t = np.linspace(dt, T - dt, N)
K = (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))  # discrete -d2/dt2 * dt^2
b = np.zeros(N); b[-1] = 1.0                            # boundary x(T)=1


def S(x, w, sign):  # sign=+1 Euclidean action, -1 real (Lorentzian-time) action
    xf = np.concatenate([[0.0], x, [1.0]])
    return np.sum(np.diff(xf) ** 2) / (2 * dt) + sign * dt * 0.5 * w**2 * np.sum(x**2)


def hess(w, sign):
    return K / dt + sign * dt * w**2 * np.eye(N)


def stationary(w, sign):  # solve dS/dx = 0 (discrete Euler-Lagrange)
    return np.linalg.solve(hess(w, sign), b / dt)


def S_batch(X, w, sign):
    Xf = np.concatenate([np.zeros((X.shape[0],1)), X, np.ones((X.shape[0],1))], 1)
    return (np.diff(Xf, axis=1)**2).sum(1)/(2*dt) + sign*dt*0.5*w**2*(X**2).sum(1)


def es_minimise(w, sign, steps=20000, sigma=0.05, pop=100):
    lr = 0.9 / np.linalg.eigvalsh(hess(w, sign)).max()
    x = np.linspace(0, 1, N + 2)[1:-1].copy()
    for it in range(steps):
        eps = rng.standard_normal((pop, N))
        f = S_batch(x + sigma*eps, w, sign) - S_batch(x - sigma*eps, w, sign)
        g = (f[:, None] * eps).mean(0) / (2 * sigma)   # Stein / ES gradient estimate (antithetic)
        x -= lr * g
        if not np.isfinite(x).all() or np.abs(x).max() > 1e3:
            return x, it, S(x, w, sign)
    return x, steps, S(x, w, sign)


print("=== (A) ES on the EUCLIDEAN action  S_E = int(1/2 x'^2 + 1/2 w^2 x^2) ===")
w = 2.0
x_es, _, s_es = es_minimise(w, +1)
x_cl = stationary(w, +1)
print(f"ES result vs Euler-Lagrange path: max|diff| = {np.abs(x_es - x_cl).max():.4f}")
print(f"analytic sinh(wt)/sinh(wT) vs discrete EL: max|diff| = {np.abs(np.sinh(w*t)/np.sinh(w*T) - x_cl).max():.4f}")
print(f"S_E(ES) = {s_es:.5f},  S_E(EL) = {S(x_cl, w, +1):.5f}")

print("\n=== (B) ES on the REAL action  S = int(1/2 x'^2 - 1/2 w^2 x^2) ===")
for w in [2.0, 4.0, 7.0]:
    lam = np.linalg.eigvalsh(hess(w, -1))
    neg = (lam < 0).sum()
    x_cl = stationary(w, -1)
    x_es, it, s_es = es_minimise(w, -1)
    status = f"converged, max|diff to EL| = {np.abs(x_es - x_cl).max():.4f}" if it == 20000 else f"DIVERGED at step {it}, S -> {s_es:.3g}"
    print(f"w={w}: wT/pi = {w*T/np.pi:.2f}, negative Hessian modes = {neg}, EL path = sin(wt)/sin(wT) exists; ES: {status}")

print("\n=== (C) Stein's identity for the Gibbs measure exp(-S_E/hbar)  ==  Schwinger-Dyson ===")
w, hbar = 2.0, 0.1
H = hess(w, +1)
mean = stationary(w, +1)
cov = hbar * np.linalg.inv(H)
L = np.linalg.cholesky(cov)
M = 400000
xs = mean + rng.standard_normal((M, N)) @ L.T
dS = (xs - mean) @ H.T                      # dS/dx_i  (S quadratic -> exact gradient)
el = dS.mean(0)
sd = xs.T @ dS / M                         # <x_j dS/dx_i>
print(f"<dS/dx_i> (Euler-Lagrange holds on average): max |.| = {np.abs(el).max():.4f}  (MC noise ~ {np.sqrt(np.diag(H)).max()*np.sqrt(hbar/M)*3:.4f})")
print(f"<x_j dS/dx_i> = hbar*delta_ij ?  diag mean = {np.diag(sd).mean():.4f} (hbar={hbar}), offdiag max = {np.abs(sd - np.diag(np.diag(sd))).max():.4f}")
