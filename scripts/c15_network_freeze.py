# C15: freeze-out spectrum in an actual ES NETWORK (periodic chain), not independent modes.
# Each agent does ES on its local loss, so the gradient noise is white across Fourier modes (G = 15 sigma^2/M, C8),
# unlike the single-mode case of C5 where G ∝ lam^2.
# eta = eta0/(t+t0)^2 (p = 2, de Sitter-like).  Exact (continuous) frozen variance per mode, for 2 lam eta0 >> t0:
#   V_inf = G eta0^2 ∫_{t0}^inf u^-4 exp(-2 lam eta0/u) du  ≈  G / (4 lam^3 eta0)      -> V ∝ lam^-3 ∝ k^-6.
# PRE-REGISTERED: slope d ln V/d ln lam = -3 (not -1 as for single-mode ES), V_meas/V_pred within ~15% for lam >> t0/(2 eta0).
#   => in a network with white ES noise, no p > 1 gives a scale-invariant (k^-3) spectrum; C6's p = 5/3 result needs G ∝ lam^2.
import numpy as np

N, R, M, sigma, eta0, t0, T = 256, 24, 4, 0.1, 1.0e4, 250.0, 200_000
rng = np.random.default_rng(15)
th = np.zeros((R, N))
for t in range(1, T + 1):
    eta = eta0 / (t + t0) ** 2
    L, Rt = np.roll(th, 1, 1), np.roll(th, -1, 1)
    eps = rng.standard_normal((R, N, M))
    xp = th[..., None] + sigma * eps
    lp = 0.5 * ((xp - L[..., None]) ** 2 + (xp - Rt[..., None]) ** 2)
    l0 = 0.5 * ((th - L) ** 2 + (th - Rt) ** 2)
    g = ((lp - l0[..., None]) * eps).mean(-1) / sigma
    th = th - eta * g
    if t == T // 2:
        half = (np.abs(np.fft.rfft(th, axis=1)) ** 2 / N).mean(0)

V = (np.abs(np.fft.rfft(th, axis=1)) ** 2 / N).mean(0)
ks = 2 * np.pi * np.fft.rfftfreq(N)
lam = 4 * np.sin(ks / 2) ** 2
G = 15 * sigma ** 2 / M
# exact integral prediction (incomplete gamma via numeric quadrature)
u = np.logspace(np.log10(t0), 9, 20001)
pred = np.array([G * eta0 ** 2 * np.trapezoid(u ** -4 * np.exp(-2 * l * eta0 / u), u) if l > 0 else np.nan for l in lam])
sel = (lam > 0.1) & (lam < 3.9)
slope = np.polyfit(np.log(lam[sel]), np.log(V[sel]), 1)[0]
print(f"fitted slope (lam in 0.1..3.9) = {slope:.3f}  (pred -3)")
for i in [8, 16, 32, 64, 96, 128]:
    print(f"  k={ks[i]:.3f} lam={lam[i]:.3f}: V={V[i]:.3e} pred={pred[i]:.3e} ratio={V[i]/pred[i]:.3f}  frozen? V(T/2)/V(T)={half[i]/V[i]:.3f}")
r = V[sel] / pred[sel]
print(f"median ratio {np.median(r):.3f}, spread (16-84%) {np.percentile(r,16):.3f}-{np.percentile(r,84):.3f}")
