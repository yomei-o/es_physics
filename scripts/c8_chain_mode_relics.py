# C8 (H7): why the ES-chain noise of C3 did not decay cleanly for p = 1.
# Periodic chain, each agent runs ES (Stein, M samples) on its local loss l_i = 1/2[(x-th_{i-1})^2 + (x-th_{i+1})^2],
# eta = eta0/(t+t0), start from 0 (pure noise).  Fourier mode k: lam_k = 4 sin^2(k/2), r_k = eta0 lam_k.
# ES noise at gradient ~ 0: curvature 2 -> per-agent noise var G = 4 * 15 sigma^2/(4M) = 15 sigma^2 / M (white).
# Continuous-time exact prediction (per normalized mode):
#   V_k(t) = G eta0^2/(2r-1) [ (t+t0)^-1 - t0^{2r-1} (t+t0)^{-2r} ]     (r != 1/2)
#   k = 0: V_0 = G eta0^2 (1/t0 - 1/(t+t0))   (frozen random walk)
# PRE-REGISTERED: measured/predicted within ~15% for all k (discreteness), relic/thermal boundary k_c = 2 asin(sqrt(1/(8 eta0))),
#   late-time spectrum t*V_k has a peak just above k_c and a frozen plateau at k -> 0.
import numpy as np

N, R, M, sigma, eta0, t0, T = 256, 40, 4, 0.1, 4.0, 20.0, 100_000
rng = np.random.default_rng(7)
th = np.zeros((R, N))
ks = 2 * np.pi * np.fft.rfftfreq(N)
lam = 4 * np.sin(ks / 2) ** 2
r = eta0 * lam
G = 15 * sigma ** 2 / M
snap = {}
for t in range(1, T + 1):
    eta = eta0 / (t + t0)
    L, Rt = np.roll(th, 1, 1), np.roll(th, -1, 1)
    eps = rng.standard_normal((R, N, M))
    xp = th[..., None] + sigma * eps
    lp = 0.5 * ((xp - L[..., None]) ** 2 + (xp - Rt[..., None]) ** 2)
    l0 = 0.5 * ((th - L) ** 2 + (th - Rt) ** 2)
    g = ((lp - l0[..., None]) * eps).mean(-1) / sigma
    th = th - eta * g
    if t in (1_000, 10_000, 100_000):
        F = np.fft.rfft(th, axis=1) / np.sqrt(N)
        snap[t] = (np.abs(F) ** 2).mean(0)


def pred(t):
    tt = t + t0
    out = np.empty_like(r)
    for i, ri in enumerate(r):
        if ri == 0:
            out[i] = G * eta0 ** 2 * (1 / t0 - 1 / tt)
        elif abs(2 * ri - 1) < 1e-9:
            out[i] = G * eta0 ** 2 * np.log(tt / t0) / tt
        else:
            out[i] = G * eta0 ** 2 / (2 * ri - 1) * (1 / tt - t0 ** (2 * ri - 1) * tt ** (-2 * ri))
    return out


kc = 2 * np.arcsin(np.sqrt(1 / (8 * eta0)))
print(f"k_c = {kc:.3f} (wavelength {2*np.pi/kc:.1f} sites), G = {G:.4f}")
idx = [0, 1, 2, 4, 7, 10, 12, 15, 20, 30, 64, 128]
for t, V in snap.items():
    P = pred(t)
    # average over small k-bins to reduce scatter (each k has 2R real dof)
    print(f"\nt = {t}:  k, r, t*V_meas, t*V_pred, ratio")
    for i in idx:
        print(f"  k={ks[i]:.3f} r={r[i]:6.3f}  {t*V[i]:10.4e} {t*P[i]:10.4e}  {V[i]/P[i]:6.3f}")
    print(f"  median |ratio-1| over all k: {np.median(abs(snap[t]/P - 1)):.3f}")
V = snap[100_000]
j = np.argmax(V[1:]) + 1
print(f"\nlate-time peak of V_k (k>0) at k = {ks[j]:.3f}, r = {r[j]:.3f}")
