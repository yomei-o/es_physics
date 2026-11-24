# F4a: ES = BP inside gauge theory.
# BP side : gradient flow of the Maxwell action  dA_mu/dt = - dS/dA_mu = sum_nu D_nu^- F_{nu mu}   (Luscher's flow)
# ES side : Gaussian smoothing of the field with variance sigma^2 = 2t (heat kernel exp(-k_lat^2 t))
# PRE-REGISTERED: the gauge-invariant field strength F after flow equals F of the Gaussian-smoothed field exactly
# (linear theory); the gauge-dependent longitudinal part is NOT smoothed by the flow (it is a zero mode of S).
import numpy as np

rng = np.random.default_rng(3)
L, D = 16, 3
A = rng.normal(size=(D, L, L, L)) * 0.1


def fwd(f, mu): return np.roll(f, -1, axis=mu) - f
def bwd(f, mu): return f - np.roll(f, 1, axis=mu)


def Fmunu(A):
    return {(m, n): fwd(A[n], m) - fwd(A[m], n) for m in range(D) for n in range(D) if m < n}


def flow(A, t, dt=0.002):
    A = A.copy()
    for _ in range(int(round(t / dt))):
        F = Fmunu(A)
        g = np.zeros_like(A)
        for (m, n), f in F.items():           # dS/dA: S = 1/2 sum F^2
            g[n] += -bwd(f, m)                 # dF_mn/dA_n = d_m  -> adjoint -bwd
            g[m] += bwd(f, n)
        A -= dt * g
    return A


def smooth(A, t):
    k = 2 * np.pi * np.fft.fftfreq(L)
    K2 = sum(np.meshgrid(*([4 * np.sin(k / 2) ** 2] * D), indexing='ij'))
    return np.real(np.fft.ifftn(np.fft.fftn(A, axes=(1, 2, 3)) * np.exp(-K2 * t), axes=(1, 2, 3)))


for t in [0.5, 2.0, 5.0]:
    Af, As = flow(A, t), smooth(A, t)
    Ff, Fs, F0 = Fmunu(Af), Fmunu(As), Fmunu(A)
    err = max(np.abs(Ff[k] - Fs[k]).max() for k in Ff)
    scale = max(np.abs(F0[k]).max() for k in F0)
    # longitudinal (pure-gauge) part: divergence of A
    div = lambda X: sum(bwd(X[m], m) for m in range(D))
    print(f"t={t}: max|F_flow - F_smooth| = {err:.2e} (F scale {scale:.2e}, F_smooth scale {max(np.abs(Fs[k]).max() for k in Fs):.2e}) | "
          f"div A: original {np.abs(div(A)).max():.3f}, after flow {np.abs(div(Af)).max():.3f}, after smoothing {np.abs(div(As)).max():.3f}")
