# C23: particle creation in the network when the global clock nu(T) changes (breakdown of the adiabatic dictionary, C2).
# SSH-type two-band chain (the zero-sum network of the sister note §7.2 in complex form): Bloch Hamiltonian
#   h(k) = [[0, f], [f*, 0]],  f = J1 + J2 e^{-ik},  J1 = J(1+m), J2 = J(1-m).
# Coincidence rule (§8): hopping J = nu^2, rest mass (gap 2 m J at k = pi) ∝ nu  ->  m = m0 / nu.
# Schedule nu(T) = (T/T0)^(-1/2) for T > T0 (c_net T = const).  Adiabatic parameter at start: eps0 = |dnu/dT| / (M nu^2) with M = 2 m0.
# Start in the lower-band instantaneous eigenstate at k = pi + q, evolve, measure |beta|^2 = population of the upper band at the end.
# PRE-REGISTERED: (a) q = 0 (particle at rest): eigenvectors are time independent -> |beta|^2 = 0 exactly.
#   (b) sudden switch-on of the time dependence at T0: |beta|^2 ∝ eps0^2 (log-log slope 2) for small eps0.
#   (c) smooth switch-on (nu frozen before T0, then rate turned on with a tanh ramp of width >> 1/M): |beta|^2 far below (b).
import numpy as np

m0, q = 0.1, 0.3


def h(T, nu_fun, k):
    nu = nu_fun(T); J = nu ** 2; m = m0 / nu
    f = J * (1 + m) + J * (1 - m) * np.exp(-1j * k)
    return np.array([[0, f], [np.conj(f), 0]])


def run(T0, k, smooth, T1_factor=30.0):
    if smooth:
        W = 40.0 / (2 * m0)                               # ramp width in ticks, >> 1/M
        def lam(T):                                       # turn the evolution on smoothly
            return 0.5 * (1 + np.tanh((T - T0 - 3 * W) / W))
        def nu_fun(T):
            # integrate d ln nu / dT = -lam(T)/(2T)  (closed form not needed: use small table)
            return np.exp(np.interp(T, Ts, lognu))
        Ts = np.linspace(T0, T0 * T1_factor + 10 * W, 200001)
        rate = -lam(Ts) / (2 * Ts)
        lognu = np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(Ts))])
        Tend = Ts[-1]
    else:
        nu_fun = lambda T: (T / T0) ** -0.5
        Tend = T0 * T1_factor
    w, v = np.linalg.eigh(h(T0, nu_fun, k)); psi = v[:, 0].astype(complex)
    T = T0
    Mfreq = 2 * m0
    dt = 0.02 / (abs(h(T0, nu_fun, k)[0, 1]) + 1e-12)
    while T < Tend:
        H1 = h(T, nu_fun, k); H2 = h(T + dt / 2, nu_fun, k); H3 = h(T + dt, nu_fun, k)
        f = lambda Hm, y: -1j * Hm @ y
        k1 = f(H1, psi); k2 = f(H2, psi + dt / 2 * k1); k3 = f(H2, psi + dt / 2 * k2); k4 = f(H3, psi + dt * k3)
        psi = psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); T += dt
    w, v = np.linalg.eigh(h(T, nu_fun, k))
    return abs(v[:, 1].conj() @ psi) ** 2, np.linalg.norm(psi)


print("(a) particle at rest, sudden start:")
for T0 in [2.0, 20.0]:
    b, n = run(T0, np.pi, False); print(f"   T0={T0}: |beta|^2 = {b:.2e} (norm {n:.6f})")
print("(b) sudden start, k = pi + q:")
res = []
for T0 in [5.0, 10.0, 20.0, 40.0, 80.0]:
    eps0 = (0.5 / T0) / (2 * m0)                         # |dnu/dT|/(M nu^2) at T0 with nu = 1
    b, n = run(T0, np.pi + q, False)
    res.append((eps0, b)); print(f"   T0={T0:5.1f} eps0={eps0:.4f}: |beta|^2 = {b:.3e}")
e, b = np.array(res).T
print(f"   log-log slope = {np.polyfit(np.log(e), np.log(b), 1)[0]:.2f} (pred 2)")
print("(c) smooth start, k = pi + q:")
for T0 in [10.0, 40.0]:
    bs, n = run(T0, np.pi + q, True, T1_factor=10.0)
    bh, _ = run(T0, np.pi + q, False, T1_factor=10.0)
    print(f"   T0={T0}: smooth {bs:.3e}  vs sudden {bh:.3e}")
