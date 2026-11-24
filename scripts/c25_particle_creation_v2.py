# C25 (review fix of C22/c23): particle creation with the time dependence switched OFF smoothly before projecting,
# and parameters kept in the continuum regime (m = m0/nu <= ~0.16).
# Rate of change: d ln nu / dT = -lam(T) / (2T), lam = on(T) * off(T).  on: step at T0 (sudden) or tanh ramp (smooth); off: tanh ramp to 0 before T_end.
# Projection on the instantaneous eigenbasis is done after nu is frozen, so no first-order adiabatic correction contaminates |beta|^2.
# PRE-REGISTERED: smooth on + frozen end: |beta|^2 << 5e-8 (exponentially small); sudden on + frozen end: |beta|^2 ∝ eps0^2 (slope ~2).
import numpy as np

m0, q = 0.05, 0.3
M = 2 * m0


def make_nu(T0, smooth, Tend):
    W = 40.0 / M
    Ts = np.linspace(T0, Tend, 400001)
    on = 0.5 * (1 + np.tanh((Ts - T0 - 3 * W) / W)) if smooth else np.ones_like(Ts)
    off = 0.5 * (1 - np.tanh((Ts - (Tend - 4 * W)) / W))
    rate = -on * off / (2 * Ts)
    lognu = np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(Ts))])
    return (lambda T: np.exp(np.interp(T, Ts, lognu))), Ts[-1]


def hmat(nu, k):
    J = nu ** 2; m = m0 / nu
    f = J * (1 + m) + J * (1 - m) * np.exp(-1j * k)
    return np.array([[0, f], [np.conj(f), 0]])


def run(T0, k, smooth, factor):
    W = 40.0 / M
    Tend = T0 * factor + (10 * W if smooth else 6 * W)
    nu_fun, Tend = make_nu(T0, smooth, Tend)
    w, v = np.linalg.eigh(hmat(nu_fun(T0), k)); psi = v[:, 0].astype(complex)
    T, numax_m = T0, 0
    dt = 0.02 / abs(hmat(1.0, k)[0, 1])
    while T < Tend:
        H1 = hmat(nu_fun(T), k); H2 = hmat(nu_fun(T + dt / 2), k); H3 = hmat(nu_fun(T + dt), k)
        k1 = -1j * H1 @ psi; k2 = -1j * H2 @ (psi + dt / 2 * k1); k3 = -1j * H2 @ (psi + dt / 2 * k2); k4 = -1j * H3 @ (psi + dt * k3)
        psi = psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); T += dt
    nu_end = nu_fun(T)
    w, v = np.linalg.eigh(hmat(nu_end, k))
    return abs(v[:, 1].conj() @ psi) ** 2, m0 / nu_end


print("sudden on, frozen end, k = pi + q:")
res = []
for T0 in [10.0, 20.0, 40.0, 80.0]:
    eps0 = (0.5 / T0) / M
    b, mend = run(T0, np.pi + q, False, 10.0)
    res.append((eps0, b)); print(f"   T0={T0:5.1f} eps0={eps0:.4f}: |beta|^2 = {b:.3e}   (m at end {mend:.3f})")
e, b = np.array(res).T
print(f"   log-log slope = {np.polyfit(np.log(e), np.log(b), 1)[0]:.2f} (pred ~2)")
print("smooth on, frozen end:")
for T0 in [10.0, 40.0]:
    b, mend = run(T0, np.pi + q, True, 10.0)
    print(f"   T0={T0}: |beta|^2 = {b:.3e}   (m at end {mend:.3f})")
