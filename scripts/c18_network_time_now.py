# C18 (review fix): local network-time exponent in LambdaCDM.
# c_net = nu^2 = a^-2, network ticks T = ∫ a dtau  =>  p_loc = -dln c_net/dln T = 2 H T / a.
# The conversion p = 2n/(1+n) (n = Ht) used earlier is valid only for a single power law.
import numpy as np
H0 = 67.4 / 977.792; Om, Or = 0.315, 9.1e-5; OL = 1 - Om - Or
lna = np.linspace(np.log(1e-9), np.log(3.0), 600001); a = np.exp(lna)
H = H0 * np.sqrt(Or/a**4 + Om/a**3 + OL)
dt = 1/H
tau = np.concatenate([[0], np.cumsum(0.5*(dt[1:]+dt[:-1])*np.diff(lna))])
T = np.concatenate([[0], np.cumsum(0.5*((a*dt)[1:]+(a*dt)[:-1])*np.diff(lna))])
p = 2*H*T/a
i0 = np.argmin(abs(a-1)); k = np.where(np.diff(np.sign(p-1)))[0][0]
print(f"today: tau0={tau[i0]:.3f} Gyr, T0/t0={T[i0]/tau[i0]:.3f}, p_loc={p[i0]:.4f}, H0*t0={H0*tau[i0]:.4f}")
print(f"p_loc = 1 at tau={tau[k]:.2f} Gyr (a={a[k]:.3f}, z={1/a[k]-1:.3f}), {tau[k]-tau[i0]:+.2f} Gyr from now")
for z in [1000, 10, 3, 1, 0.5, 0.2, 0, -0.3]:
    m = np.argmin(abs(a-1/(1+z))); print(f"  z={z:6}: p_loc={p[m]:.3f}")
q0 = 0.5*Om + Or - OL
print(f"q0={q0:.3f}, instantaneous w_eff={(2*q0-1)/3:.3f}")
