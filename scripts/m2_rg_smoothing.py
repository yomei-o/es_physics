# M2: ES smoothing vs exact integration of fast modes (0-dimensional phi^4, exactly computable by quadrature).
# V(phi) = m2/2 phi^2 + lam/24 phi^4 ; fast mode f ~ N(0, s2) (free propagator).
# ES smoothing:      V_s(phi)   = E_f[ V(phi + f) ]                      (first cumulant)
# exact (Wilsonian): V_eff(phi) = -ln E_f[ exp(-V_int(phi + f)) ] + m2/2 phi^2 terms, V_int = lam/24 phi^4
# PRE-REGISTERED: smoothing shifts the phi^2 coefficient by lam/2 s2 and leaves phi^4 unchanged;
#                 the exact result has the same phi^2 shift to first order in lam, plus a phi^4 shift of order lam^2 s2^2
#                 that smoothing misses (smoothing = linear/heat-equation half of the RG).
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

xg, wg = hermegauss(80); wg = wg / wg.sum()            # E over N(0,1)


def smooth(V, phi, s2): return np.array([np.sum(wg * V(p + np.sqrt(s2) * xg)) for p in phi])
def exact(Vint, phi, s2): return np.array([-np.log(np.sum(wg * np.exp(-Vint(p + np.sqrt(s2) * xg)))) for p in phi])


phi = np.linspace(-0.6, 0.6, 41)
print(" lam    s2  | smoothing: d(phi^2 coef)  d(lam)   | exact: d(phi^2 coef)  d(lam)     | pred lam*s2/2   1-loop 1PI dlam=-1.5 lam^2 s2^2")
for lam in [0.1, 0.5, 1.0]:
    for s2 in [0.1, 0.3]:
        Vint = lambda x: lam / 24 * x**4
        ys = smooth(Vint, phi, s2) - smooth(Vint, [0.0], s2)[0]
        ye = exact(Vint, phi, s2) - exact(Vint, [0.0], s2)[0]
        cs = np.polyfit(phi, ys, 6)[::-1]; ce = np.polyfit(phi, ye, 6)[::-1]
        # coefficients: c2 = (mass shift)/2 ; c4 = lam'/24
        print(f" {lam:4.1f}  {s2:4.1f} |   {2*cs[2]:+.5f}             {24*cs[4]-lam:+.5f}  |   {2*ce[2]:+.5f}           {24*ce[4]-lam:+.5f}    |   {lam*s2/2:.5f}        {-1.5*lam**2*s2**2:+.5f}")
