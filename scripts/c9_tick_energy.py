# C9 (H8): if the network HAD a global tick clock, what would the relaxed Hamiltonian constraint look like?
# Minisuperspace FRW with lapse fixed by the time gauge N = a^beta (units 3/(8 pi G) = 1):
#   L(a, a') = - a^{1-beta} a'^2 / 1  -  N a^3 rho(a),   rho = rho0 a^{-3(1+w)}     (' = d/dT)
# T does not appear explicitly -> conserved E = -a^{1-beta} a'^2 + a^{3+beta} rho.   Friedmann (GR) = constraint E = 0.
# With E != 0:  H^2 = a'^2/(a^2 N^2) = rho - E a^{-(3+beta)}   =>  E behaves as a fluid with w_E = beta/3.
# Network clock (C2): N = nu = 1/a -> beta = -1 -> w_E = -1/3 : the "c t = const" component (curvature-like, rho ∝ a^-2).
# Known special cases: beta = -3 (unimodular time) -> Lambda; beta = 0 (cosmic/dust time) -> dust; beta = 1 (conformal) -> radiation.
# PRE-REGISTERED: numerically integrating the Euler-Lagrange equation with E != 0, the extra density
#   rho_E = H^2 - rho  scales as a^{-(3+beta)} with fitted w_E = beta/3 for beta = -3, -1, 0, 1.
import numpy as np


def run(beta, w=0.0, rho0=1.0, E=-0.3):
    V = lambda a: rho0 * a ** (beta - 3 * w)                 # N a^3 rho
    dV = lambda a: rho0 * (beta - 3 * w) * a ** (beta - 3 * w - 1)
    a = 1.0
    ad = np.sqrt((V(a) - E) / a ** (1 - beta))               # choose a' > 0 with the given E
    y = np.array([a, ad])
    def f(y):
        a, ad = y
        add = (dV(a) - (1 - beta) * a ** (-beta) * ad ** 2) / (2 * a ** (1 - beta))
        return np.array([ad, add])
    h, A, R = 1e-4, [], []
    Es = []
    while y[0] < 50:
        k1 = f(y); k2 = f(y + h/2*k1); k3 = f(y + h/2*k2); k4 = f(y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        a, ad = y
        N = a ** beta
        H2 = ad ** 2 / (a ** 2 * N ** 2)
        rho = rho0 * a ** (-3 * (1 + w))
        A.append(a); R.append(H2 - rho)
        Es.append(-a ** (1 - beta) * ad ** 2 + a ** (3 + beta) * rho)
        h = 1e-4 * a ** (1 - beta / 2 + 0.5)                 # adaptive-ish step
    A, R, Es = np.array(A), np.array(R), np.array(Es)
    sel = A > 2
    slope = np.polyfit(np.log(A[sel]), np.log(np.abs(R[sel])), 1)[0]
    wE = -slope / 3 - 1
    return wE, np.ptp(Es) / abs(E)


for beta, name in [(-3, "unimodular"), (-1, "network tick (N = 1/a)"), (0, "cosmic time"), (1, "conformal time")]:
    wE, drift = run(beta)
    print(f"beta={beta:+d} {name:>24}: fitted w_E = {wE:+.4f}, predicted beta/3 = {beta/3:+.4f}, rel. drift of E = {drift:.1e}")
