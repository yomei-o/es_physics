# D8: perihelion precession (2nd-order test, PPN beta).
# Particle Hamiltonian implied by the coincidence rule (mass M0*nu, hopping ~ nu^2):
#     H = sqrt(M^2 + c^2 p^2) with M = nu, c = nu^2   ->  H = nu * sqrt(1 + nu^2 p^2)      (units M0 = c0 = 1)
# Two candidate field laws for the clock rate around a point mass (U = GM/r):
#     additive  : nu harmonic          -> nu = 1 - U
#     multiplic.: ln(nu) harmonic      -> nu = exp(-U)   (redshift factors compose multiplicatively)
# PRE-REGISTERED: precession / GR-value = (2 + 2 gamma - beta)/3 with gamma = 1:
#     additive  -> beta = 1/2 -> 7/6 = 1.1667 ;  multiplicative -> beta = 1 -> 1.0000
import numpy as np
from scipy.integrate import solve_ivp

GM = 1e-3


def make_rhs(law):
    if law == "additive":
        nu = lambda r: 1 - GM / r; dnu = lambda r: GM / r**2
    else:
        nu = lambda r: np.exp(-GM / r); dnu = lambda r: np.exp(-GM / r) * GM / r**2

    def rhs(t, s):
        x, y, px, py = s
        r = np.hypot(x, y); n = nu(r); p2 = px * px + py * py
        S = np.sqrt(1 + n * n * p2)
        dHdp = n**3 / S                                # dH/dp_i = n^3 p_i / S
        dHdn = S + n * n * p2 / S                      # dH/dnu
        g = dHdn * dnu(r) / r
        return [dHdp * px, dHdp * py, -g * x, -g * y]
    return rhs


def precession(law, a=1.0, e=0.3, orbits=6):
    # start at perihelion with Newtonian velocity, measure angle advance between successive perihelia
    rp = a * (1 - e)
    v = np.sqrt(GM * (1 + e) / rp)
    s0 = [rp, 0.0, 0.0, v]
    rhs = make_rhs(law)
    def peri(t, s):  # radial velocity zero crossing from - to +
        return s[0] * s[2] + s[1] * s[3]
    peri.direction = 1
    T = 2 * np.pi * np.sqrt(a**3 / GM)
    sol = solve_ivp(rhs, [0, orbits * T * 1.01], s0, events=peri, rtol=1e-12, atol=1e-14, method="DOP853")
    ev = sol.y_events[0]
    angs = np.unwrap(np.arctan2(ev[:, 1], ev[:, 0]))
    dphi = np.diff(np.concatenate([[0.0], angs]))
    return dphi[1:].mean() if len(dphi) > 1 else dphi.mean()


a, e = 1.0, 0.3
gr = 6 * np.pi * GM / (a * (1 - e**2))
print(f"GR prediction per orbit: {gr:.6e} rad  (GM={GM}, a={a}, e={e})")
for law, pred in [("additive", 7 / 6), ("multiplicative", 1.0)]:
    d = precession(law, a, e)
    print(f"{law:15s}: precession {d:.6e} rad/orbit -> ratio to GR {d/gr:.4f}  (pred {pred:.4f})")


# ---- control: exact Schwarzschild (isotropic coords) with the SAME initial conditions --------------
# ds^2 = -A dt^2 + B dx^2, A = ((1-U/2)/(1+U/2))^2, B = (1+U/2)^4, U = GM/r ;  H = sqrt(A (1 + p^2/B))
def rhs_gr(t, s):
    x, y, px, py = s
    r = np.hypot(x, y); U = GM / r; dU = -GM / r**2
    A = ((1 - U / 2) / (1 + U / 2))**2; B = (1 + U / 2)**4
    dA = 2 * (1 - U / 2) / (1 + U / 2) * (-(1 / 2) * (1 + U / 2) - (1 - U / 2) * (1 / 2)) / (1 + U / 2)**2 * dU
    dB = 4 * (1 + U / 2)**3 * 0.5 * dU
    p2 = px * px + py * py
    H = np.sqrt(A * (1 + p2 / B))
    dHdp = A / (B * H)
    dHdr = (dA * (1 + p2 / B) - A * p2 * dB / B**2) / (2 * H)
    return [dHdp * px, dHdp * py, -dHdr * x / r, -dHdr * y / r]


def precession_rhs(rhs, a=1.0, e=0.3, orbits=6):
    rp = a * (1 - e); v = np.sqrt(GM * (1 + e) / rp)
    def peri(t, s): return s[0] * s[2] + s[1] * s[3]
    peri.direction = 1
    T = 2 * np.pi * np.sqrt(a**3 / GM)
    sol = solve_ivp(rhs, [0, orbits * T * 1.01], [rp, 0, 0, v], events=peri, rtol=1e-12, atol=1e-14, method="DOP853")
    angs = np.unwrap(np.arctan2(sol.y_events[0][:, 1], sol.y_events[0][:, 0]))
    return np.diff(np.concatenate([[0.0], angs]))[1:].mean()


print("\n--- same initial conditions, compared with exact Schwarzschild ---")
d_gr = precession_rhs(rhs_gr)
print(f"Schwarzschild (exact): {d_gr:.6e} rad/orbit")
for law, pred in [("additive", 7 / 6), ("multiplicative", 1.0)]:
    d = precession_rhs(make_rhs(law))
    print(f"{law:15s}: ratio to exact Schwarzschild = {d/d_gr:.4f}  (pred {pred:.4f})")
