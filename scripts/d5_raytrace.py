# Exact geometric-optics check for d5: Hamiltonian H = J(r) * 2*sqrt(sin^2(px/2)+sin^2(py/2)) (lattice dispersion),
# k = 0.6 initial along +x, launched far left at impact parameter b. Compare with the full wave simulation.
import numpy as np
from scipy.integrate import solve_ivp
alpha, soft = 4.0, 6.0
J = lambda x, y: 1 - alpha/np.sqrt(x*x+y*y+soft*soft)
def gradJ(x, y):
    r3 = (x*x+y*y+soft*soft)**1.5
    return alpha*x/r3, alpha*y/r3
def rhs(t, s, lattice):
    x, y, px, py = s
    if lattice:
        w = 2*np.sqrt(np.sin(px/2)**2+np.sin(py/2)**2)
        dwx, dwy = np.sin(px)/w, np.sin(py)/w          # d w / d p (since d sin^2(p/2)/dp = sin(p)/2)
    else:
        w = np.hypot(px, py); dwx, dwy = px/w, py/w
    gx, gy = gradJ(x, y)
    return [J(x,y)*dwx, J(x,y)*dwy, -gx*w, -gy*w]
for b, meas in [(40, -0.3273), (60, -0.1973), (90, -0.1000)]:
    out = []
    for lat in (True, False):
        sol = solve_ivp(rhs, [0, 2000], [-1500, b, 0.6, 0.0], args=(lat,), rtol=1e-10, atol=1e-12)
        px, py = sol.y[2, -1], sol.y[3, -1]
        if lat:   # direction of group velocity
            w = 2*np.sqrt(np.sin(px/2)**2+np.sin(py/2)**2); vx, vy = np.sin(px)/w, np.sin(py)/w
        else:
            vx, vy = px, py
        out.append(np.arctan2(vy, vx))
    print(f"b={b}: wave sim {meas:+.4f} | exact rays lattice {out[0]:+.4f}, continuum n=1/J {out[1]:+.4f} | GR-like 2x(continuum) {2*out[1]:+.4f}")
