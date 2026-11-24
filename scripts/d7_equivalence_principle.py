# D7: equivalence-principle check of the coincidence rule.
# Same physical set-up placed in two uniform clock rates nu = 1.0 and nu = 0.7:
#   linear part: mass rotation M0*nu per agent, hopping t0*nu_i*nu_j (coincidence rule)
#   packet: same width in LOCAL rods (rod = (2 t0/M0) nu sites), same total norm (particle number)
# Nonlinear interaction, two variants:
#   "single": agent's own rotation rate M0*nu*(1 + lam*rho_i)              (1-agent process, rate ~ nu)
#   "pair"  : extra rotation lam*M0*sum_j nu_i nu_j rho_j / 2   (j = neighbours; 2-agent process, rate ~ nu^2)
# Observable (dimensionless, local units): nonlinear frequency shift / local clock rate (M0 nu).
# PRE-REGISTERED: single -> ratio(nu=0.7)/ratio(nu=1) = 1/0.7 = 1.43 (EP violated); pair -> 1.00 (EP holds).
import numpy as np

M0, t0, lam = 0.4, 0.5, 0.05


def simulate(nu, variant, nonlinear=True):
    rod = 2 * t0 / M0 * nu                       # sites per local rod
    N = 600
    pos = np.arange(N) - N / 2
    w_sites = 12 * rod
    a = np.exp(-pos**2 / (2 * w_sites**2)); b = np.zeros(N)
    nrm = np.sqrt((a**2).sum() / 40.0); a /= nrm  # total norm 40 (fixed particle number)
    tick = 1 / (M0 * nu)                          # local clock period
    T = 60 * tick; dt = 0.02 * tick; steps = int(T / dt)
    hop = t0 * nu * nu
    def f(a, b):
        rho = a**2 + b**2
        rate = M0 * nu * np.ones(N)
        if nonlinear:
            if variant == "single":
                rate = rate + lam * M0 * nu * rho
            else:
                rate = rate + lam * M0 * 0.5 * nu * nu * (np.roll(rho, 1) + np.roll(rho, -1))
        da = rate * b + hop * (np.roll(a, -1) - np.roll(a, 1))
        db = -rate * a - hop * (np.roll(b, -1) - np.roll(b, 1))
        return da, db
    phase = []
    c = N // 2
    for s in range(steps):
        k1 = f(a, b); k2 = f(a + dt/2*k1[0], b + dt/2*k1[1]); k3 = f(a + dt/2*k2[0], b + dt/2*k2[1]); k4 = f(a + dt*k3[0], b + dt*k3[1])
        a = a + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); b = b + dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        phase.append(np.arctan2(-b[c], a[c]))
    ph = np.unwrap(phase)
    n0 = len(ph) // 4                             # skip transient
    omega = np.polyfit(np.arange(n0, len(ph)) * dt, ph[n0:], 1)[0]
    return omega


for variant in ["single", "pair"]:
    shifts = {}
    for nu in [1.0, 0.7]:
        w_nl = simulate(nu, variant, True); w_lin = simulate(nu, variant, False)
        shifts[nu] = (w_nl - w_lin) / (M0 * nu)
        print(f"{variant:6s} nu={nu}: omega_lin/(M0 nu) = {w_lin/(M0*nu):.5f}, nonlinear shift in local units = {shifts[nu]:.5f}")
    print(f"   -> ratio shift(nu=0.7)/shift(nu=1) = {shifts[0.7]/shifts[1.0]:.3f}   (pred: {'1.43, EP violated' if variant=='single' else '1.00, EP holds'})\n")
