# C10 (H9): Sorkin-type estimate for the fluid conjugate to the network's update count.
# C9: with time gauge N = a^beta, the conserved tick energy is a fluid with w = beta/3, and in the action
#   S_E = - rho_E0 * U,   U = V_c * T_total,   T_total = ∫ dT = ∫ a^{-beta} d tau   (Planck units, hbar = c = 1)
# If the number of updates U is Poisson (Delta U ~ sqrt U) then rho_E0 ~ 1/sqrt(U)  (Sorkin's "everpresent Lambda" for beta = -3).
# (Review: only beta=-3 is normalization independent; the others assume sites are Planck-sized today.)
# PRE-REGISTERED: Omega_E = rho_E0 / rho_crit is O(1) (0.3 - 5) for beta = -3 (Lambda), 0 (dust: per-site proper time), -1 (network tick: w=-1/3),
#   because U ~ (t0/tP)^4 and rho_crit ~ (t0/tP)^-2 for every counting.
import numpy as np

tP = 5.391247e-44                       # s
H0 = 67.4 / 3.0857e19                   # 1/s
Om, Or = 0.315, 9.1e-5
OL = 1 - Om - Or
rho_crit = 3 * (H0 * tP) ** 2 / (8 * np.pi)     # in Planck density units
V_c = 4 * np.pi / 3 / (H0 * tP) ** 3            # Hubble volume today, Planck units (a0 = 1)

lna = np.linspace(np.log(1e-12), 0, 400001)
a = np.exp(lna)
H = H0 * tP * np.sqrt(Or / a**4 + Om / a**3 + OL)   # Planck units
dtau = 1 / H                                     # dtau/dlna
tau0 = np.trapezoid(dtau, lna)
print(f"t0/tP = {tau0:.3e},  rho_crit = {rho_crit:.3e} rho_P,  V_c = {V_c:.3e} lP^3")
for beta, name in [(-3, "4-volume (unimodular)      w=-1  "), (0, "per-site proper time (dust) w=0   "), (-1, "network tick N=1/a          w=-1/3")]:
    T = np.trapezoid(a ** (-beta) * dtau, lna)
    U = V_c * T
    Omega = U ** -0.5 / rho_crit
    print(f"beta={beta:+d} {name}: T_total/t0 = {T/tau0:.3f},  U = {U:.2e},  Omega ~ {Omega:.2f}")
print("observed: Omega_L = 0.685, Omega_cdm = 0.265, |Omega_(w=-1/3)| << 1 (curvature-like, ~1e-2 or below)")
