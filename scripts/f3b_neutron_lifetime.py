# F3b: neutron lifetime assembled from its parts (not via the master formula), with an error budget.
# Gamma = G_F^2 |Vud|^2 (1 + 3 lambda^2) m_e^5 f (1 + RC) / (2 pi^3)      (natural units, hbar = c = 1)
# PRE-REGISTERED: (1) f with the relativistic Fermi function incl. finite proton size within 0.1 % of 1.6887
#                 (2) with measured lambda, Vud the lifetime lands near the bottle value (~878 s), not the beam (~888 s)
# All numerical inputs are from memory (PDG-era values) and must be checked before any citation.
import numpy as np
from scipy.integrate import quad
from scipy.special import loggamma

me = 0.51099895          # MeV
dm = 1.29333236          # m_n - m_p, MeV
Mn = 939.56542           # MeV (for recoil estimate)
alpha = 1 / 137.035999
GF = 1.1663787e-11       # MeV^-2
hbar = 6.582119569e-22   # MeV s
R = 0.84 / 386.159       # proton radius in units of hbar/(m_e c)

# endpoint with recoil: E0 = (Mn^2 - Mp^2 + me^2) / (2 Mn)  (total electron energy)
Mp = Mn - dm
E0 = (Mn**2 - Mp**2 + me**2) / (2 * Mn) / me     # in units of me
E0_norecoil = dm / me


def F_rel(W, Z=1):
    p = np.sqrt(W * W - 1); g = np.sqrt(1 - (alpha * Z) ** 2); eta = alpha * Z * W / p
    lg = 2 * np.real(loggamma(g + 1j * eta)) - 2 * np.real(loggamma(2 * g + 1))
    return 2 * (1 + g) * (2 * p * R) ** (2 * g - 2) * np.exp(np.pi * eta + lg)


def F_nr(W, Z=1):
    p = np.sqrt(W * W - 1); eta = alpha * Z * W / p
    return 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))


def f_int(E0, F):
    return quad(lambda W: np.sqrt(W * W - 1) * W * (E0 - W) ** 2 * F(W), 1, E0, limit=200)[0]


f_none = f_int(E0, lambda W: 1.0)
f_nr = f_int(E0, F_nr)
f_rel = f_int(E0, F_rel)
print("=== (1) phase-space integral f ===")
print(f"endpoint E0 = {E0*me:.6f} MeV (recoil) vs {dm:.6f} (no recoil)")
print(f"f: no Coulomb {f_none:.4f} | non-rel Fermi {f_nr:.4f} | relativistic Fermi + finite size {f_rel:.4f} | literature 1.6887 "
      f"(rel. diff {100*(f_rel/1.6887-1):+.2f} %)")

print("\n=== (2) lifetime ===")
K = 2 * np.pi**3 * hbar / (GF**2 * me**5)        # seconds
print(f"2 pi^3 hbar / (G_F^2 m_e^5) = {K:.1f} s")
RC = 0.03886                                      # total radiative correction (1+RC), from memory
Vud = 0.97373; dVud = 0.00031
for name, lam, dlam in [("PDG average", 1.2754, 0.0013), ("PERKEO III", 1.27641, 0.00056)]:
    tau = K / (f_rel * (1 + RC) * Vud**2 * (1 + 3 * lam**2))
    # error budget
    d_l = tau * 6 * lam * dlam / (1 + 3 * lam**2)
    d_v = tau * 2 * dVud / Vud
    d_rc = tau * 0.0004 / (1 + RC)
    print(f"lambda={lam} ({name}): tau = {tau:.1f} s   errors: lambda {d_l:.1f} s, Vud {d_v:.1f} s, RC {d_rc:.1f} s")
print("measured (from memory): bottle 877.75 +- 0.3 s (UCNtau 2021), beam 887.7 +- 2.2 s (2013)")
print("constant check: K/(f(1+RC)) =", round(K / (f_rel * (1 + RC)), 1), "s  vs master-formula constant 4908.6 s")

print("\n=== (3) what it would take to explain the 9 s anomaly with each input alone ===")
tau0 = K / (f_rel * (1 + RC) * Vud**2 * (1 + 3 * 1.2754**2))
for target in [877.75, 887.7]:
    r = tau0 / target
    lam_need = np.sqrt(((1 + 3 * 1.2754**2) * r - 1) / 3)
    print(f"to get {target} s: lambda would have to be {lam_need:.4f} (now 1.2754, shift {100*(lam_need/1.2754-1):+.2f} %), "
          f"or m_n-m_p shifted by {100*((r)**(1/5.4)-1):+.2f} % (f ~ dm^5.4 locally)")
print("dark-decay reading: Br(invisible) = 1 - tau_bottle/tau_beam =", round(1 - 877.75 / 887.7, 4))
