# F3: weak force and neutron lifetime, seen from the zero-sum agent network.
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

# ---------------- (a) decay emerges inside a norm-conserving (zero-sum) network -------------------------
# One localised mode (frequency w0, "neutron") weakly coupled (g/sqrt(N)) to every mode of a ring of N agents
# (dispersion eps_k = 2J cos k). Whole system is unitary (= real antisymmetric generator, Sec.5.3).
# PRE-REGISTERED: survival |a(t)|^2 ~ exp(-Gamma t), Gamma = 2 g^2 / sqrt(4J^2 - w0^2)   (Fermi golden rule)
print("=== (a) exponential decay from a zero-sum network (Fermi golden rule) ===")
N, J = 1500, 1.0
eps = 2 * J * np.cos(2 * np.pi * np.arange(N) / N)
for w0, g in [(0.5, 0.08), (1.0, 0.08), (0.5, 0.12)]:
    H = np.diag(np.concatenate([[w0], eps]))
    H[0, 1:] = H[1:, 0] = g / np.sqrt(N)
    E, V = np.linalg.eigh(H)
    c0 = V[0, :]
    ts = np.linspace(0, 60, 61)
    amp = np.array([abs(np.sum(np.abs(c0) ** 2 * np.exp(-1j * E * t))) ** 2 for t in ts])
    sel = (amp > 1e-3) & (ts > 3)
    G_meas = -np.polyfit(ts[sel], np.log(amp[sel]), 1)[0]
    G_pred = 2 * g**2 / np.sqrt(4 * J**2 - w0**2)
    resid = np.abs(np.log(amp[sel]) - np.polyval(np.polyfit(ts[sel], np.log(amp[sel]), 1), ts[sel])).max()
    print(f"w0={w0}, g={g}: Gamma measured {G_meas:.5f}, golden rule {G_pred:.5f} (ratio {G_meas/G_pred:.3f}); "
          f"max deviation from a straight line in log|a|^2: {resid:.3f};  total norm conserved: {np.allclose(np.sum(np.abs(c0)**2), 1)}")

# ---------------- (b) Higgs: agents' frames align -> gauge field gets a mass (short-range force) ------------
# U(1) link field A on a 3D lattice + agents with phases theta (|psi|=1). Energy:
#   1/2 sum_plaq F^2  +  kappa * sum_links (1 - cos(theta_i - theta_j + A_ij)) ~ kappa/2 (A + D theta)^2
# Unitary gauge (theta absorbed) -> mass term kappa A^2.
# PRE-REGISTERED: 3 modes with omega^2 = k_lat^2 + 2 kappa... (transverse) and the longitudinal one omega^2 = 2 kappa
#   at k->0 all three have the same gap m^2 = 2 kappa ; no gauge zero mode.
print("\n=== (b) frame alignment (Higgs) gives the gauge field a mass ===")
kappa = 0.05
for kk in [0.0, 0.3]:
    k = np.array([0.0, 0.0, max(kk, 1e-9)])
    d = np.exp(1j * k) - 1
    V = np.zeros((3, 3), complex)
    for mu in range(3):
        for nu in range(mu + 1, 3):
            vec = np.zeros(3, complex); vec[nu] = d[mu]; vec[mu] = -d[nu]
            V += np.outer(vec.conj(), vec)
    V = 0.5 * (V + V.conj().T) + 2 * kappa * np.eye(3) / 2 * 2   # mass term: (kappa/2)*2... -> omega^2 shift 2*kappa
    w = np.linalg.eigvalsh(V)
    print(f"|k|={kk}: omega^2 = {np.round(w, 5)}  (pred transverse {4*np.sin(kk/2)**2 + 2*kappa:.5f} x2, longitudinal {2*kappa:.5f}); zero modes: {np.sum(np.abs(w) < 1e-10)}")
print(f"-> force range ~ 1/m = 1/sqrt(2 kappa) = {1/np.sqrt(2*kappa):.2f} lattice units (Yukawa), vs infinite for the photon")

# ---------------- (c) neutron lifetime: structure and sensitivity ----------------------------------------
print("\n=== (c) neutron beta decay n -> p e nu ===")
me = 0.51099895            # MeV
dm = 1.29333236            # m_n - m_p, MeV  (endpoint total electron energy E0 = dm, recoil neglected)
alpha = 1 / 137.035999


def fermi(E, Z=1):
    p = np.sqrt(E**2 - me**2); eta = alpha * Z * E / p
    return 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))   # non-relativistic Fermi function


def f_int(E0, with_F=True):
    g = lambda E: np.sqrt(E**2 - me**2) * E * (E0 - E) ** 2 * (fermi(E) if with_F else 1)
    return quad(g, me, E0)[0] / me**5


f0 = f_int(dm, False); fF = f_int(dm, True)
print(f"phase-space integral f (no Coulomb) = {f0:.4f}, with Fermi function = {fF:.4f}   (literature f = 1.6887 incl. all corrections)")
Vud, lam = 0.97373, 1.2754                       # PDG-type inputs (from memory, to be checked)
tau = 4908.6 / (Vud**2 * (1 + 3 * lam**2))       # Czarnecki-Marciano-Sirlin master formula (from memory)
print(f"tau_n = 4908.6 s / (|Vud|^2 (1+3 lambda^2)) = {tau:.1f} s   [inputs Vud={Vud}, lambda={lam}]")
print("measured: bottle ~ 877.8 s, beam ~ 887.7 s  (values from memory; the ~9 s 'neutron lifetime anomaly' is open)")

# Sargent rule: f ~ E0^5/30 for E0 >> me
for E0 in [5, 20, 100]:
    print(f"  Sargent check E0={E0} me: f/(E0^5/30) = {f_int(E0*me, False) / (E0**5 / 30):.3f}")
# sensitivity of the lifetime to the n-p mass difference
for frac in [-0.01, +0.01]:
    r = f_int(dm, True) / f_int(dm * (1 + frac), True)
    print(f"  m_n - m_p changed by {frac*100:+.0f}% -> lifetime x {r:.3f}")
print(f"  neutron becomes stable if m_n - m_p < m_e = {me} MeV (now {dm} MeV); if m_n < m_p, hydrogen is unstable")

# why so slow: Gamma ~ G_F^2 Q^5 with G_F ~ g^2/M_W^2
GF = 1.1663787e-11       # MeV^-2
hbar = 6.582119569e-22   # MeV s
MW = 80377.0             # MeV
Q = dm
print(f"  (M_W/Q)^4 = {(MW/Q)**4:.2e} : the heavy mediator (aligned frames, (b)) suppresses the rate;"
      f" a strong-interaction time hbar/(1 GeV) = {hbar/1000:.1e} s vs tau_n ~ 880 s, ratio {880/(hbar/1000):.1e}")
