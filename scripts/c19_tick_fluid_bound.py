# C19: observational bound on the global-tick fluid of C9 (w = -1/3 in FLAT space, rho ∝ a^-2),
# compared with ordinary spatial curvature (same expansion history, different geometry).
# Data: DESI DR2 BAO (arXiv:2503.14738, Table 4; values extracted via page summaries, cross-checked twice, verify before citing)
#       + Planck 2018 CMB shift parameter R = 1.7502 ± 0.0046 at z* = 1089.92 (distance priors, arXiv:1808.05724).
# Parameters: Omega_m, Omega_X; A = c/(H0 r_d) profiled analytically (BAO only).  Omega_r = 9.1e-5 fixed.
# PRE-REGISTERED: curvature version gives Omega_k ~ 0.002 ± 0.002 (looser than DESI+full-CMB 0.0023 ± 0.0011);
#                 flat E-fluid is less constrained, sigma(Omega_E) ~ 0.005 - 0.01.
import numpy as np

Or = 9.1e-5
zs_star, R_obs, R_err = 1089.92, 1.7502, 0.0046
bao = [  # z, DM, sDM, DH, sDH, r
    (0.510, 13.588, 0.167, 21.863, 0.425, -0.459),
    (0.706, 17.351, 0.177, 19.455, 0.330, -0.404),
    (0.934, 21.576, 0.152, 17.641, 0.193, -0.416),
    (1.321, 27.601, 0.318, 14.176, 0.221, -0.434),
    (1.484, 30.512, 0.760, 12.817, 0.516, -0.500),
    (2.330, 38.988, 0.531, 8.632, 0.101, -0.431),
]
bgs = (0.295, 7.942, 0.075)

d, cov_blocks = [bgs[1]], [np.array([[bgs[2] ** 2]])]
for z, DM, sM, DH, sH, r in bao:
    d += [DM, DH]
    cov_blocks.append(np.array([[sM ** 2, r * sM * sH], [r * sM * sH, sH ** 2]]))
d = np.array(d)
C = np.zeros((len(d), len(d))); i = 0
for b in cov_blocks:
    n = b.shape[0]; C[i:i + n, i:i + n] = b; i += n
Ci = np.linalg.inv(C)

zg = np.concatenate([np.linspace(0, 3, 6001), np.logspace(np.log10(3.0005), np.log10(zs_star), 6000)])


def model(Om, OX, curved):
    OL = 1 - Om - Or - OX
    E = np.sqrt(Or * (1 + zg) ** 4 + Om * (1 + zg) ** 3 + OX * (1 + zg) ** 2 + OL)
    if np.any(~np.isfinite(E)) or np.any(E <= 0):
        return None
    DC = np.concatenate([[0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zg))])
    if curved and abs(OX) > 1e-12:
        k = np.sqrt(abs(OX))
        DM = np.sinh(k * DC) / k if OX > 0 else np.sin(k * DC) / k
    else:
        DM = DC
    f = lambda arr, z: np.interp(z, zg, arr)
    z0 = bgs[0]
    m = [(z0 * f(DM, z0) ** 2 / f(E, z0)) ** (1 / 3)]
    for z, *_ in bao:
        m += [f(DM, z), 1 / f(E, z)]
    m = np.array(m)
    A = (m @ Ci @ d) / (m @ Ci @ m)
    chi_bao = (d - A * m) @ Ci @ (d - A * m)
    R = np.sqrt(Om) * f(DM, zs_star)
    return chi_bao + ((R - R_obs) / R_err) ** 2


Oms = np.linspace(0.27, 0.36, 91)
OXs = np.linspace(-0.04, 0.04, 161)
for curved, name in [(True, "curvature Omega_k"), (False, "flat w=-1/3 tick fluid Omega_E")]:
    chi = np.array([[model(om, ox, curved) or 1e9 for ox in OXs] for om in Oms])
    L = np.exp(-0.5 * (chi - chi.min()))
    post = L.sum(0); post /= post.sum()
    mean = (post * OXs).sum(); sd = np.sqrt((post * (OXs - mean) ** 2).sum())
    cdf = np.cumsum(post)
    lo, hi = OXs[np.searchsorted(cdf, 0.025)], OXs[np.searchsorted(cdf, 0.975)]
    j = np.unravel_index(chi.argmin(), chi.shape)
    print(f"{name:>32}: mean {mean:+.4f} sd {sd:.4f}  95% [{lo:+.4f}, {hi:+.4f}]  best Om={Oms[j[0]]:.3f} OX={OXs[j[1]]:+.4f} chi2min={chi.min():.2f} (N=14)")
