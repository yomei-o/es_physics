# C24: add Pantheon+ supernovae to the C20b/C22 fit (DESI DR2 BAO + Planck distance priors).
# usage: python -I scripts/c24_tick_fluid_with_sn.py <dir containing Pantheon+SH0ES.dat and Pantheon+SH0ES_STAT+SYS.cov>
# Data: https://github.com/PantheonPlusSH0ES/DataRelease (4_DISTANCES_AND_COVAR). Standard cosmology cut zHD > 0.01,
#   m_b_corr with STAT+SYS covariance, absolute magnitude M marginalized analytically. mu = 5 log10(D_L/10pc), D_L = (1+zHEL) D_M(zHD).
# PRE-REGISTERED: SN prefer higher Omega_m (~0.33 in LCDM) -> the negative Omega_E of C22 moves toward 0 (|mean|/sd < 1.5), sd ~0.007;
#   curvature essentially unchanged.
import sys
import numpy as np

ddir = sys.argv[1]
src = open("scripts/c20_tick_fluid_bound_full.py", encoding="utf8").read().split("Oms = np.linspace")[0]
ns = {}; exec(src, ns)
chi2_bc, background, zg = ns["chi2"], ns["background"], ns["zg"]
nm_src = open("scripts/c22_tick_fluid_profile_nm.py", encoding="utf8").read()
nm_src = nm_src[nm_src.index("def nelder_mead"):nm_src.index("for curved, name, OXs")]
exec(nm_src, ns); nelder_mead = ns["nelder_mead"]

cols = open(f"{ddir}/Pantheon+SH0ES.dat").readline().split()
raw = np.genfromtxt(f"{ddir}/Pantheon+SH0ES.dat", skip_header=1, usecols=(cols.index("zHD"), cols.index("zHEL"), cols.index("m_b_corr")))
cov_flat = np.loadtxt(f"{ddir}/Pantheon+SH0ES_STAT+SYS.cov")
n = int(cov_flat[0]); cov = cov_flat[1:].reshape(n, n)
sel = raw[:, 0] > 0.01
zHD, zHEL, mb = raw[sel, 0], raw[sel, 1], raw[sel, 2]
Ci = np.linalg.inv(cov[np.ix_(sel, sel)])
one = np.ones(sel.sum()); a1 = one @ Ci @ one
print(f"SNe used: {sel.sum()}")


def chi2_sn(Om, h, OX, curved):
    E = background(Om, h, OX)
    DC = np.concatenate([[0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zg))])
    if curved and abs(OX) > 1e-12:
        k = np.sqrt(abs(OX)); DM = np.sinh(k * DC) / k if OX > 0 else np.sin(k * DC) / k
    else:
        DM = DC
    mu = 5 * np.log10((1 + zHEL) * np.interp(zHD, zg, DM))       # up to a constant (absorbed in M)
    r = mb - mu
    b = one @ Ci @ r
    return r @ Ci @ r - b * b / a1                                 # analytic marginalization over M (flat prior)


for curved, name, OXs in [(True, "curvature Omega_k", np.linspace(-0.006, 0.010, 17)),
                          (False, "flat tick fluid Omega_E", np.linspace(-0.04, 0.02, 25))]:
    prof, x = [], [0.31, 0.68, 0.02236]
    for ox in OXs:
        f = lambda v: chi2_bc(v[0], v[1], ox, curved, v[2]) + chi2_sn(v[0], v[1], ox, curved)
        best = (None, 1e18)
        for start in ([0.31, 0.68, 0.02236], list(x)):
            xb, vb = nelder_mead(f, start, [0.01, 0.01, 0.0002])
            xb, vb = nelder_mead(f, xb, [0.002, 0.002, 0.00005])
            if vb < best[1]: best = (xb, vb)
        x = best[0]; prof.append(best[1])
    prof = np.array(prof); dchi = prof - prof.min()
    L = np.exp(-0.5 * dchi); L /= L.sum()
    mean = (L * OXs).sum(); sd = np.sqrt((L * (OXs - mean) ** 2).sum())
    j = np.argmin(prof)
    print(f"{name:>26}: mean {mean:+.4f} sd {sd:.4f}, best {OXs[j]:+.4f}, sqrt(dchi2 at 0) = {np.sqrt(max(np.interp(0.0, OXs, dchi), 0)):.2f}")
    print("   dchi2:", " ".join(f"{v:.1f}" for v in dchi))
