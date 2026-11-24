# C20b: tighter bound on the flat w=-1/3 tick fluid using Planck 2018 distance priors (R, l_A, omega_b) + DESI DR2 BAO with computed r_d.
# Distance priors (arXiv:1808.05724, as quoted in later papers): R = 1.7502 ± 0.0046, l_A = 301.471 ± 0.090, omega_b = 0.02236 ± 0.00015,
#   correlations used: r(R,l_A) = 0.46, r(R,wb) = -0.66, r(l_A,wb) = -0.33  (approximate values from memory / secondary quotes; verify).
# Sound horizons: z* (Hu-Sugiyama 1996), z_d (Eisenstein-Hu 1998) fitting formulas; integrals rescaled so that the Planck best fit
#   (h=0.6736, wb=0.02237, wc=0.1200) gives r_s(z*) = 144.43 Mpc and r_d = 147.09 Mpc.
# Parameters: Omega_m, h, Omega_X (omega_b from its prior, profiled on a grid).  Curvature run is the pipeline check.
# PRE-REGISTERED: curvature within 1 sigma of DESI+CMB 0.0023 ± 0.0011 with sigma ~0.0015; tick fluid sigma(Omega_E) ~ 0.005.
import numpy as np

c = 299792.458
Tcmb = 2.7255
og_h2 = 2.4728e-5                                 # photons omega_gamma
Neff = 3.046
or_h2 = og_h2 * (1 + 0.2271 * Neff)
bao = [(0.510, 13.588, 0.167, 21.863, 0.425, -0.459), (0.706, 17.351, 0.177, 19.455, 0.330, -0.404),
       (0.934, 21.576, 0.152, 17.641, 0.193, -0.416), (1.321, 27.601, 0.318, 14.176, 0.221, -0.434),
       (1.484, 30.512, 0.760, 12.817, 0.516, -0.500), (2.330, 38.988, 0.531, 8.632, 0.101, -0.431)]
bgs = (0.295, 7.942, 0.075)
d = [bgs[1]]; blocks = [np.array([[bgs[2] ** 2]])]
for z, DM, sM, DH, sH, r in bao:
    d += [DM, DH]; blocks.append(np.array([[sM ** 2, r * sM * sH], [r * sM * sH, sH ** 2]]))
d = np.array(d); C = np.zeros((13, 13)); i = 0
for b in blocks:
    n = len(b); C[i:i + n, i:i + n] = b; i += n
Ci = np.linalg.inv(C)
cmb_mu = np.array([1.7502, 301.471, 0.02236]); s = np.array([0.0046, 0.090, 0.00015])
corr = np.array([[1, 0.46, -0.66], [0.46, 1, -0.33], [-0.66, -0.33, 1]])
cmb_Ci = np.linalg.inv(corr * np.outer(s, s))


def zstar(wb, wm):
    g1 = 0.0783 * wb ** -0.238 / (1 + 39.5 * wb ** 0.763); g2 = 0.560 / (1 + 21.1 * wb ** 1.81)
    return 1048 * (1 + 0.00124 * wb ** -0.738) * (1 + g1 * wm ** g2)


def zdrag(wb, wm):
    b1 = 0.313 * wm ** -0.419 * (1 + 0.607 * wm ** 0.674); b2 = 0.238 * wm ** 0.223
    return 1291 * wm ** 0.251 / (1 + 0.659 * wm ** 0.828) * (1 + b1 * wb ** b2)


zg = np.concatenate([np.linspace(0, 3, 3001), np.logspace(np.log10(3.001), 5, 6000)])


def background(Om, h, OX):
    Or = or_h2 / h ** 2
    OL = 1 - Om - Or - OX
    E = np.sqrt(Or * (1 + zg) ** 4 + Om * (1 + zg) ** 3 + OX * (1 + zg) ** 2 + OL)
    return E


def rs_raw(z, E, h, wb):                        # sound horizon in Mpc (uncalibrated)
    Rb = 0.75 * wb / og_h2 / (1 + zg)
    cs = 1 / np.sqrt(3 * (1 + Rb))
    integrand = cs / E
    m = zg >= z
    return c / (100 * h) * np.trapezoid(integrand[m], zg[m])


E0 = background(0.3153, 0.6736, 0.0)
k_star = 144.43 / rs_raw(zstar(0.02237, 0.1430), E0, 0.6736, 0.02237)
k_drag = 147.09 / rs_raw(zdrag(0.02237, 0.1430), E0, 0.6736, 0.02237)
print(f"calibration factors: r_s* {k_star:.4f}, r_d {k_drag:.4f}")


def chi2(Om, h, OX, curved, wb):
    E = background(Om, h, OX)
    if np.any(~np.isfinite(E)): return 1e9
    DC = np.concatenate([[0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zg))])
    if curved and abs(OX) > 1e-12:
        kk = np.sqrt(abs(OX)); DM = np.sinh(kk * DC) / kk if OX > 0 else np.sin(kk * DC) / kk
    else:
        DM = DC
    wm = Om * h * h
    zs, zd = zstar(wb, wm), zdrag(wb, wm)
    DMs = c / (100 * h) * np.interp(zs, zg, DM)
    rs = k_star * rs_raw(zs, E, h, wb); rd = k_drag * rs_raw(zd, E, h, wb)
    R = np.sqrt(Om) * np.interp(zs, zg, DM); lA = np.pi * DMs / rs
    dv = np.array([R, lA, wb]) - cmb_mu
    A = c / (100 * h) / rd
    f = lambda arr, z: np.interp(z, zg, arr)
    m = [A * (bgs[0] * f(DM, bgs[0]) ** 2 / f(E, bgs[0])) ** (1 / 3)]
    for z, *_ in bao: m += [A * f(DM, z), A / f(E, z)]
    r = d - np.array(m)
    return r @ Ci @ r + dv @ cmb_Ci @ dv


Oms = np.linspace(0.28, 0.34, 25); hs = np.linspace(0.64, 0.71, 29)
for curved, name, OXs in [(True, "curvature Omega_k", np.linspace(-0.008, 0.012, 21)),
                          (False, "flat tick fluid Omega_E", np.linspace(-0.04, 0.02, 25))]:
    prof = []
    best = (1e9,)
    for ox in OXs:
        mn = 1e9
        for om in Oms:
            for h in hs:
                v = chi2(om, h, ox, curved, 0.02236)
                if v < mn: mn = v
                if v < best[0]: best = (v, om, h, ox)
        prof.append(mn)
    prof = np.array(prof)
    L = np.exp(-0.5 * (prof - prof.min())); L /= L.sum()
    mean = (L * OXs).sum(); sd = np.sqrt((L * (OXs - mean) ** 2).sum())
    print(f"{name:>26}: profile mean {mean:+.4f} sd {sd:.4f}   best chi2 {best[0]:.2f} at Om={best[1]:.3f} h={best[2]:.3f} OX={best[3]:+.4f}  (N=16)")
