# C22 (review fix for C20b): continuous profile likelihood.  For each Omega_X, minimize chi2 over (Omega_m, h, omega_b)
# with Nelder-Mead (numpy-only implementation), instead of a coarse grid (l_A has 3e-4 relative error -> grid steps were too coarse).
# Reuses the model of scripts/c20_tick_fluid_bound_full.py.
# PRE-REGISTERED: Omega_E ≈ -0.02 ± 0.009 (unchanged), curvature ≈ +0.0027 ± 0.0013, smooth profiles.
import numpy as np
src = open("scripts/c20_tick_fluid_bound_full.py", encoding="utf8").read().split("Oms = np.linspace")[0]
ns = {}; exec(src, ns); chi2 = ns["chi2"]


def nelder_mead(f, x0, step, iters=300):
    n = len(x0); pts = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float); p[i] += step[i]; pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i])
    i = int(np.argmin(vals)); return pts[i], vals[i]


for curved, name, OXs in [(True, "curvature Omega_k", np.linspace(-0.006, 0.010, 17)),
                          (False, "flat tick fluid Omega_E", np.linspace(-0.05, 0.015, 27))]:
    prof, x = [], [0.31, 0.68, 0.02236]
    for ox in OXs:
        f = lambda v: chi2(v[0], v[1], ox, curved, v[2])
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
    sig0 = np.sqrt(max(np.interp(0.0, OXs, dchi), 0))
    print(f"{name:>26}: mean {mean:+.4f} sd {sd:.4f}, best {OXs[j]:+.4f} chi2min {prof.min():.2f}, sqrt(dchi2 at 0) = {sig0:.2f}")
    print("   dchi2 profile:", " ".join(f"{v:.1f}" for v in dchi))
