# F5 analysis (written BEFORE looking at the production data).
# Fixed choices: lambda = 0.005; plateau window for Feynman-Hellmann slopes t = 3..6 (pairs (3,4),(4,5),(5,6),(6,7));
# nucleon effective mass window t = 3..6; pion effective-mass (cosh) window t = 4..7; jackknife over configurations.
# g_A/g_V = s_A / s_V, s = slope of R(t) = [C(+lam) - C(-lam)] / (2 lam C0(t)); g_V^bare = 2 kappa |s_V| (check only).
import glob, sys, numpy as np

lam, kappa, T = 0.005, 0.164, 16
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "f5_out_*.npz"))
D = {}
for f in files:
    d = np.load(f)
    for k in d.files: D.setdefault(k, []).append(d[k])
D = {k: np.concatenate(v) for k, v in D.items()}
N = len(D["plaq"]); print(f"{len(files)} chains, {N} configurations, max solver residual {D['resid'].max():.1e}")
print(f"plaquette {D['plaq'].mean():.4f} +- {D['plaq'].std()/np.sqrt(N):.4f} (pre-registered 0.549 +- 0.003)")


def jack(fn):
    full = fn(np.arange(N))
    js = np.array([fn(np.delete(np.arange(N), i)) for i in range(N)])
    return full, np.sqrt((N - 1) * np.mean((js - js.mean(0)) ** 2, axis=0))


def meff_pion(idx):
    c = D["pion"][idx].mean(0)
    return np.array([np.arccosh((c[t - 1] + c[t + 1]) / (2 * c[t])) for t in range(1, T // 2)])


def C0(idx, a="CVun"):
    return 0.5 * (D[a + "_p"][idx].mean(0) + D[a + "_m"][idx].mean(0)).real


def meff_N(idx):
    c = C0(idx); return np.log(c[:T // 2] / c[1:T // 2 + 1])


def comp(z):  # pick the real or imaginary part, whichever carries the signal (decided on the full sample)
    return z


def R(idx, kind):
    if kind == "V":
        num = (D["CVun_p"][idx].mean(0) - D["CVun_m"][idx].mean(0)); den = C0(idx, "CVun")
    else:
        num = (D["CApol_p"][idx].mean(0) - D["CApol_m"][idx].mean(0)); den = C0(idx, "CAun")
    return num / (2 * lam * den)


# choose component (real/imag) for each R from the full sample, by magnitude in the window
Rv_full, Ra_full = R(np.arange(N), "V"), R(np.arange(N), "A")
pick = lambda z: (lambda x: x.real) if np.abs(z.real[3:8]).sum() >= np.abs(z.imag[3:8]).sum() else (lambda x: x.imag)
pv, pa = pick(Rv_full), pick(Ra_full)


def slopes(idx):
    rv, ra = pv(R(idx, "V")), pa(R(idx, "A"))
    sv = np.diff(rv)[3:7].mean(); sa = np.diff(ra)[3:7].mean()
    return np.array([sv, sa, sa / sv, 2 * kappa * abs(sv)])


mp, dmp = jack(meff_pion); mn, dmn = jack(meff_N)
print("pion m_eff(t=1..7):", " ".join(f"{a:.3f}({b:.3f})" for a, b in zip(mp, dmp)))
print("nucleon m_eff(t=0..7):", " ".join(f"{a:.3f}({b:.3f})" for a, b in zip(mn, dmn)))
mpi = mp[3:7].mean(); mN = mn[3:7].mean()
print(f"m_pi a = {mpi:.3f}, m_N a = {mN:.3f}, m_N/m_pi = {mN/mpi:.2f}  (pre-registered: m_pi a 0.4-0.9, ratio 1.3-2.0)")
print("R_V(t):", np.round(pv(Rv_full)[:9], 3)); print("R_A(t):", np.round(pa(Ra_full)[:9], 3))
print("component used: V", "real" if pv(1j) == 0 else "imag", "| A", "real" if pa(1j) == 0 else "imag")
v, e = jack(slopes)
print(f"s_V = {v[0]:.4f}({e[0]:.4f}), s_A = {v[1]:.4f}({e[1]:.4f})")
print(f"g_V^bare (local current) = {v[3]:.3f}({e[3]:.3f})   [expect ~ 1/Z_V]")
print(f"g_A/g_V (bare ratio) = {v[2]:.3f} +- {e[2]:.3f}   (pre-registered window 0.95-1.35, stat error < 15 %; experiment 1.2754)")
