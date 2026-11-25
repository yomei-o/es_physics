# D15: strong-field static solution of the bond-clock network (= Regge calculus), Wong-type construction.
# Nested regular polyhedral shells (p equilateral faces per vertex: 3 tetra, 4 octa, 5 icosa) joined by radial struts.
# Blocks = flat right triangular frusta. Loop frustration = deficit angle. Vacuum time-symmetric constraint per vertex:
#   p * a_i * delta_tan(i) + m_{i-1} delta_s(i-1) + m_i delta_s(i) = 0      (discrete R^(3) = 0)
# Mirror-symmetric throat at shell 0. Lapse from discrete Laplace equation on the strut network.
# Radii are vertex (circum)radii r = k_p * a consistently, so the ratio b_c / M does not depend on that choice.
# PRE-REGISTERED (es_physics_notes.html 14.2): Misner-Sharp mass constant within 2 %; b_c/M near 3 sqrt3 = 5.196, not 2e = 5.437,
# provided the angular-discretisation error (spread across polyhedra) is below 2 %.
import numpy as np
from scipy.optimize import brentq

KP = {3: np.sqrt(6) / 4, 4: 1 / np.sqrt(2), 5: np.sin(2 * np.pi / 5)}   # circumradius / edge
NV = {3: 4, 4: 6, 5: 12}                                                  # vertices per shell


def frustum(a, b, m):
    ra, rb = a / np.sqrt(3), b / np.sqrt(3)
    h2 = m * m - (rb - ra) ** 2
    if h2 <= 0: return None
    h = np.sqrt(h2); ph = 2 * np.pi * np.arange(3) / 3
    B = np.stack([ra * np.cos(ph), ra * np.sin(ph), np.zeros(3)], 1)
    T = np.stack([rb * np.cos(ph), rb * np.sin(ph), np.full(3, h)], 1)
    cen = np.array([0, 0, h / 2])
    def side_normal(k):
        k1 = (k + 1) % 3
        n = np.cross(B[k1] - B[k], T[k] - B[k]); n /= np.linalg.norm(n)
        return n if np.dot(n, (B[k] + B[k1] + T[k] + T[k1]) / 4 - cen) > 0 else -n
    n0, n2 = side_normal(0), side_normal(2)
    inter = lambda u, v: np.pi - np.arccos(np.clip(np.dot(u, v), -1, 1))
    th_lat = inter(n0, n2)                       # at lateral edge B0-T0 (between sides 2 and 0)
    th_bot = inter(n0, np.array([0, 0, -1.0]))
    th_top = inter(n0, np.array([0, 0, 1.0]))
    return th_lat, th_bot, th_top


def build(p, r0, q, nshell):
    r = r0 * q ** np.arange(nshell + 1)
    a = r / KP[p]
    m = np.zeros(nshell)
    def ds(i, mi):
        f = frustum(a[i], a[i + 1], mi); return 2 * np.pi - p * f[0], f
    lo = lambda i: abs(a[i + 1] - a[i]) / np.sqrt(3) * (1 + 1e-12) + 1e-15
    # throat
    def g0(m0):
        d, f = ds(0, m0); return p * a[0] * (2 * np.pi - 4 * f[1]) + 2 * m0 * d
    m[0] = brentq(g0, lo(0), 200 * (r[1] - r[0]) + r[0])
    for i in range(1, nshell):
        fprev = frustum(a[i - 1], a[i], m[i - 1]); dprev = 2 * np.pi - p * fprev[0]
        def gi(mi):
            d, f = ds(i, mi)
            return p * a[i] * (2 * np.pi - 2 * fprev[2] - 2 * f[1]) + m[i - 1] * dprev + mi * d
        hi = 50 * (r[i + 1] - r[i]) + 1e-9
        m[i] = brentq(gi, lo(i), hi)
    return r, m


def analyse(r, m):
    rb = 0.5 * (r[1:] + r[:-1]); dr = np.diff(r)
    Mms = rb / 2 * (1 - (dr / m) ** 2)
    N = np.concatenate([[0.0], np.cumsum(m / (4 * np.pi * rb ** 2))])   # flux-conserving Laplace, throat N=0
    n = len(r); sel = slice(int(0.7 * n), n)
    A = np.vstack([np.ones(n)[sel], -1 / r[sel]]).T
    Ninf, C = np.linalg.lstsq(A, N[sel], rcond=None)[0]
    N = N / Ninf
    M = np.mean(Mms[int(0.7 * len(Mms)):])
    bc = np.min(r[1:] / N[1:])
    return M, Mms, N, bc


def schwarzschild_grid(r0, q, nshell):
    """same radial grid, exact Schwarzschild proper distances, same Laplace discretisation (isolates radial error)"""
    M = r0 / 2; r = r0 * q ** np.arange(nshell + 1)
    ell = lambda x: np.sqrt(x * (x - 2 * M)) + 2 * M * np.log(np.sqrt(x / (2 * M)) + np.sqrt(x / (2 * M) - 1))
    m = np.diff(ell(r))
    return r, m


# flat check: concentric polyhedra with flat struts -> deficits ~ 0
for p in (3, 4, 5):
    a1, a2 = 1.0, 1.1; f = frustum(a1, a2, KP[p] * (a2 - a1))
    print(f"flat check p={p}: strut deficit {2*np.pi - p*f[0]:+.2e}")

print("\n p  q      | M (Misner-Sharp)  spread of M_i over shells | b_c/M   | same grid, exact Schwarzschild b_c/M")
for q in [1.05, 1.02, 1.01]:
    nshell = int(np.log(400) / np.log(q))
    rs, ms = schwarzschild_grid(1.0, q, nshell); Ms, _, _, bcs = analyse(rs, ms)
    for p in (3, 4, 5):
        try:
            r, m = build(p, 1.0, q, nshell)
        except Exception as e:
            print(f" {p}  {q}: failed ({e})"); continue
        M, Mi, N, bc = analyse(r, m)
        spread = (Mi[5:].max() - Mi[5:].min()) / M
        print(f" {p}  {q:.2f}   | {M:.4f} (r0/2 = 0.5)   {100*spread:5.1f} %                  | {bc/M:.3f}  | {bcs/Ms:.3f}")
print(f"\nreference: GR 3 sqrt3 = {3*np.sqrt(3):.3f}, exponential metric 2e = {2*np.e:.3f}")
