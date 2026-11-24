# D10: can per-BOND clock rates carry tensor (plus/cross) gravitational-wave polarisations,
#      while per-AGENT clock rates (coincidence rule t_ij = nu_i nu_j) carry only the scalar breathing mode?
# Triangular lattice, bond directions e1=(1,0), e2=(1/2,sqrt3/2), e3=(-1/2,sqrt3/2); field u on sites, v on bonds,
# u' = div(t v), v' = t grad u (zero-sum).  Plane-wave dispersion for small k:  omega^2 = sum_b t_b^2 (k.e_b)^2.
# PRE-REGISTERED:
#   (1) for uniform t_b: omega^2 = g^{ij} k_i k_j with g^{ij} = sum_b t_b^2 e_b^i e_b^j  (3 rates <-> 3 metric comps)
#   (2) a plus wave  g^{ij} = (3/2)(delta + h diag(1,-1)) and a cross wave (3/2)(delta + h offdiag) are both realisable
#       by real bond rates; the light-speed ellipse then matches GR's plus/cross pattern.
#   (3) per-agent rule t_b = nu_i nu_j with uniform nu changes only the trace (breathing); it cannot make plus/cross.
import numpy as np

E = np.array([[1, 0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]])


def exact_dispersion(t, k):
    # exact lattice dispersion of u'' = sum_b t_b^2 (u(x+e_b) - 2u + u(x-e_b)):  omega^2 = sum_b t_b^2 * 4 sin^2(k.e_b/2)
    return np.sum(t**2 * 4 * np.sin((E @ k) / 2) ** 2)


def fit_metric(t, kmag=0.01):
    # sample directions, fit omega^2 = a kx^2 + 2b kx ky + c ky^2
    rows, rhs = [], []
    for th in np.linspace(0, np.pi, 12, endpoint=False):
        k = kmag * np.array([np.cos(th), np.sin(th)])
        rows.append([k[0] ** 2, 2 * k[0] * k[1], k[1] ** 2]); rhs.append(exact_dispersion(t, k))
    a, b, c = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return np.array([[a, b], [b, c]])


def rates_for_metric(G):
    # solve sum_b w_b e_b e_b^T = G for w_b = t_b^2 (3 equations, 3 unknowns)
    Mx = np.array([[e[0] * e[0], e[0] * e[1], e[1] * e[1]] for e in E]).T
    w = np.linalg.solve(Mx, [G[0, 0], G[0, 1], G[1, 1]])
    return np.sqrt(w)


h = 0.05
targets = {
    "flat":     1.5 * np.eye(2),
    "plus":     1.5 * (np.eye(2) + h * np.diag([1, -1])),
    "cross":    1.5 * (np.eye(2) + h * np.array([[0, 1], [1, 0]])),
    "breathing": 1.5 * (1 + h) * np.eye(2),
}
print("(1)(2) per-bond rates realising each metric, and the metric read back from the lattice dispersion")
for name, G in targets.items():
    t = rates_for_metric(G)
    Gfit = fit_metric(t)
    print(f"{name:9s}: bond rates t = {np.round(t, 5)} | fitted g^ij = {np.round(Gfit.flatten(), 5)} | target {np.round(G.flatten(), 5)} "
          f"| max err {np.abs(Gfit - G).max():.1e}")

print("\n(3) per-agent rule t_ij = nu_i nu_j, uniform nu: which metrics are reachable?")
for nu in [1.0, 1.02, 0.97]:
    t = np.full(3, nu * nu)
    Gfit = fit_metric(t)
    trace_part = np.trace(Gfit) / 2
    traceless = Gfit - trace_part * np.eye(2)
    print(f"nu={nu}: g^ij = {np.round(Gfit.flatten(), 5)}, traceless (plus/cross) part max = {np.abs(traceless).max():.1e}")
print("-> per-agent clocks only rescale g (breathing / conformal). A plus or cross wave needs bond-dependent rates.")

print("\n(2b) light-speed pattern of a plus wave vs GR: c(theta)^2 = g^ij n_i n_j / 1.5 ; GR plus: 1 + h cos(2 theta)")
t = rates_for_metric(targets["plus"])
for th in [0, 30, 45, 60, 90]:
    n = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
    c2 = exact_dispersion(t, 0.01 * n) / 0.01**2 / 1.5
    print(f"  theta={th:2d} deg: c^2 = {c2:.5f}   GR plus pattern 1+h cos2theta = {1 + h*np.cos(2*np.radians(th)):.5f}")
