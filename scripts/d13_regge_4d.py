# D13: spacetime loops. 4D (Euclidean) Regge action on the Kuhn-triangulated hypercubic lattice.
# 15 bonds per agent (all nonzero {0,1}^4 vectors), 24 four-simplices per hypercube.
# Hinges are triangles; loop frustration = deficit angle of a triangle; energy S = sum_t A_t delta_t.
# At the flat lattice (delta=0): Hessian H_ef = sum_t (dA_t/dl_e)(d delta_t/dl_f).
# PRE-REGISTERED:
#  (a) 4 gauge zero modes per k (agents can be moved in 4 directions)
#  (b) after removing the stiff lattice modes, the quadratic form on metric-like perturbations equals the 4D
#      Euclidean Fierz-Pauli form (fully SO(4) invariant) with ratio -> 1, for k along a "time" axis, a "space"
#      axis and mixed diagonals  => after Wick rotation Lorentz invariant => c_gw = c
#  (c) 5 stiff lattice modes (15 bonds - 10 metric components)
import numpy as np, itertools

D = [np.array(v) for v in itertools.product([0, 1], repeat=4) if any(v)]
NB = len(D)                         # 15
dindex = {tuple(d): i for i, d in enumerate(D)}
PERMS = list(itertools.permutations(range(4)))


def simplex(base, perm):
    pts = [np.array(base)]
    cur = np.array(base)
    for ax in perm:
        cur = cur.copy(); cur[ax] += 1; pts.append(cur)
    return [tuple(p) for p in pts]


def edge_key(p, q):
    d = tuple(np.subtract(q, p))
    if d in dindex: return (tuple(p), dindex[d])
    return (tuple(q), dindex[tuple(-np.array(d))])


def simplices_containing(p, q):
    out = []
    lo = np.minimum(p, q) - 1
    for base in itertools.product(*[range(lo[i], lo[i] + 2) for i in range(4)]):
        for perm in PERMS:
            s = simplex(base, perm)
            if p in s and q in s: out.append(s)
    return out


def all_simplices_containing_triangle(tri):
    out = []
    lo = np.min(np.array(tri), 0) - 1
    for base in itertools.product(*[range(lo[i], lo[i] + 2) for i in range(4)]):
        for perm in PERMS:
            s = simplex(base, perm)
            if all(v in s for v in tri): out.append(s)
    return out


def length(p, q, pert):
    k = edge_key(p, q)
    return np.linalg.norm(np.subtract(q, p)) + pert.get(k, 0.0)


def dihedral(s, tri, pert):
    a = tri[0]; others = [v for v in s if v != a]
    L = {(i, j): length(i, j, pert) for i in s for j in s if i != j}
    G = np.array([[(L[(a, i)]**2 + L[(a, j)]**2 - (L[(i, j)]**2 if i != j else 0)) / 2 for j in others] for i in others])
    X = np.linalg.cholesky(G)                       # rows: coordinates of 'others' relative to a
    pos = {v: X[n] for n, v in enumerate(others)}; pos[a] = np.zeros(4)
    b, c = tri[1], tri[2]
    d, e = [v for v in s if v not in tri]
    Q, _ = np.linalg.qr(np.stack([pos[b], pos[c]], 1))
    def perp(x): return x - Q @ (Q.T @ x)
    u, w = perp(pos[d]), perp(pos[e])
    return np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1))


def triangles_of(s):
    return [tuple(sorted(t)) for t in itertools.combinations(s, 3)]


# flat check
origin = (0, 0, 0, 0)
t_test = (origin, (1, 0, 0, 0), (1, 1, 0, 0))
tot = sum(dihedral(s, t_test, {}) for s in all_simplices_containing_triangle(t_test))
print(f"flat check: sum of dihedral angles around a triangle = {tot:.12f} (2pi = {2*np.pi:.12f})")


def area_grad(tri):
    # dA/dl for each edge of the triangle (Heron)
    p, q, r = tri
    a, b, c = np.linalg.norm(np.subtract(q, r)), np.linalg.norm(np.subtract(p, r)), np.linalg.norm(np.subtract(p, q))
    s = (a + b + c) / 2; A = np.sqrt(s * (s - a) * (s - b) * (s - c))
    g = {}
    for (x, y, z, l1, l2, l3) in [(q, r, p, a, b, c), (p, r, q, b, a, c), (p, q, r, c, a, b)]:
        g[edge_key(x, y)] = l1 * (l2**2 + l3**2 - l1**2) / (8 * A)
    return g


# Hessian entries H[(R, a), (0, b)]
eps = 1e-6
H = {}
for b in range(NB):
    p, q = origin, tuple(D[b])
    simps = simplices_containing(p, q)
    tris = set(t for s in simps for t in triangles_of(s))
    for t in tris:
        dd = 0.0
        for s in simps:
            if all(v in s for v in t):
                dd -= (dihedral(s, t, {(origin, b): eps}) - dihedral(s, t, {(origin, b): -eps})) / (2 * eps)
        if abs(dd) < 1e-12: continue
        for ek, gA in area_grad(t).items():
            key = (ek[0], ek[1], b)
            H[key] = H.get(key, 0.0) + gA * dd
print("nonzero Hessian couplings to the 15 bonds at the origin:", len(H))


def Hk(k):
    M = np.zeros((NB, NB), complex)
    for (R, a, b), v in H.items():
        M[a, b] += v * np.exp(-1j * np.dot(k, R))
    return 0.5 * (M + M.conj().T)


idx = [(i, j) for i in range(4) for j in range(i, 4)]       # 10 metric components


def Mmap(k):
    M = np.zeros((NB, 10), complex)
    for a, d in enumerate(D):
        u = d / np.linalg.norm(d); ph = np.exp(1j * k @ d / 2)
        for c, (i, j) in enumerate(idx):
            M[a, c] = 0.5 * np.linalg.norm(d) * u[i] * u[j] * (1 if i == j else 2) * ph
    return M


def VFP(k):
    K2 = k @ k
    def to_mat(v):
        Hh = np.zeros((4, 4))
        for a, (i, j) in enumerate(idx): Hh[i, j] = Hh[j, i] = v[a]
        return Hh
    def f(v):
        Hh = to_mat(v); h = np.trace(Hh); kh = Hh @ k
        return 0.25 * K2 * np.sum(Hh**2) - 0.5 * kh @ kh + 0.5 * (k @ Hh @ k) * h - 0.25 * K2 * h * h
    Q = np.zeros((10, 10)); I = np.eye(10)
    for a in range(10):
        for b in range(10):
            Q[a, b] = 0.5 * (f(I[a] + I[b]) - f(I[a]) - f(I[b]))
    return Q


print("\n(a)(c) spectrum of the 15x15 block (k along time axis), eigenvalue/k^2:")
for kk in [0.1, 0.2]:
    k = np.array([0, 0, 0, kk])
    w = np.linalg.eigvalsh(-Hk(k))
    print(f"  |k|={kk}: zero modes {np.sum(np.abs(w) < 1e-7)}, k^2-scaling modes {np.round(w[(np.abs(w) > 1e-7) & (np.abs(w) < 5 * kk)] / kk**2, 3)}, stiff {np.sum(np.abs(w) >= 5 * kk)}")

print("\n(b) effective long-wavelength form vs 4D Euclidean Fierz-Pauli (eigenvalue ratios on nonzero modes)")
dirs = {"time (0,0,0,1)": [0, 0, 0, 1], "space (1,0,0,0)": [1, 0, 0, 0], "time+space (1,0,0,1)": [1, 0, 0, 1],
        "time-space (1,0,0,-1)": [1, 0, 0, -1], "(1,2,-1,3)": [1, 2, -1, 3]}
for name, dv in dirs.items():
    kd = np.array(dv, float) / np.linalg.norm(dv)
    for kk in [0.05, 0.1]:
        k = kd * kk
        Hm = -Hk(k)
        w, U = np.linalg.eigh(Hm)
        stiff = U[:, np.abs(w) >= 5 * kk]
        P = np.eye(NB) - stiff @ stiff.conj().T
        Mk = P @ Mmap(k)
        A = Mk.conj().T @ Hm @ Mk; A = 0.5 * (A + A.conj().T).real
        wl = np.sort(np.linalg.eigvalsh(A)); wf = np.sort(np.linalg.eigvalsh(VFP(k)))
        nz = np.abs(wf) > 1e-12
        print(f"  {name:22s} |k|={kk}: stiff={stiff.shape[1]} nonzero FP modes={nz.sum()} ratio {np.round(wl[nz] / wf[nz], 3)}")
