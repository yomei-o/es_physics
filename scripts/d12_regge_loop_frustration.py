# D12: does "loop frustration" energy on an actual 3D network give the gravitational potential (Fierz-Pauli)?
# Network: periodic cubic lattice, Kuhn triangulation (6 tetrahedra per cube), 7 bonds per agent:
#   (1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1).  Bond length l = 1/(bond rate).
# Loop frustration at a bond = deficit angle  delta_e = 2*pi - sum of tetrahedral dihedral angles around it
# (how far the loops of agents around that bond fail to close).  Energy S = sum_e l_e delta_e (3D Regge action).
# Schlaefli identity: dS/dl_e = delta_e  ->  Hessian = d delta_e / d l_f.
# PRE-REGISTERED (per Fourier mode k, 7x7 block):
#   (a) 3 zero modes (gauge: moving agents around changes no loop frustration) for every k
#   (b) at small k the quadratic form restricted to metric-like perturbations h_ij is proportional to the
#       Fierz-Pauli potential V_FP(k) (2 TT positive, 1 scalar negative, 3 gauge zero) -- same as d11 R2
#   (c) one extra lattice mode (7 bonds vs 6 metric components)
import numpy as np
import itertools

import sys
L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
DIRS = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)])
dir_index = {tuple(d): i for i, d in enumerate(DIRS)}
NV = L**3


def vid(x, y, z): return ((x % L) * L + (y % L)) * L + (z % L)


def edge_of(p, q):
    # edge between lattice points p,q (q-p in +-DIRS); returns global edge index
    d = np.array(q) - np.array(p)
    if tuple(d) in dir_index:
        return vid(*p) * 7 + dir_index[tuple(d)]
    return vid(*q) * 7 + dir_index[tuple(-d)]


tets = []
for x, y, z in itertools.product(range(L), repeat=3):
    base = np.array([x, y, z])
    for perm in itertools.permutations(range(3)):
        pts = [base.copy()]
        cur = base.copy()
        for ax in perm:
            cur = cur.copy(); cur[ax] += 1; pts.append(cur)
        tets.append(pts)
NT = len(tets)
# for each tet: its 6 edges in order (01,02,03,12,13,23), and for each edge the opposite pair
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
tet_edges = np.array([[edge_of(t[a], t[b]) for a, b in pairs] for t in tets])
NE = NV * 7
l0 = np.array([np.linalg.norm(DIRS[e % 7]) for e in range(NE)])


def dihedrals(l):
    # l: (NT,6) lengths in order 01,02,03,12,13,23. Embed and compute dihedral angle at each edge.
    d01, d02, d03, d12, d13, d23 = [l[:, i] for i in range(6)]
    p1x = d01
    p2x = (d01**2 + d02**2 - d12**2) / (2 * d01); p2y = np.sqrt(np.maximum(d02**2 - p2x**2, 1e-30))
    p3x = (d01**2 + d03**2 - d13**2) / (2 * d01)
    p3y = (d02**2 + d03**2 - d23**2 - 2 * p2x * p3x) / (2 * p2y)
    p3z = np.sqrt(np.maximum(d03**2 - p3x**2 - p3y**2, 1e-30))
    z = np.zeros_like(d01)
    P = np.stack([np.stack([z, z, z], 1), np.stack([p1x, z, z], 1), np.stack([p2x, p2y, z], 1), np.stack([p3x, p3y, p3z], 1)], 1)
    out = np.zeros((len(l), 6))
    for k, (a, b) in enumerate(pairs):
        c, d = [i for i in range(4) if i not in (a, b)]
        e = P[:, b] - P[:, a]
        n1 = np.cross(e, P[:, c] - P[:, a]); n2 = np.cross(e, P[:, d] - P[:, a])
        cos = np.einsum('ij,ij->i', n1, n2) / (np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1))
        out[:, k] = np.arccos(np.clip(cos, -1, 1))
    return out


def deficits(l):
    th = dihedrals(l[tet_edges])
    s = np.zeros(NE); np.add.at(s, tet_edges.ravel(), th.ravel())
    return 2 * np.pi - s


d0 = deficits(l0)
print(f"flat lattice: max |deficit| = {np.abs(d0).max():.2e}  (loops close: no frustration)")

# Hessian columns for edges of vertex 0 only (translation invariance), central differences
eps = 1e-5
cols = np.zeros((NE, 7))
for a in range(7):
    lp = l0.copy(); lp[a] += eps; lm = l0.copy(); lm[a] -= eps
    cols[:, a] = (deficits(lp) - deficits(lm)) / (2 * eps)
# H(k)_{ab} = sum_R H_{(R,a),(0,b)} e^{-ik.R}  (vertex 0 at origin)
coords = np.array([[(v // (L * L)), (v // L) % L, v % L] for v in range(NV)])


def Hk(kvec):
    ph = np.exp(-1j * (coords @ kvec))
    M = np.einsum('v,vab->ab', ph, cols.reshape(NV, 7, 7))
    return 0.5 * (M + M.conj().T)


print("\n(a)(c) eigenvalues of the 7x7 frustration-energy block per Fourier mode (k along z, units 2pi/L):")
for n in [1, 2, 4]:
    kv = np.array([0, 0, 2 * np.pi * n / L])
    w = np.linalg.eigvalsh(Hk(kv))
    nz = np.sum(np.abs(w) < 1e-6)
    print(f"  k = {2*np.pi*n/L:.3f}: eigenvalues {np.round(w, 5)}  -> zero modes: {nz};  nonzero/k^2: {np.round(w[np.abs(w)>1e-6]/(2*np.pi*n/L)**2, 3)}")

# (b) compare with Fierz-Pauli on metric-like perturbations: delta l_e = (1/2) l_e  u_e.h.u_e  (u_e unit vector)
idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
def Mmap():
    M = np.zeros((7, 6))
    for a, d in enumerate(DIRS):
        u = d / np.linalg.norm(d)
        for c, (i, j) in enumerate(idx):
            M[a, c] = 0.5 * np.linalg.norm(d) * (u[i] * u[j] * (1 if i == j else 2))
    return M


def VFP(kvec):
    K2 = kvec @ kvec
    def to_mat(v):
        H = np.zeros((3, 3))
        for a, (i, j) in enumerate(idx): H[i, j] = H[j, i] = v[a]
        return H
    def f(v):
        H = to_mat(v); h = np.trace(H); kh = H @ kvec
        return 0.25 * K2 * np.sum(H**2) - 0.5 * kh @ kh + 0.5 * (kvec @ H @ kvec) * h - 0.25 * K2 * h * h
    Q = np.zeros((6, 6)); I = np.eye(6)
    for a in range(6):
        for b in range(6):
            Q[a, b] = 0.5 * (f(I[a] + I[b]) - f(I[a]) - f(I[b]))
    return Q


print("\n(b) restricted to metric-like perturbations: eigenvalues of M^T H(k) M  vs  Fierz-Pauli V_FP(k)")
M = Mmap()
for n in [1, 2]:
    for kdir in [np.array([0, 0, 1.0]), np.array([1, 1, 0]) / np.sqrt(2)]:
        kv = kdir * 2 * np.pi * n / L
        # lattice-phase correction: perturbation of bond (R,a) for long-wavelength h(x) = h e^{ik.x} at bond midpoint
        ph = np.array([np.exp(1j * kv @ d / 2) for d in DIRS])
        Mk = M * ph[:, None]
        A = (Mk.conj().T @ Hk(kv) @ Mk).real
        wl = np.sort(np.linalg.eigvalsh(0.5 * (A + A.T)))
        wf = np.sort(np.linalg.eigvalsh(VFP(kv)))
        ratio = wl[np.abs(wf) > 1e-9] / wf[np.abs(wf) > 1e-9]
        print(f"  k dir {np.round(kdir,3)}: lattice {np.round(wl, 5)}")
        print(f"                 FP      {np.round(wf, 5)}   ratio on nonzero modes {np.round(ratio, 3)}")


print("\n(b') relax the stiff non-geometric lattice mode (long-wavelength effective energy), compare with -V_FP")
for n in [1, 2]:
    for kint in [np.array([0, 0, 1.0]), np.array([1, 1, 0.0]), np.array([1, 2, 3.0]), np.array([1, -1, 0.0])]:
        kv = kint * 2 * np.pi * n / L          # lattice momentum: every component a multiple of 2pi/L
        kdir = kint / np.linalg.norm(kint)
        Hm = -Hk(kv)                                    # sign convention: energy = -(Regge action)
        w, U = np.linalg.eigh(Hm)
        q = U[:, np.argmax(w)]                          # stiff mode
        P = np.eye(7) - np.outer(q, q.conj())
        ph = np.array([np.exp(1j * kv @ d / 2) for d in DIRS])
        Mk = P @ (M * ph[:, None])
        A = (Mk.conj().T @ Hm @ Mk).real
        wl = np.sort(np.linalg.eigvalsh(0.5 * (A + A.T))); wf = np.sort(np.linalg.eigvalsh(VFP(kv)))
        nz = np.abs(wf) > 1e-9
        print(f"  |k|={np.linalg.norm(kv):.3f} dir {np.round(kdir,3)}: lattice {np.round(wl,5)} | FP {np.round(wf,5)} | ratio {np.round(wl[nz]/wf[nz],3)}")
