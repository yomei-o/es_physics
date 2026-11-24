# D11: mode counting for dynamics of bond rates h_ij (spatial metric perturbation, 6 components) in 3D.
# Fourier mode k = (0,0,k). Kinetic and potential are quadratic forms on the 6-dim space of symmetric h.
# Rules:
#  R1 naive     : T = 1/4 hdot_ij hdot_ij,   V = 1/4 k^2 h_ij h_ij        (each bond its own wave)
#  R2 background-independent: T = same (ES-like positive update cost),
#                 V = linearised Einstein-Hilbert spatial part (Fierz-Pauli), invariant under h -> h + k xi + xi k
#  R2' same with DeWitt kinetic T = 1/4 (hdot_ij hdot_ij - hdot^2)   (what GR actually has)
#  R3 = R2 + one scalar constraint C(h) = k_i k_j h_ij - k^2 h = 0 (linearised Hamiltonian constraint,
#       read as "local update budget conserved"), physical space = constraint surface / gauge.
# PRE-REGISTERED: R1 -> 6 propagating modes (omega = k). R2 -> 2 TT with omega=k, 3 gauge omega=0, 1 scalar unstable (omega^2<0).
#                 R2' -> scalar has omega^2>0 but negative kinetic energy (ghost). R3 -> exactly 2 modes, omega = k.
import numpy as np

idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def to_mat(v):
    H = np.zeros((3, 3))
    for a, (i, j) in enumerate(idx):
        H[i, j] = H[j, i] = v[a]
    return H


def quad(f):  # matrix of quadratic form f on R^6 (f must be quadratic)
    Q = np.zeros((6, 6)); I = np.eye(6)
    for a in range(6):
        for b in range(6):
            Q[a, b] = 0.5 * (f(I[a] + I[b]) - f(I[a]) - f(I[b]))
    return Q


k = np.array([0.0, 0.0, 1.0]); K2 = k @ k
T_plain = quad(lambda v: 0.25 * np.sum(to_mat(v) ** 2))
T_dewitt = quad(lambda v: 0.25 * (np.sum(to_mat(v) ** 2) - np.trace(to_mat(v)) ** 2))
V_naive = quad(lambda v: 0.25 * K2 * np.sum(to_mat(v) ** 2))


def V_fp(v):
    H = to_mat(v); h = np.trace(H); kh = H @ k
    return 0.25 * K2 * np.sum(H**2) - 0.5 * kh @ kh + 0.5 * (k @ H @ k) * h - 0.25 * K2 * h * h


V_FP = quad(V_fp)

# classification helpers
def classify(v):
    H = to_mat(v); P = np.eye(3) - np.outer(k, k) / K2
    tt = P @ H @ P - 0.5 * P * np.trace(P @ H @ P)
    s = np.linalg.norm(tt); tot = np.linalg.norm(H)
    if s > 0.99 * tot: return "TT (plus/cross)"
    trT = np.trace(P @ H @ P); long = k @ H @ k
    vec = np.linalg.norm(P @ H @ k)
    if vec > 0.99 * tot: return "vector"
    return "scalar"


xi_basis = [np.eye(3)[i] for i in range(3)]
gauge = np.array([[(np.outer(k, xi) + np.outer(xi, k))[i, j] for (i, j) in idx] for xi in xi_basis])
print("gauge invariance of Fierz-Pauli potential: max |V_FP(gauge)| =",
      max(abs(g @ V_FP @ g) for g in gauge), " | naive:", max(abs(g @ V_naive @ g) for g in gauge))


def modes(T, V, name):
    print(f"\n{name}")
    w, U = np.linalg.eig(np.linalg.solve(T, V))
    order = np.argsort(w.real)
    for i in order:
        u = U[:, i].real; kin = u @ T @ u
        tag = classify(u)
        status = ("omega^2 < 0: UNSTABLE" if w[i].real < -1e-9 else
                  ("omega = 0 (frozen)" if abs(w[i]) < 1e-9 else f"omega = {np.sqrt(w[i].real):.3f} k"))
        print(f"   {tag:16s} {status:24s} kinetic sign {'+' if kin > 0 else '-'}")


modes(T_plain, V_naive, "R1 naive: each bond its own wave")
modes(T_plain, V_FP, "R2 background independent (curvature-only energy), positive ES kinetic")
modes(T_dewitt, V_FP, "R2' same potential, DeWitt kinetic (as in GR)")

# R3: constraint surface / gauge
c = np.array([(np.outer(k, k) - K2 * np.eye(3))[i, j] * (1 if i == j else 2) for (i, j) in idx])  # C(h) = c . v
print("\ngauge directions satisfy the constraint:", np.allclose(gauge @ c, 0))
# basis of constraint surface, then quotient by gauge using T-orthogonal complement
N = np.linalg.svd(c[None, :])[2][1:]                    # 5 vectors with c.v = 0
G = gauge
# T-orthogonal complement of gauge inside constraint surface
A = N @ T_plain @ G.T                                    # 5x3
null = np.linalg.svd(A.T)[2][3:]                          # 2 combos
B = null @ N                                              # physical basis (2 x 6)
Tp, Vp = B @ T_plain @ B.T, B @ V_FP @ B.T
w = np.linalg.eigvals(np.linalg.solve(Tp, Vp))
print("R3 = R2 + budget constraint: physical modes =", len(w), "; omega/k =", np.round(np.sqrt(np.abs(w)), 4),
      "; types:", [classify(b) for b in B])
