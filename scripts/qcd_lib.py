# Minimal quenched lattice QCD in numpy: SU(3) Metropolis, Wilson-Dirac operator, BiCGStab, nucleon 2-pt functions.
# Axes: field[x, y, z, t, ...]; gauge U[mu] shape (L, L, L, T, 3, 3); fermion psi shape (L, L, L, T, 4, 3).
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab

# ---------------- gamma matrices (Euclidean, chiral basis) ----------------
s1 = np.array([[0, 1], [1, 0]], complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2); Z2 = np.zeros((2, 2))
G = [np.block([[Z2, -1j * s], [1j * s, Z2]]) for s in (s1, s2, s3)] + [np.block([[Z2, I2], [I2, Z2]])]
G5 = G[0] @ G[1] @ G[2] @ G[3]
# charge conjugation: C gamma_mu C^-1 = -gamma_mu^T
C = None
for cand in [G[1] @ G[3], G[3] @ G[1], G[0] @ G[2], G[2] @ G[0], G[1], G[3] @ G[1] @ G[4 - 1]]:
    Ci = np.linalg.inv(cand)
    if all(np.allclose(cand @ g @ Ci, -g.T) for g in G):
        C = cand; break
assert C is not None
CG5 = C @ G5


def check_gammas():
    for m in range(4):
        for n in range(4):
            assert np.allclose(G[m] @ G[n] + G[n] @ G[m], 2 * np.eye(4) * (m == n))
    assert np.allclose(G5 @ G5, np.eye(4))
    return True


# ---------------- SU(3) gauge field ----------------
def su3_project(M):
    Q, R = np.linalg.qr(M)
    d = np.diagonal(R, axis1=-2, axis2=-1)
    Q = Q * (d / np.abs(d))[..., None, :]
    det = np.linalg.det(Q)
    return Q / (det ** (1 / 3))[..., None, None]


def random_near_identity(shape, eps, rng):
    H = rng.normal(size=shape + (3, 3)) + 1j * rng.normal(size=shape + (3, 3))
    H = 0.5 * (H + np.conj(np.swapaxes(H, -1, -2)))
    H -= np.trace(H, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return su3_project(np.eye(3) + 1j * eps * H)


def mm(a, b): return np.einsum('...ij,...jk->...ik', a, b)
def dag(a): return np.conj(np.swapaxes(a, -1, -2))
def sh(a, mu, s): return np.roll(a, -s, axis=mu)


def staple(U, mu):
    S = 0
    for nu in range(4):
        if nu == mu: continue
        S = S + mm(mm(sh(U[nu], mu, 1), dag(sh(U[mu], nu, 1))), dag(U[nu]))
        Ub = sh(U[nu], nu, -1)
        S = S + mm(mm(dag(sh(Ub, mu, 1)), dag(sh(U[mu], nu, -1))), Ub)
    return S


def plaquette(U):
    tot = 0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            P = mm(mm(U[mu], sh(U[nu], mu, 1)), dag(mm(U[nu], sh(U[mu], nu, 1))))
            tot += np.mean(np.real(np.trace(P, axis1=-2, axis2=-1))) / 3
    return tot / 6


def metropolis_sweep(U, beta, eps, rng, hits=4):
    shape = U[0].shape[:4]
    par = np.indices(shape).sum(0) % 2
    acc = 0
    for mu in range(4):
        for p in (0, 1):
            St = staple(U, mu)
            mask = (par == p)
            for _ in range(hits):
                X = random_near_identity(shape, eps, rng)
                new = mm(X, U[mu])
                dS = -(beta / 3) * np.real(np.trace(mm(new - U[mu], St), axis1=-2, axis2=-1))
                ok = mask & (rng.random(shape) < np.exp(-np.minimum(dS, 50)))
                U[mu] = np.where(ok[..., None, None], new, U[mu])
                acc += ok.sum() / mask.sum()
    return acc / (8 * hits)


def reunitarize(U):
    return [su3_project(u) for u in U]


# ---------------- Wilson-Dirac operator ----------------
P_minus = [np.eye(4) - g for g in G]
P_plus = [np.eye(4) + g for g in G]


def shift_ferm(psi, mu, s):
    out = np.roll(psi, -s, axis=mu)
    if mu == 3:  # antiperiodic in time
        T = psi.shape[3]
        if s == 1: out[:, :, :, T - 1] *= -1
        else: out[:, :, :, 0] *= -1
    return out


PmT = [p.T.copy() for p in P_minus]
PpT = [p.T.copy() for p in P_plus]


def dslash(U, psi, kappa, extra=None):
    """D psi = psi - kappa sum_mu [(1-g_mu) U_mu(x) psi(x+mu) + (1+g_mu) U_mu(x-mu)^dag psi(x-mu)] + extra psi
    psi[..., s, a]: colour multiply = psi @ U^T, spin multiply = P @ psi (done as matmul for speed)"""
    out = psi.copy()
    for mu in range(4):
        Ut = np.swapaxes(U[mu], -1, -2)
        fwd = np.matmul(shift_ferm(psi, mu, 1), Ut)                       # U(x) psi(x+mu)
        bwd = shift_ferm(np.matmul(psi, np.conj(U[mu])), mu, -1)          # U(x-mu)^dag psi(x-mu)
        out -= kappa * (np.matmul(P_minus[mu], fwd) + np.matmul(P_plus[mu], bwd))
    if extra is not None:
        out += np.matmul(extra, psi)
    return out


def dslash_dag(U, psi, kappa, extra=None):
    """D^dag = g5 D g5 for the Wilson part; the extra (site-local) term is daggered explicitly"""
    g5 = lambda v: np.einsum('st,...tb->...sb', G5, v)
    out = g5(dslash(U, g5(psi), kappa))
    if extra is not None:
        out += np.einsum('st,...tb->...sb', extra.conj().T, psi)
    return out


def solve(U, kappa, src, extra=None, tol=1e-10, x0=None):
    """CGNR: solve D^dag D x = D^dag b with conjugate gradients (always converges for invertible D)"""
    from scipy.sparse.linalg import cg
    shape = src.shape; n = src.size
    A = LinearOperator((n, n), matvec=lambda v: dslash_dag(U, dslash(U, v.reshape(shape), kappa, extra), kappa, extra).ravel(), dtype=complex)
    b = dslash_dag(U, src, kappa, extra).ravel()
    x, info = cg(A, b, x0=None if x0 is None else x0.ravel(), rtol=tol, maxiter=20000)
    res = np.linalg.norm(dslash(U, x.reshape(shape), kappa, extra) - src) / np.linalg.norm(src)
    return x.reshape(shape), info, res


def propagator(U, kappa, extra=None, tol=1e-10, guess=None):
    """point source at origin; returns S[x,y,z,t, s_sink, c_sink, s_src, c_src]"""
    L1, L2, L3, T = U[0].shape[:4]
    S = np.zeros((L1, L2, L3, T, 4, 3, 4, 3), complex)
    worst = 0
    for s in range(4):
        for c in range(3):
            src = np.zeros((L1, L2, L3, T, 4, 3), complex); src[0, 0, 0, 0, s, c] = 1
            x, info, res = solve(U, kappa, src, extra, tol, None if guess is None else guess[..., s, c])
            worst = max(worst, res)
            S[..., s, c] = x
    return S, worst


# ---------------- correlators ----------------
EPS = [((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1), ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)]


def pion(S):
    # C(t) = sum_x |S|^2 (gamma5-hermiticity)
    return (np.abs(S) ** 2).sum(axis=(0, 1, 2, 4, 5, 6, 7))


def nucleon(Su, Sd, Gproj):
    """proton P = eps_abc (u_a^T C g5 d_b) u_c; source with Cbar = g4 (C g5)^dag g4. Returns C(t) = sum_x Tr[Gproj C_{aa'}]"""
    M = CG5; Mb = G[3] @ CG5.conj().T @ G[3]
    Su = Su.sum(axis=(0, 1, 2)); Sd = Sd.sum(axis=(0, 1, 2))  # NOTE: zero-momentum projection must be applied to the
    raise NotImplementedError  # product, not to each propagator -- implemented in nucleon_full below


def nucleon_full(Su, Sd, Gproj):
    M = CG5; Mb = G[3] @ CG5.conj().T @ G[3]
    T = Su.shape[3]
    out = np.zeros(T, complex)
    # S[..., s, c, s', c'] ; indices: sink (alpha,beta,gamma), source (alpha',beta',gamma')
    for (a, b, c), e1 in EPS:
        for (a2, b2, c2), e2 in EPS:
            sgn = e1 * e2
            Sd_bb = Sd[..., :, b, :, b2]          # [x..., gamma, gamma']
            Su_cc = Su[..., :, c, :, c2]          # [alpha, alpha']
            Su_aa = Su[..., :, a, :, a2]          # [beta, beta']
            Su_ca = Su[..., :, c, :, a2]          # [alpha, beta']
            Su_ac = Su[..., :, a, :, c2]          # [beta, alpha']
            # direct: Gproj_{a' a} Su_cc[a,a'] * M_{b g} Mb_{g' b'} Sd[g,g'] Su_aa[b,b']
            diq = np.einsum('bg,...gh,hk,...bk->...', M, Sd_bb, Mb, Su_aa)
            direct = np.einsum('...', np.einsum('pa,...ap->...', Gproj, Su_cc) * diq) if False else \
                np.einsum('pa,...ap->...', Gproj, Su_cc) * diq
            # exchange: - Gproj_{a' a} M_{b g} Mb_{g' b'} Sd[g,g'] Su_ca[a,b'] Su_ac[b,a']
            exch = np.einsum('pa,bg,...gh,hk,...ak,...bp->...', Gproj, M, Sd_bb, Mb, Su_ca, Su_ac)
            out += sgn * (direct - exch).sum(axis=(0, 1, 2))
    return out
