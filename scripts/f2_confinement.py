# F2: confinement. Agents with an internal N-dim complex state and a free local frame -> bond variables U in the
# gauge group (U(1) for N=1, SU(2) for N=2). Energy = frame frustration around plaquette loops:
#   S = beta * sum_P (1 - Re Tr U_P / N)
# Sampled with population/ES-style local Metropolis updates (= sampling exp(-S), stochastic quantization, Sec.2.3).
# Observable: Wilson loops W(R,T); Creutz ratio chi(R) = -ln[ W(R,R) W(R-1,R-1) / W(R,R-1)^2 ] ~ string tension.
# PRE-REGISTERED:
#   SU(2) beta=2.3 : chi(2), chi(3) roughly equal and > 0.1           (area law: CONFINEMENT)
#   U(1)  beta=1.5 : chi small (< 0.05) and decreasing with R          (Coulomb phase, no confinement)
#   U(1)  beta=0.8 : chi large (strong-coupling lattice confinement, known lattice artefact)
#   SU(2) chi(2) decreases with beta (2.2 -> 2.4 -> 2.6)                (asymptotic freedom)
import numpy as np, sys, time

rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
import os
L, D = int(os.environ.get('LAT', 8)), 4
RMAX = int(os.environ.get('RMAX', 3))


class U1:
    N = 1
    @staticmethod
    def one(shape): return np.zeros(shape)
    @staticmethod
    def mul(a, b): return a + b
    @staticmethod
    def dag(a): return -a
    @staticmethod
    def retr(a): return np.cos(a)
    @staticmethod
    def rand_near(shape, eps): return rng.uniform(-eps, eps, shape)
    zero_add = None


class SU2:
    N = 2
    @staticmethod
    def one(shape):
        q = np.zeros(shape + (4,)); q[..., 0] = 1; return q
    @staticmethod
    def mul(a, b):
        a0, av = a[..., :1], a[..., 1:]; b0, bv = b[..., :1], b[..., 1:]
        r0 = a0 * b0 - np.sum(av * bv, -1, keepdims=True)
        rv = a0 * bv + b0 * av - np.cross(av, bv)
        return np.concatenate([r0, rv], -1)
    @staticmethod
    def dag(a): return np.concatenate([a[..., :1], -a[..., 1:]], -1)
    @staticmethod
    def retr(a): return 2 * a[..., 0]
    @staticmethod
    def rand_near(shape, eps):
        v = rng.normal(size=shape + (3,)) * eps
        q = np.concatenate([np.ones(shape + (1,)), v], -1)
        return q / np.linalg.norm(q, axis=-1, keepdims=True)


def shift(a, mu, s):  # a(x + s*mu)
    return np.roll(a, -s, axis=mu)


def staple(G, U, mu):
    tot = None
    for nu in range(D):
        if nu == mu: continue
        up = G.mul(G.mul(shift(U[nu], mu, 1), G.dag(shift(U[mu], nu, 1))), G.dag(U[nu]))
        dn_nu = shift(U[nu], nu, -1)
        dn = G.mul(G.mul(G.dag(shift(dn_nu, mu, 1)), G.dag(shift(U[mu], nu, -1))), dn_nu)
        s = up + dn if G is SU2 else None
        if G is U1:
            s = np.exp(1j * up) + np.exp(1j * dn)
        tot = s if tot is None else tot + s
    return tot


def local_action(G, Umu, st, beta):
    if G is U1:
        return -beta * np.real(np.exp(1j * Umu) * st)
    # Re Tr(U * staple_sum) for quaternion sums: staple sum is a (non-unit) quaternion; Re Tr = 2*(u0 s0 - u.s)
    return -beta / 2 * 2 * (Umu[..., 0] * st[..., 0] - np.sum(Umu[..., 1:] * st[..., 1:], -1))


coords = np.indices((L,) * D).sum(0)
parity = [(coords % 2 == 0), (coords % 2 == 1)]


def sweep(G, U, beta, eps, hits=3):
    acc = 0; tot = 0
    for mu in range(D):
        for par in parity:
            st = staple(G, U, mu)
            for _ in range(hits):
                old = U[mu]
                new = G.mul(G.rand_near(old.shape[:D], eps), old)
                dS = local_action(G, new, st, beta) - local_action(G, old, st, beta)
                ok = par & (rng.random(dS.shape) < np.exp(-dS))
                if G is SU2:
                    U[mu] = np.where(ok[..., None], new, old)
                else:
                    U[mu] = np.where(ok, new, old)
                acc += ok.sum(); tot += par.sum()
    return acc / tot


def line(G, U, mu, n):
    r = U[mu].copy()
    for s in range(1, n):
        r = G.mul(r, shift(U[mu], mu, s))
    return r


def wilson(G, U, R, T):
    vals = []
    for mu in range(D):
        for nu in range(mu + 1, D):
            for (a, b, n1, n2) in [(mu, nu, R, T), (nu, mu, R, T)]:
                A = line(G, U, a, n1); B = shift(line(G, U, b, n2), a, n1)
                C = shift(line(G, U, a, n1), b, n2); Dl = line(G, U, b, n2)
                loop = G.mul(G.mul(A, B), G.dag(G.mul(Dl, C)))
                vals.append(np.mean(G.retr(loop)) / G.N)
    return np.mean(vals)


def run(group, beta, ntherm=150, nmeas=150):
    G = SU2 if group == "SU2" else U1
    U = [G.one((L,) * D) for _ in range(D)]
    eps = 0.5 if G is SU2 else 1.0
    for i in range(ntherm):
        a = sweep(G, U, beta, eps)
        eps *= 1.05 if a > 0.55 else 0.95
    Ws = {(r, t): [] for r in range(1, RMAX + 1) for t in range(1, RMAX + 1)}
    for i in range(nmeas):
        sweep(G, U, beta, eps)
        if i % 3 == 0:
            for key in Ws: Ws[key].append(wilson(G, U, *key))
    W = {k: np.mean(v) for k, v in Ws.items()}
    chi = {R: -np.log(W[(R, R)] * W[(R - 1, R - 1)] / W[(R, R - 1)] ** 2) for R in range(2, RMAX + 1)}
    return W, chi


if __name__ == "__main__":
    group, beta = sys.argv[1], float(sys.argv[2])
    t0 = time.time()
    W, chi = run(group, beta)
    print(f"{group} beta={beta} L={L} seed={sys.argv[3] if len(sys.argv)>3 else 0}: plaq={W[(1,1)]:.4f} chi=" + " ".join(f"{chi[R]:.4f}" for R in sorted(chi)) + f"  ({time.time()-t0:.0f}s)", flush=True)
