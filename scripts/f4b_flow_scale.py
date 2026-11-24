# F4b: BP (gradient flow) defines the physical scale of the SU(2) gauge network.
# Flow: V' = -g0^2 (dS) V ; for SU(2) quaternions with q = U * staple_sum:  V <- exp(-i dt qv.sigma) V
# Energy density E = 2 * sum_{plaquettes per site} ReTr(1 - U_p)  (plaquette definition), t0: t^2 <E> = REF;  PRIMARY REF = 0.2 (fixed before seeing data), 0.1 and 0.3 secondary
# PRE-REGISTERED: sqrt(8 t0)/a grows with beta; ratio a(2.2)/a(2.4) and a(2.3)/a(2.4) agree with 2-loop
# asymptotic scaling within 10-20 %.
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f2_confinement as F
from f2_confinement import SU2, staple, sweep, D, L

beta = float(sys.argv[1]); seed = int(sys.argv[2])
F.rng = np.random.default_rng(seed)


def plaq_energy(U):
    tot = 0.0
    for mu in range(D):
        for nu in range(mu + 1, D):
            P = SU2.mul(SU2.mul(U[mu], F.shift(U[nu], mu, 1)), SU2.dag(SU2.mul(U[nu], F.shift(U[mu], nu, 1))))
            tot += np.mean(SU2.retr(P))
    nplaq = D * (D - 1) // 2
    return 2 * (2 * nplaq - tot)            # 2 * sum_p ReTr(1 - U_p), per site


def flow_step(U, dt):
    for mu in range(D):                    # (sequential Euler per direction)
        st = staple(SU2, U, mu)
        q = SU2.mul(U[mu], st)
        v = q[..., 1:]
        ang = np.linalg.norm(v, axis=-1, keepdims=True) * dt      # g0^2 dS = 2 qv (oriented plaquettes)
        n = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)
        R = np.concatenate([np.cos(ang), -np.sin(ang) * n], -1)
        U[mu] = SU2.mul(R, U[mu])
        U[mu] /= np.linalg.norm(U[mu], axis=-1, keepdims=True)


U = [SU2.one((L,) * D) for _ in range(D)]
eps = 0.5
for i in range(200):
    a = sweep(SU2, U, beta, eps); eps *= 1.05 if a > 0.55 else 0.95
REFS = [0.1, 0.2, 0.3]
t0s = {r: [] for r in REFS}
for c in range(4):
    for i in range(25): sweep(SU2, U, beta, eps)
    V = [u.copy() for u in U]
    t, dt, prev = 0.0, 0.01, (0.0, 0.0)
    todo = list(REFS)
    while t < 5 and todo:
        flow_step(V, dt); t += dt
        y = t * t * plaq_energy(V)
        while todo and y >= todo[0]:
            r = todo.pop(0); t0s[r].append(prev[0] + (r - prev[1]) * (t - prev[0]) / (y - prev[1]))
        prev = (t, y)
out = " | ".join(f"ref {r}: t0/a^2 = {np.mean(v):.4f} +- {np.std(v)/np.sqrt(max(len(v),1)):.4f} (n={len(v)})" for r, v in t0s.items())
print(f"beta={beta} L={L} seed={seed}: {out}", flush=True)
