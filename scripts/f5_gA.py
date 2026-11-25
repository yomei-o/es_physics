# F5 production: quenched SU(3), beta=5.7, 8^3x16, Wilson kappa=0.164, Feynman-Hellmann for g_A and g_V.
# usage: python f5_gA.py SEED NCONF   -> writes f5_out_SEED.npz
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, qcd_lib as Q

seed, nconf = int(sys.argv[1]), int(sys.argv[2])
L, T = int(os.environ.get('LS', 8)), int(os.environ.get('LT', 16))
beta, kappa, lam = 5.7, 0.164, 0.005
NTHERM, NSEP = int(os.environ.get('NTHERM', 200)), int(os.environ.get('NSEP', 20))
rng = np.random.default_rng(seed)
GA = Q.G[2] @ Q.G5            # gamma_3 gamma_5
GV = Q.G[3]                   # gamma_4
Punpol = (np.eye(4) + Q.G[3]) / 2
Ppol = Punpol @ (1j * Q.G5 @ Q.G[2])    # spin projection along z

U = [np.broadcast_to(np.eye(3, dtype=complex), (L, L, L, T, 3, 3)).copy() for _ in range(4)]
eps = 0.25; t0 = time.time()
for i in range(NTHERM):
    a = Q.metropolis_sweep(U, beta, eps, rng); eps *= 1.05 if a > 0.5 else 0.95
    if i % 10 == 9: U = Q.reunitarize(U)
print(f"seed {seed}: thermalised in {time.time()-t0:.0f}s, plaquette {Q.plaquette(U):.4f}", flush=True)

res = {k: [] for k in ["plaq", "pion", "CApol_p", "CApol_m", "CAun_p", "CAun_m", "CVun_p", "CVun_m", "resid"]}
for n in range(nconf):
    if n > 0:
        for i in range(NSEP):
            a = Q.metropolis_sweep(U, beta, eps, rng); eps *= 1.05 if a > 0.5 else 0.95
            if i % 10 == 9: U = Q.reunitarize(U)
    t1 = time.time()
    SpA, r1 = Q.propagator(U, kappa, +lam * GA, tol=1e-8)
    SmA, r2 = Q.propagator(U, kappa, -lam * GA, tol=1e-8, guess=SpA)
    SpV, r3 = Q.propagator(U, kappa, +lam * GV, tol=1e-8, guess=SpA)
    SmV, r4 = Q.propagator(U, kappa, -lam * GV, tol=1e-8, guess=SpV)
    res["plaq"].append(Q.plaquette(U))
    res["pion"].append(Q.pion(0.5 * (SpV + SmV)))
    res["CApol_p"].append(Q.nucleon_full(SpA, SmA, Ppol)); res["CApol_m"].append(Q.nucleon_full(SmA, SpA, Ppol))
    res["CAun_p"].append(Q.nucleon_full(SpA, SmA, Punpol)); res["CAun_m"].append(Q.nucleon_full(SmA, SpA, Punpol))
    res["CVun_p"].append(Q.nucleon_full(SpV, SmV, Punpol)); res["CVun_m"].append(Q.nucleon_full(SmV, SpV, Punpol))
    res["resid"].append(max(r1, r2, r3, r4))
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"f5_out_{seed}.npz"), **{k: np.array(v) for k, v in res.items()})
    print(f"seed {seed}: config {n+1}/{nconf} done in {time.time()-t1:.0f}s, plaq {res['plaq'][-1]:.4f}, resid {res['resid'][-1]:.1e}", flush=True)
