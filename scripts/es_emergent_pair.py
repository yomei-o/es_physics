# Can a single "i" (one rotating plane = two competing collective populations) EMERGE
# from many ES agents with random interactions, without putting q.H.p in by hand?
#
# N agents, agent i controls x_i and minimises its OWN loss
#     L_i(x) = -x_i * sum_{j!=i} W_ij x_j - 1/2 W_ii x_i^2      (so dL_i/dx_i = -(W x)_i)
# W = (1-c) S + c A,  S symmetric (shared/cooperative part), A antisymmetric (zero-sum part).
# Total resource conserved: |x| = 1 after each step.  Each agent only uses ES on its own coordinate.
import numpy as np

rng = np.random.default_rng(2)


def random_W(N, c):
    G = rng.standard_normal((N, N)) / np.sqrt(N)
    S, A = (G + G.T) / np.sqrt(2), (G - G.T) / np.sqrt(2)
    return (1 - c) * S + c * A


print("=== A. ensemble: is the long-time survivor a rotation (complex leading eigenvalue)? ===")
print("   P(rotation) = fraction of random games whose eigenvalue with largest real part is complex")
cs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
print("   c    " + "  ".join(f"N={N:<4d}" for N in [10, 50, 200]))
for c in cs:
    row = []
    for N in [10, 50, 200]:
        trials = 2000 if N < 200 else 400
        cnt = 0
        for _ in range(trials):
            ev = np.linalg.eigvals(random_W(N, c))
            lead = ev[np.argmax(ev.real)]
            cnt += abs(lead.imag) > 1e-9
        row.append(cnt / trials)
    print(f"  {c:.1f}   " + "  ".join(f"{r:6.3f}" for r in row))


print("\n=== B. actual ES dynamics (agents only perturb their own coordinate) ===")


def es_run(W, steps=6000, eta=0.02, pop=16, sigma=1e-2):
    N = W.shape[0]
    x = rng.standard_normal(N); x /= np.linalg.norm(x)
    traj = []
    for s in range(steps):
        g = np.zeros(N)
        for i in range(N):                       # each agent: ES on its own loss, own coordinate only
            eps = rng.standard_normal(pop)
            def Li(xi):
                y = x.copy(); y[i] = xi
                return -xi * (W[i] @ y - W[i, i] * xi) - 0.5 * W[i, i] * xi**2
            d = np.array([Li(x[i] + sigma * e) - Li(x[i] - sigma * e) for e in eps])
            g[i] = (d * eps).mean() / (2 * sigma)
        x = x - eta * g
        x /= np.linalg.norm(x)                   # resource conservation
        if s >= steps - 1500:
            traj.append(x.copy())
    return np.array(traj)


N = 30
for c in [0.0, 0.5, 0.9]:
    for trial in range(2):
        W = random_W(N, c)
        ev = np.linalg.eigvals(W)
        lead = ev[np.argmax(ev.real)]
        tr = es_run(W)
        sv = np.linalg.svd(tr - 0 * tr.mean(0), compute_uv=False)
        var = sv**2 / (sv**2).sum()
        # measured rotation frequency in the dominant plane
        U = np.linalg.svd(tr.T, full_matrices=False)[0][:, :2]
        z = tr @ U[:, 0] + 1j * tr @ U[:, 1]
        ang = np.unwrap(np.angle(z))
        omega = abs(np.polyfit(np.arange(len(ang)) * 0.02, ang, 1)[0])
        print(f"c={c:.1f} run{trial}: leading eig = {lead.real:+.3f}{lead.imag:+.3f}i | trajectory variance in "
              f"top-2 dirs = {var[0]:.3f},{var[1]:.3f} (rest {var[2:].sum():.4f}) | rotation freq {omega:.3f} vs |Im lambda| {abs(lead.imag):.3f}")
