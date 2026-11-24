# F1: local freedom of each agent's phase convention (its own J) -> bond "connection" A_ij -> photon.
# psi_i -> e^{i theta_i} psi_i ,  A_ij -> A_ij + theta_i - theta_j  leaves sum_ij psi_i^* e^{iA_ij} psi_j invariant.
# Gauge-invariant energy = phase frustration around plaquette loops (magnetic flux) F = sum of A around the loop.
# Linearised on a 3D cubic lattice of agents: A_mu(x) (3 per site); V = 1/2 sum_plaq F^2 ; T = 1/2 sum Adot^2.
# PRE-REGISTERED: per k: 2 transverse modes with omega = |k_lat| (lattice: k_lat_mu = 2 sin(k_mu/2)),
#                 1 gauge (longitudinal) zero mode; with Gauss constraint (no global phase reference: invariance
#                 under time-dependent theta) the longitudinal mode is removed -> exactly 2 photon polarisations,
#                 isotropic speed 1 at small k.
import numpy as np


def modes(k):
    # lattice derivative: (D_mu f)(k) = (e^{ik_mu}-1) f ; curl on links -> plaquettes
    d = np.exp(1j * k) - 1
    # V = 1/2 |F_{mu nu}|^2 summed over mu<nu,  F_{mu nu} = d_mu A_nu - d_nu A_mu
    V = np.zeros((3, 3), complex)
    for mu in range(3):
        for nu in range(3):
            if mu == nu: continue
            # contribution of plaquette (mu,nu) (counted for mu<nu) : |d_mu A_nu - d_nu A_mu|^2
            pass
    for mu in range(3):
        for nu in range(mu + 1, 3):
            vec = np.zeros(3, complex); vec[nu] = d[mu]; vec[mu] = -d[nu]
            V += np.outer(vec.conj(), vec)
    V = 0.5 * (V + V.conj().T)
    w, U = np.linalg.eigh(V)          # T = identity -> omega^2 = eigenvalues of V
    return w, U, d


print(" k direction        |k|    omega/|k| of nonzero modes     zero modes   zero-mode is longitudinal (gauge)?")
for kd in [(0, 0, 1), (1, 1, 0), (1, 2, 3)]:
    for kk in [0.05, 0.5, 1.5]:
        k = np.array(kd, float) / np.linalg.norm(kd) * kk
        w, U, d = modes(k)
        nz = w > 1e-10
        gauge_dir = d / np.linalg.norm(d)              # A = D theta  (pure gauge)
        zero_vec = U[:, ~nz][:, 0]
        overlap = abs(np.vdot(gauge_dir, zero_vec))
        print(f" {str(kd):12s} {kk:6.2f}   {np.round(np.sqrt(w[nz]) / kk, 4)}            {np.sum(~nz)}           overlap {overlap:.6f}")

print("\nGauss constraint (sum_mu conj(D_mu) Adot_mu = 0) removes the gauge/longitudinal direction:")
k = np.array([1, 2, 3.0]) / np.sqrt(14) * 0.05
w, U, d = modes(k)
g = d / np.linalg.norm(d)
P = np.eye(3) - np.outer(g, g.conj())                  # physical subspace
Vp = P @ (U @ np.diag(w) @ U.conj().T) @ P
wp = np.linalg.eigvalsh(Vp)
print(f"  physical modes: {np.sum(wp > 1e-12)}, omega/|k| = {np.round(np.sqrt(wp[wp > 1e-12]) / 0.05, 5)}")
