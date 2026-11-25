# K8: why only pole masses? QED one-loop: MSbar mass  m_i(mu) = M_i [1 - (alpha/pi)(1 + (3/2) ln(mu/M_i))]
# The mu-dependence is common to all three leptons -> rescales all masses equally -> Q unchanged.
# The flavour-dependent part (3 alpha/2pi) ln M_i makes running masses ~ M_i^(1+eps), eps = 3 alpha/(2 pi).
# PRE-REGISTERED: Q_running - Q_pole ~ +1.3e-3, independent of mu (one loop, fixed alpha);
#                 consistent with the remembered M_Z value 0.667929.
import numpy as np

alpha = 1 / 137.035999
M = np.array([0.51099895, 105.6583755, 1776.86])


def Q(m): return m.sum() / np.sqrt(m).sum() ** 2


def msbar(mu, a=alpha):
    return M * (1 - (a / np.pi) * (1 + 1.5 * np.log(mu / M)))


print(f"Q(pole) = {Q(M):.7f}")
for mu in [1.0, 10.0, 1000.0, 91187.6]:
    print(f"mu = {mu:9.1f} MeV: Q(MSbar, one loop) = {Q(msbar(mu)):.7f}, Delta Q = {Q(msbar(mu)) - Q(M):+.3e}")
eps = 3 * alpha / (2 * np.pi)
print(f"power-law form M^(1+eps), eps = {eps:.5f}: Q = {Q(M**(1+eps)) * 1:.7f}, Delta Q = {Q(M**(1+eps)/M.sum()**eps) - Q(M):+.3e}")
dQ = (Q(M**(1 + 1e-6)) - Q(M)) / 1e-6
print(f"dQ/deps = {dQ:.4f} -> Delta Q = eps * dQ/deps = {eps * dQ:+.3e}")
print(f"remembered M_Z running masses give Delta Q = {0.667929 - Q(M):+.3e}")
print(f"ratio |QED shift| / |pole deviation from 2/3| = {abs(eps*dQ) / abs(Q(M) - 2/3):.0f}")
# if Koide held exactly for running masses, what pole Q would follow?
print(f"if Q_running = 2/3 exactly, then Q_pole = {2/3 - eps*dQ:.6f}  (observed {Q(M):.6f}, exp. uncertainty from m_tau ~ 1e-5)")
