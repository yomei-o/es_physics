# F1b: count phase frustration on SPACETIME plaquettes (4D, Euclidean). PRE-REGISTERED: quadratic form isotropic in 4D:
# eigenvalues {0 (gauge), |k|^2 x3} for every 4D direction incl. time/space mixtures -> after Wick rotation c_photon = c
# (the same c as gravitational waves, sister notes Sec.12).
import numpy as np
for kd in [(0,0,0,1), (1,0,0,0), (1,0,0,1), (1,2,-1,3)]:
    for kk in [0.05, 0.5]:
        k = np.array(kd, float) / np.linalg.norm(kd) * kk
        d = np.exp(1j * k) - 1
        V = np.zeros((4, 4), complex)
        for mu in range(4):
            for nu in range(mu + 1, 4):
                v = np.zeros(4, complex); v[nu] = d[mu]; v[mu] = -d[nu]; V += np.outer(v.conj(), v)
        w = np.linalg.eigvalsh(0.5 * (V + V.conj().T))
        print(f"k dir {str(kd):14s} |k|={kk}: eigenvalues/|k|^2 = {np.round(w / kk**2, 4)}")
