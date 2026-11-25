# C7 (H3): Dirac's large number hypothesis is a statement about DIMENSIONLESS numbers, so no frame can rescue it.
# N1 = e^2/(4 pi eps0 G m_p m_e) (electric/gravitational force), N2 = c t0 / r_e (age in atomic light-crossing units).
# LNH: N1 ∝ N2 ∝ t  => with atomic constants fixed, G ∝ 1/t => Gdot/G = -1/t0.
# PRE-REGISTERED: N1 ~ 2e39, N2 ~ 5e40; LNH needs |Gdot/G| ~ 7e-11 /yr; LLR bound ~1e-13 /yr -> excluded by >100 sigma.
import numpy as np
e, eps0, G, mp, me, c = 1.602176634e-19, 8.8541878128e-12, 6.67430e-11, 1.67262192e-27, 9.1093837e-31, 2.99792458e8
re = 2.8179403262e-15
t0 = 13.79e9 * 3.15576e7
N1 = e**2 / (4*np.pi*eps0*G*mp*me)
N2 = c*t0/re
print(f"N1 = {N1:.3e}, N2 = {N2:.3e}, N2/N1 = {N2/N1:.1f}")
need = -1/13.79e9
llr, llr_err = 7.1e-14, 7.6e-14     # Lunar laser ranging (Hofmann & Mueller 2018), value from memory
print(f"LNH: Gdot/G = {need:.2e} /yr ; LLR: ({llr:.1e} ± {llr_err:.1e}) /yr ; tension = {(llr-need)/llr_err:.0f} sigma")
# what is still allowed: N1 ∝ t^k with k
print(f"allowed |k| (2 sigma): {2*llr_err*13.79e9:.4f}")
