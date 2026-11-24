# D9: strong-field prediction. Critical impact parameter (shadow radius) b_c = min_r  r * sqrt(B/A)
#   network (multiplicative law): A = exp(-2M/r), B = exp(2M/r)   -> b = r exp(2M/r)
#   Schwarzschild (isotropic):    A = ((1-M/2r)/(1+M/2r))^2, B = (1+M/2r)^4
import numpy as np
from scipy.optimize import minimize_scalar
M = 1.0
net = minimize_scalar(lambda r: r*np.exp(2*M/r), bounds=(0.6, 20), method="bounded")
sch = minimize_scalar(lambda r: r*(1+M/(2*r))**3/(1-M/(2*r)), bounds=(0.51, 20), method="bounded")
print(f"network  : photon sphere at isotropic r = {net.x:.4f} M, shadow radius b_c = {net.fun:.4f} M  (analytic 2e = {2*np.e:.4f})")
print(f"Schwarzschild: photon sphere at isotropic r = {sch.x:.4f} M, b_c = {sch.fun:.4f} M  (analytic 3*sqrt3 = {3*np.sqrt(3):.4f})")
print(f"shadow size ratio network/GR = {net.fun/sch.fun:.4f}  (+{100*(net.fun/sch.fun-1):.1f} %)")
print(f"horizon: network nu=exp(-M/r) > 0 for all r -> none ; Schwarzschild horizon at isotropic r = M/2")
