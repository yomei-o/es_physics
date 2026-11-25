# C6 (H4): Friedmann equation in network ticks.
# Dictionary (C2): a = 1/nu, d tau = nu dT, c_net = nu^2.  H = (da/dtau)/a = -(dnu/dT)/nu^2.
# Friedmann H^2 = (8 pi G/3) rho, rho ∝ a^{-3(1+w)} = nu^{3(1+w)}
#   =>  (dnu/dT)^2 = nu^4 * nu^{3(1+w)}  (units 8piG/3 = 1)   =>  c_net ∝ T^{-p},  p = 4/(5+3w).
# PRE-REGISTERED: p(w=1/3)=2/3, p(0)=0.8, p(-1/3)=1, p(-1)=2, p(1)=1/2.
#   Robbins-Monro window 1/2 < p <= 1  <=>  -1/3 <= w < 1.
import numpy as np

print(f"{'w':>6} {'p fitted':>9} {'4/(5+3w)':>9}")
for w in [1.0, 1/3, 0.0, -1/3, -0.6, -1.0]:
    e = 7 + 3 * w                       # dnu/dT = -nu^{e/2}
    nu, T = 1.0, 1.0
    Ts, cs = [], []
    # integrate in log T with RK4 on y = ln nu:  dy/dlnT = -T nu^{e/2 - 1}
    f = lambda lnT, y: -np.exp(lnT) * np.exp(y * (e / 2 - 1))
    lnT, y, h = 0.0, 0.0, 1e-3
    while lnT < 30:
        k1 = f(lnT, y); k2 = f(lnT + h/2, y + h/2*k1); k3 = f(lnT + h/2, y + h/2*k2); k4 = f(lnT + h, y + h*k3)
        y += h/6*(k1 + 2*k2 + 2*k3 + k4); lnT += h
        if lnT > 20 and len(Ts) < 2000 and int(lnT / h) % 5 == 0:
            Ts.append(lnT); cs.append(2 * y)       # ln c_net
    pf = -np.polyfit(Ts, cs, 1)[0]
    print(f"{w:6.3f} {pf:9.4f} {4/(5+3*w):9.4f}")

# LambdaCDM today: effective w_eff = -1 + 2/3 (1+q) ... use n = H t = 0.9506 -> p = 2n/(1+n)
n = 0.9506
p = 2 * n / (1 + n)
print(f"\nLCDM today: p = {p:.4f}  ->  effective w = {(4/p - 5)/3:+.4f}")
