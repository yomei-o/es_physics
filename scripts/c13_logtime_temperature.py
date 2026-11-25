# C13 (H12): the Minkowski vacuum seen with the c*t = const clock (log time s = ln t) is thermal.
# 1+1 massless field, observer at x = 0.  Wightman function of d_t phi:  W(t1,t2) = -1/(4 pi (t1 - t2 - i eps)^2).
# Weight-1 operator in the s-frame: O(s) = t d_t phi = e^s d_t phi  ->  W_s = -1/(16 pi sinh^2((s1 - s2 - i eps)/2)),  stationary in s.
# Detector response  F(omega) = ∫ du e^{-i omega u} W(u)   (omega > 0: excitation).
# PRE-REGISTERED: s-clock detector:  F(omega)/F(-omega) = exp(-2 pi omega)  (KMS at T_s = 1/(2 pi), i.e. T_phys = hbar/(2 pi k t)).
#                 t-clock (inertial proper time) detector: F(omega > 0) = 0  (no excitation).
import numpy as np

eps = 0.02
u = np.linspace(-400, 400, 2_000_001)
du = u[1] - u[0]
Ws = -1 / (16 * np.pi * np.sinh((u - 1j * eps) / 2) ** 2)
Wt = -1 / (4 * np.pi * (u - 1j * eps) ** 2)


def F(W, om):
    return (np.sum(np.exp(-1j * om * u) * W) * du).real


print(" omega   F_s(+w)     F_s(-w)     ratio      exp(-2 pi w) |  F_t(+w)      F_t(-w)")
for om in [0.1, 0.25, 0.5, 0.75, 1.0]:
    fp, fm = F(Ws, om), F(Ws, -om)
    tp, tm = F(Wt, om), F(Wt, -om)
    print(f" {om:5.2f}  {fp:+.4e}  {fm:+.4e}  {fp/fm:.5f}   {np.exp(-2*np.pi*om):.5f}   | {tp:+.2e}  {tm:+.4e}")

hbar, kB = 1.054571817e-34, 1.380649e-23
t0 = 13.79e9 * 3.15576e7
H0 = 67.4 / 3.0857e19
print(f"\nT = hbar/(2 pi kB t0) = {hbar/(2*np.pi*kB*t0):.2e} K ;  Gibbons-Hawking H0 hbar/(2 pi kB) = {hbar*H0/(2*np.pi*kB):.2e} K ; ratio = H0 t0 = {H0*t0:.4f}")
