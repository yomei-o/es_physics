# C1: when is "c * t = const" true in LambdaCDM?
# Measure light speed with COMOVING rulers (rulers that grow with the universe): c_com = c / a(t).
# Then c_com * t = c t / a. This is constant iff a ∝ t (coasting universe, "R_h = ct").
# d/dt (t/a) = 0  <=>  H t = 1, so in LambdaCDM c_com*t has a stationary point exactly where H t = 1.
# PRE-REGISTERED (written before running):
#   P1: H0 t0 ~ 0.95; H t = 1 happens in the future, t ~ 17 Gyr.
#   P2: c t / a changes by < 1% over the last 2 Gyr.
#   P3: distance modulus LambdaCDM vs coasting (a ∝ t, flat) differ by < 0.15 mag for z < 1.5.
import numpy as np

H0_kms = 67.4                      # Planck 2018
Om, Or = 0.315, 9.1e-5
OL = 1 - Om - Or
H0 = H0_kms / 977.792              # 1/Gyr

lna = np.linspace(np.log(1e-9), np.log(20.0), 400001)
a = np.exp(lna)
H = H0 * np.sqrt(Or / a**4 + Om / a**3 + OL)
dt = 1 / H                         # dt/dlna
t = np.concatenate([[0.5 * a[0]**2 / (H0 * np.sqrt(Or))],   # radiation era start, t = a^2/(2 H0 sqrt(Or))
                    np.zeros(len(a) - 1)])
t[1:] = t[0] + np.cumsum(0.5 * (dt[1:] + dt[:-1]) * np.diff(lna))

i0 = np.argmin(abs(a - 1))
t0 = t[i0]
n = H * t                          # local power-law exponent: a ∝ t^n locally
print(f"t0 = {t0:.3f} Gyr, H0 t0 = {n[i0]:.4f}")

k = np.argmin(abs(n - 1))
print(f"H t = 1 at t = {t[k]:.2f} Gyr, a = {a[k]:.3f} (z = {1/a[k]-1:.3f}), {t[k]-t0:+.2f} Gyr from now")

# deceleration -> acceleration (a'' = 0) for comparison
q = Om / a**3 * 0.5 + Or / a**4 - OL       # sign of -a''  (times H0^2/ (H^2) factor irrelevant)
j = np.where(np.diff(np.sign(q)))[0][0]
print(f"a''=0 (start of acceleration) at t = {t[j]:.2f} Gyr, z = {1/a[j]-1:.3f}")

X = t / a                          # c t / a  in units of c*Gyr (comoving)
Xmax = X[k]
print(f"\nc t / a relative to its maximum:")
for dT in [-10, -5, -2, -1, 0, 1, 2, 5]:
    m = np.argmin(abs(t - (t0 + dT)))
    print(f"  t0{dT:+3d} Gyr (z={1/a[m]-1:6.3f}): c t/a / max = {X[m]/Xmax:.5f}")
m2 = np.argmin(abs(t - (t0 - 2)))
print(f"change of c t / a over last 2 Gyr: {abs(X[i0]/X[m2]-1)*100:.3f} %")
print(f"exponent n = Ht at t0-2 Gyr: {n[m2]:.4f}")

# history of the exponent
print("\nlocal exponent n = d ln a / d ln t  (c_com ∝ t^-n):")
for z in [1e5, 3400, 1000, 100, 10, 3, 1, 0.5, 0]:
    m = np.argmin(abs(a - 1/(1+z)))
    print(f"  z={z:>8g}: t={t[m]:.3e} Gyr, n={n[m]:.4f}")

# P3: distance modulus, LambdaCDM vs flat coasting (a ∝ t => H = H0 (1+z))
c = 299792.458
zs = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
def dL_lcdm(z):
    zz = np.linspace(0, z, 20001)
    E = np.sqrt(Or*(1+zz)**4 + Om*(1+zz)**3 + OL)
    return (1+z) * c / H0_kms * np.trapezoid(1/E, zz)
print("\ndistance modulus difference mu(coasting) - mu(LCDM)  [same H0]:")
diffs = []
for z in zs:
    d1 = dL_lcdm(z)
    d2 = (1+z) * c / H0_kms * np.log(1+z)          # flat, a ∝ t
    d3 = c / H0_kms * z * (1 + z/2)                 # Milne (empty, open)
    diffs.append(5*np.log10(d2/d1))
    print(f"  z={z:3.1f}: coasting {5*np.log10(d2/d1):+.3f} mag, Milne {5*np.log10(d3/d1):+.3f} mag")
# SN analyses fit an absolute magnitude offset, so only the SHAPE matters: remove best constant over z<1.5
sel = zs <= 1.5
off = np.mean(np.array(diffs)[sel])
print(f"after removing a constant offset ({off:+.3f}) over z<=1.5: max |dmu| = {np.max(abs(np.array(diffs)[sel]-off)):.3f} mag")
