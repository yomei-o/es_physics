# D5: light bending in a 2D zero-sum network with a "slow clock" region.
# Fields: u on sites, vx / vy on bonds.  u' = div(J v), v' = J grad u   (antisymmetric coupling: div = -grad^T)
# -> u'' = div(J^2 grad u): waves with local speed c = J(r).  Lattice spacing is fixed (no spatial metric change).
# Pre-registered prediction: rays follow refractive index n = 1/J ("Einstein 1911": only the clock varies),
#   deflection = integral of grad_perp ln n along the ray = HALF of GR for the same Newtonian potential.
#   Newtonian potential defined from slow massive packets (verified in 1D): a = -c^2 grad ln J  -> Phi = ln J.
#   GR would give twice the ray-optics value computed with n = 1/J.
import numpy as np

L = 700
alpha, soft = 4.0, 6.0
X, Y = np.meshgrid(np.arange(L) - L / 2, np.arange(L) - L / 2, indexing='ij')


def Jfun(x, y):
    return 1 - alpha / np.sqrt(x**2 + y**2 + soft**2)


Jx = Jfun(X + 0.5, Y)   # bond (i+1/2, j)
Jy = Jfun(X, Y + 0.5)   # bond (i, j+1/2)


def run(b, flat=False):
    jx = np.ones_like(Jx) if flat else Jx
    jy = np.ones_like(Jy) if flat else Jy
    k, w = 0.6, 14.0
    x0 = -220
    env = np.exp(-((X - x0) ** 2 + (Y - b) ** 2) / (2 * w**2))
    u = env * np.cos(k * (X - x0))
    vx = -np.roll(u, -1, 0) * 0.5 - u * 0.5          # right-moving: v ~ -u (on bond)
    vy = np.zeros_like(u)
    dt, steps = 0.4, 1100
    cents = []
    for s in range(steps):
        # leapfrog (symplectic): v half-kick from grad u, then u from div v
        vx += dt * jx * (np.roll(u, -1, 0) - u)
        vy += dt * jy * (np.roll(u, -1, 1) - u)
        fx, fy = jx * vx, jy * vy
        u += dt * ((fx - np.roll(fx, 1, 0)) + (fy - np.roll(fy, 1, 1)))
        if s in (700, 1100 - 1):
            e = u**2
            cents.append(((X * e).sum() / e.sum(), (Y * e).sum() / e.sum()))
    (x1, y1), (x2, y2) = cents
    return np.arctan2(y2 - y1, x2 - x1)


def ray_prediction(b):
    # small-angle: theta = integral d/dy ln n(x, b) dx  with n = 1/J  (toward the centre is negative y if b>0)
    xs = np.linspace(-L / 2, L / 2, 200001)
    h = 1e-3
    dln = (np.log(1 / Jfun(xs, b + h)) - np.log(1 / Jfun(xs, b - h))) / (2 * h)
    return np.trapezoid(dln, xs)


print(f"J(r) = 1 - {alpha}/sqrt(r^2+{soft}^2)   (slow clock near the centre)")
print(" b    measured deflection   ray optics n=1/J ('Einstein 1911')   GR (2x)   flat-lattice control")
for b in [40, 60, 90]:
    th = run(b) - run(b, flat=True)
    th0 = run(b, flat=True)
    pr = ray_prediction(b)
    print(f"{b:3d}   {th:+.4f} rad            {pr:+.4f}                            {2*pr:+.4f}   {th0:+.5f}")
