# D6: "coincidence rule".  Agents update asynchronously with local probability nu_i per tick.
#   internal oscillation of one agent (mass)        : rate  M_i  = M0 * nu_i
#   interaction between neighbours i,j (propagation): rate  t_ij = t0 * nu_i * nu_j   (both must update)
# compared with the earlier "1911 rule": t_ij = t0 * (nu_i + nu_j)/2.
# PRE-REGISTERED predictions (written before running):
#   (A) slow massive packets: same Newtonian acceleration under both rules,  a = -c^2 grad(ln M) ~ -(2 t0)^2 nu'
#   (B) light deflection: coincidence rule = 2 x (1911 rule)  -> PPN gamma = 1 (GR) instead of 0.
import numpy as np
import sys

which = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---------------- (A) 1D two-component chain (a_i, b_i), real antisymmetric ----------------
if which in ("all", "A"):
    print("=== (A) massive packet at rest in a clock gradient nu(x) = 1 + g x ===")
    N, M0, t0, g = 1200, 0.4, 0.5, 3e-4
    pos = np.arange(N); nu = 1 + g * (pos - N / 2)

    def build(rule):
        A = np.zeros((2 * N, 2 * N)); a = lambda i: 2 * i; b = lambda i: 2 * i + 1
        for i in range(N):
            A[a(i), b(i)] += M0 * nu[i]; A[b(i), a(i)] -= M0 * nu[i]
        for i in range(N - 1):
            t = t0 * (nu[i] * nu[i + 1] if rule == "coincidence" else 0.5 * (nu[i] + nu[i + 1]))
            A[a(i), a(i + 1)] += t; A[a(i + 1), a(i)] -= t
            A[b(i), b(i + 1)] -= t; A[b(i + 1), b(i)] += t
        return A

    ts = np.arange(0, 121, 20)
    for rule in ["1911", "coincidence"]:
        A = build(rule)
        lam, V = np.linalg.eig(A)
        env = np.exp(-(pos - N / 2) ** 2 / (2 * 40.0**2))
        x0 = np.zeros(2 * N); x0[0::2] = env                       # k = 0: bottom of the gap
        coef = np.linalg.solve(V, x0.astype(complex)); c = np.where(lam.imag > 0, coef, 0)
        cs = []
        for t in ts:
            psi = V @ (c * np.exp(lam * t)); p = np.abs(psi[0::2]) ** 2 + np.abs(psi[1::2]) ** 2
            cs.append((pos * p).sum() / p.sum())
        d = np.array(cs) - cs[0]
        acc = 2 * np.polyfit(ts, d, 2)[0]
        print(f"{rule:12s}: shifts {' '.join(f'{v:+.2f}' for v in d)} | accel {acc:+.3e}  (pred -(2 t0)^2 g = {-(2*t0)**2*g:+.3e})")

# ---------------- (B) 2D light bending ------------------------------------------------------
if which in ("all", "B"):
    print("\n=== (B) light deflection past a slow-clock region nu(r) = 1 - 4/sqrt(r^2+36) ===")
    L, alpha, soft = 700, 4.0, 6.0
    X, Y = np.meshgrid(np.arange(L) - L / 2, np.arange(L) - L / 2, indexing='ij')
    nu = 1 - alpha / np.sqrt(X**2 + Y**2 + soft**2)

    def bonds(rule):
        nx, ny = np.roll(nu, -1, 0), np.roll(nu, -1, 1)
        if rule == "coincidence":
            return nu * nx, nu * ny
        return 0.5 * (nu + nx), 0.5 * (nu + ny)

    def run(b, rule):
        jx, jy = bonds(rule)
        k, w, x0 = 0.6, 14.0, -220
        env = np.exp(-((X - x0) ** 2 + (Y - b) ** 2) / (2 * w**2))
        u = env * np.cos(k * (X - x0)); vx = -0.5 * (np.roll(u, -1, 0) + u); vy = np.zeros_like(u)
        dt, steps, cents = 0.4, 1100, []
        for s in range(steps):
            vx += dt * jx * (np.roll(u, -1, 0) - u); vy += dt * jy * (np.roll(u, -1, 1) - u)
            fx, fy = jx * vx, jy * vy
            u += dt * ((fx - np.roll(fx, 1, 0)) + (fy - np.roll(fy, 1, 1)))
            if s in (700, steps - 1):
                e = u**2; cents.append(((X * e).sum() / e.sum(), (Y * e).sum() / e.sum()))
        (x1, y1), (x2, y2) = cents
        return np.arctan2(y2 - y1, x2 - x1)

    for b in [60, 90, 130]:
        th1911 = run(b, "1911"); thc = run(b, "coincidence")
        print(f"b={b:3d}: 1911 rule {th1911:+.4f} | coincidence rule {thc:+.4f} | ratio {thc/th1911:.3f} (pred 2)")
