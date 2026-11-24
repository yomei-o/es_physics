# D14: why only ONE time direction?
# Event lattice in 3 directions. Each direction mu is either cooperative (s=+1, symmetric coupling) or
# zero-sum (s=-1: coupling carries i, i^2=-1 flips the sign of that direction's term in the action).
# Action Q = sum_mu s_mu (lattice derivative_mu phi)^2  ->  equation sum_mu s_mu D_mu^2 phi = 0.
# Evolve along direction 0:  phi'' = -s_0 * sum_{mu!=0} s_mu D_mu^2 phi ; in Fourier (D^2 -> -K_mu, K = 4 sin^2(k/2)):
#   first-order generator G = [[0, 1], [s_0 * sum_{mu!=0} s_mu K_mu, 0]]
# PRE-REGISTERED:
#   0 zero-sum directions: G has real eigenvalues for every k (exponential growth): no direction can serve as time
#   1 zero-sum direction, evolve along it: all eigenvalues purely imaginary (pure 'i', zero-sum, stable, light cone)
#   2 zero-sum directions: evolving along either one, some k give real eigenvalues (blow-up) -> no good time
import numpy as np

rng = np.random.default_rng(11)
L = 64
ks = 2 * np.pi * np.fft.fftfreq(L)
K1, K2 = np.meshgrid(4 * np.sin(ks / 2) ** 2, 4 * np.sin(ks / 2) ** 2, indexing='ij')


def analyse(signs, along):
    others = [m for m in range(3) if m != along]
    coef = signs[along] * (signs[others[0]] * K1 + signs[others[1]] * K2)   # phi'' = coef * phi  (Fourier)
    # eigenvalues of G are +-sqrt(coef): real if coef>0
    max_growth = np.sqrt(np.maximum(coef, 0)).max()
    frac_bad = np.mean(coef > 1e-12)
    return max_growth, frac_bad


def simulate(signs, along, steps=400, dt=0.2):
    others = [m for m in range(3) if m != along]
    coef = signs[along] * (signs[others[0]] * K1 + signs[others[1]] * K2)
    phi = np.fft.fft2(rng.standard_normal((L, L)) * 1e-6); pi = np.zeros_like(phi)
    n0 = np.linalg.norm(phi)
    for _ in range(steps):                      # symplectic leapfrog
        pi += 0.5 * dt * coef * phi; phi += dt * pi; pi += 0.5 * dt * coef * phi
    return np.linalg.norm(phi) / n0


cases = {
    "0 zero-sum (+,+,+)": [1, 1, 1],
    "1 zero-sum (-,+,+)": [-1, 1, 1],
    "2 zero-sum (-,-,+)": [-1, -1, 1],
    "3 zero-sum (-,-,-)": [-1, -1, -1],
}
print(f"{'signature':22s} {'evolve along':>12s}  {'max growth rate':>15s}  {'fraction of modes blowing up':>28s}  {'|phi(T)|/|phi(0)|':>18s}")
for name, s in cases.items():
    for along in range(3):
        g, f = analyse(s, along)
        amp = simulate(s, along)
        kind = "zero-sum" if s[along] == -1 else "cooperative"
        print(f"{name:22s} {along} ({kind:11s})  {g:15.3f}  {f:28.3f}  {amp:18.3g}")


print("\n4D: which signatures allow a stable evolution direction?  (stable along mu iff s_mu*s_nu = -1 for all nu != mu)")
import itertools
for p in range(5):
    s = [-1] * p + [1] * (4 - p)
    stable = [mu for mu in range(4) if all(s[mu] * s[nu] == -1 for nu in range(4) if nu != mu)]
    # numeric confirmation on a 4D Fourier grid
    kk = 4 * np.sin(np.pi * np.fft.fftfreq(12)) ** 2
    grids = np.meshgrid(kk, kk, kk, indexing='ij')
    ok = []
    for mu in range(4):
        others = [nu for nu in range(4) if nu != mu]
        coef = s[mu] * sum(s[nu] * g for nu, g in zip(others, grids))
        ok.append(bool((coef <= 1e-12).all()))
    print(f"  signature {tuple(s)} ({p} of one kind, {4-p} of the other): stable directions (rule) {stable}, (numeric) {[m for m in range(4) if ok[m]]}")
