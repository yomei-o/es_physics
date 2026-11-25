# K2-K6: tests of the pre-registered rule R (koide_notes.html K1). Mass inputs are from memory -> check before citing.
import numpy as np, itertools
from scipy.optimize import brentq


def Q(m, signs=(1, 1, 1)):
    m = np.asarray(m, float); s = np.asarray(signs) * np.sqrt(m)
    return m.sum() / s.sum() ** 2


def delta(m, signs=(1, 1, 1)):
    s = np.asarray(signs) * np.sqrt(np.asarray(m, float))
    F = np.fft.fft(s) / 3
    r = 2 * abs(F[1]) / F[0].real
    ph = np.angle(F[1]) % (2 * np.pi / 3)
    return r, ph


print("=== K2 charged leptons (pole masses, MeV) ===")
lep = [0.51099895, 105.6583755, 1776.86]
r, ph = delta(lep)
print(f"Q = {Q(lep):.6f}, r = {r:.5f} (sqrt2 = 1.41421), phase mod 2pi/3 = {ph:.5f} (2/9 = {2/9:.5f})")
lep_mz = [0.486570, 102.718, 1746.24]  # running at M_Z (from memory)
print(f"running masses at M_Z: Q = {Q(lep_mz):.6f}")

print("\n=== K3 quarks (rule R predicts 2/3, pass if within 1 %) ===")
sets = {
    "PDG mixed schemes": dict(u=2.16, d=4.67, s=93.4, c=1270, b=4180, t=172690),
    "all running at M_Z": dict(u=1.27, d=2.90, s=55.0, c=619, b=2890, t=171700),
}
for name, q in sets.items():
    for trip in [("u", "c", "t"), ("d", "s", "b")]:
        val = Q([q[x] for x in trip])
        print(f"{name:20s} {''.join(trip)}: Q = {val:.4f}  ({100*(val/(2/3)-1):+.1f} %)  -> {'PASS' if abs(val/(2/3)-1) < 0.01 else 'FAIL'}")

print("\n=== K3b look-elsewhere: all triplets of the 9 charged fermions ===")
allm = dict(e=lep[0], mu=lep[1], tau=lep[2], **sets["PDG mixed schemes"])
hits = []
for trip in itertools.combinations(allm, 3):
    v = Q([allm[x] for x in trip])
    if abs(v / (2 / 3) - 1) < 0.01: hits.append((trip, round(v, 4)))
print(f"{len(hits)} of 84 triplets within 1 % of 2/3: {hits}")
# null model: masses log-uniform over the same range (0.5 MeV .. 173 GeV)
rng = np.random.default_rng(0)
lo, hi = np.log(0.5), np.log(173000)
cnt = 0; trials = 20000; best = []
for _ in range(trials):
    ms = np.exp(rng.uniform(lo, hi, 9))
    k = sum(abs(Q(c) / (2 / 3) - 1) < 0.01 for c in itertools.combinations(ms, 3))
    cnt += k > 0; best.append(k)
print(f"null model: P(at least one of 84 random triplets within 1 %) = {cnt/trials:.3f}, mean number = {np.mean(best):.2f}")
cnt2 = sum(abs(Q(np.exp(rng.uniform(lo, hi, 3))) / (2 / 3) - 1) < 1e-5 for _ in range(200000))
print(f"null model: P(a single random triplet within 1e-5 of 2/3, like the leptons) ~ {cnt2/200000:.1e}")

print("\n=== K4 neutrinos (Delta m^2 from memory: 7.42e-5, 2.514e-3 eV^2) ===")
dm21, dm31 = 7.42e-5, 2.514e-3
def masses_NO(m1): return [m1, np.sqrt(m1**2 + dm21), np.sqrt(m1**2 + dm31)]
def masses_IO(m3): m1 = np.sqrt(m3**2 + dm31 - dm21); return [m1, np.sqrt(m1**2 + dm21), m3]
grid = np.logspace(-6, 0, 2000)
for name, f in [("normal", masses_NO), ("inverted", masses_IO)]:
    qs = [Q(np.square(f(x))) if False else Q(f(x)) for x in grid]
    print(f"{name} ordering, all sqrt(m) positive: max Q over lightest mass = {max(qs):.4f} (needs 2/3) -> "
          f"{'solution exists' if max(qs) >= 2/3 else 'NO SOLUTION: rule R with positive amplitudes is excluded'}")
# variant with one negative amplitude (Brannen) -- NOT in the pre-registration
for sgn in [(-1, 1, 1), (1, -1, 1)]:
    g = lambda m1: Q(masses_NO(m1), sgn) - 2 / 3
    xs = grid[[i for i in range(len(grid) - 1) if np.sign(g(grid[i])) != np.sign(g(grid[i + 1]))]]
    for x in xs:
        m1 = brentq(g, x, x * 1.01); ms = masses_NO(m1)
        r_, ph_ = delta(ms, sgn)
        print(f"  (post hoc) NO with signs {sgn}: m = {np.round(ms, 5)} eV, sum = {sum(ms):.4f} eV, phase mod 2pi/3 = {ph_:.4f}")

print("\n=== K5 phases of near-2/3 triplets ===")
for trip in [("c", "b", "t")]:
    m = [allm[x] for x in trip]
    print(f"{trip}: Q = {Q(m):.4f}, r = {delta(m)[0]:.4f}, phase = {delta(m)[1]:.4f}  (lepton 0.2222)")

print("\n=== K6 Z3-symmetric zero-sum dynamics: angle of the long-time selected mode ===")
angles = []
for _ in range(2000):
    c = rng.normal(size=3)
    W = np.array([[c[(j - i) % 3] for j in range(3)] for i in range(3)])  # circulant (Z3 symmetric)
    w, V = np.linalg.eig(W)
    v = V[:, np.argmax(w.real)]
    v = np.real(v) if abs(np.imag(v)).max() < 1e-12 else np.real(v)  # real part of the rotating plane
    ang = np.degrees(np.arccos(min(1, abs(v.sum()) / np.sqrt(3) / np.linalg.norm(v))))
    angles.append(round(ang, 3))
vals, counts = np.unique(np.round(angles, 1), return_counts=True)
print("angles of selected modes (deg) and counts:", dict(zip(vals, counts)))
# resonance (singlet and doublet degenerate): random initial amplitudes, equipartition per REAL dof
s = rng.normal(size=100000); d = rng.normal(size=(100000, 2))
P = s**2 / (s**2 + (d**2).sum(1))
print(f"degenerate case, random initial state: mean P_sym = {P.mean():.3f} (per-real-dof 1/3, rule R 1/2), "
      f"P(|P_sym-1/2|<0.005) = {np.mean(abs(P-0.5)<0.005):.4f}")
