# M4: sign problem. Z = int dx exp(-S), S = s/2 x^2 + l/4 x^4 with complex s. Complex Langevin (CL):
# dz = -S'(z) dt + sqrt(2 dt) eta (real noise), adaptive step. Compare <x^2> with exact quadrature.
# Diagnostics: (i) tail of the CL distribution in Im z, (ii) Aarts criterion <L z^2> = <2 - 2 z S'(z)> should be 0.
# PRE-REGISTERED: some parameters correct, some wrong; correctness tracks fast decay of the distribution.
import numpy as np
from scipy.integrate import quad

rng = np.random.default_rng(7)


def exact_x2(s, l):
    w = lambda x: np.exp(-(s / 2 * x**2 + l / 4 * x**4))
    num = quad(lambda x: (x**2 * w(x)).real, -12, 12, limit=400)[0] + 1j * quad(lambda x: (x**2 * w(x)).imag, -12, 12, limit=400)[0]
    den = quad(lambda x: w(x).real, -12, 12, limit=400)[0] + 1j * quad(lambda x: w(x).imag, -12, 12, limit=400)[0]
    return num / den


def cl(s, l, nwalk=4000, nsteps=20000, dt0=1e-3):
    z = np.zeros(nwalk, complex); acc = []; ys = []; crit = []
    for n in range(nsteps):
        drift = -(s * z + l * z**3)
        dt = np.minimum(dt0, dt0 * 10 / np.maximum(np.abs(drift), 1e-12))   # adaptive step
        z = z + drift * dt + np.sqrt(2 * dt) * rng.normal(size=nwalk)
        if n > nsteps // 4 and n % 20 == 0:
            acc.append(np.mean(z**2)); ys.append(z.imag.copy()); crit.append(np.mean(2 - 2 * z * (s * z + l * z**3)))
    ys = np.concatenate(ys)
    return np.mean(acc), np.std(acc) / np.sqrt(len(acc) / 50), np.mean(crit), ys


print(f"{'sigma':>10s} {'lambda':>6s} | {'exact <x^2>':>18s} | {'CL <x^2>':>18s} | {'|err|':>6s} | P(|Im z|>2)  P(|Im z|>4)  99.9% |Im z| | <L z^2>")
for s, l in [(1 + 1j, 1.0), (1 + 3j, 1.0), (0.2 + 1j, 1.0), (-1 + 1j, 1.0), (1 + 5j, 0.2), (0.0 + 4j, 0.1)]:
    ex = exact_x2(s, l)
    m, e, crit, ys = cl(s, l)
    ay = np.abs(ys)
    print(f"{str(s):>10s} {l:6.1f} | {ex.real:+.4f}{ex.imag:+.4f}i | {m.real:+.4f}{m.imag:+.4f}i | {abs(m-ex):6.3f} | "
          f"{np.mean(ay>2):.2e}    {np.mean(ay>4):.2e}    {np.quantile(ay, 0.999):6.2f}    | {abs(crit):.3f}")
