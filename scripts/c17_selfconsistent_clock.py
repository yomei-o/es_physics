# C17: can optimal learning fix the network clock schedule nu(T)?  (trial model, rules partly arbitrary)
# N agents estimate a shared constant from noisy neighbour messages, Kalman gain = 1/(n_i + 1), n_i = messages received.
# (i)  intake fixed: Poisson(1) messages per tick                      -> gain ∝ 1/T -> p = 1
# (ii) intake ∝ hopping c_net (coincidence rule, rate ∝ nu_i nu_j = c_net), and c_net := population-mean gain
#      -> dn/dT = c = 1/n -> n = sqrt(2T) -> p = 1/2  (w = 1, stiff)
# PRE-REGISTERED: (i) p = 1, (ii) p = 1/2; neither gives LCDM's present p = 0.975 or the matter era 0.8.
import numpy as np
rng = np.random.default_rng(17)
N, T = 20_000, 100_000
for mode in ["fixed", "coupled"]:
    n = np.zeros(N); est = np.zeros(N); c = 1.0
    ts, cs, errs = [], [], []
    for t in range(1, T + 1):
        rate = 1.0 if mode == "fixed" else c
        k = rng.poisson(rate, N)
        got = k > 0
        msg = 1.0 + rng.standard_normal(N) / np.sqrt(np.maximum(k, 1))   # mean of k noisy messages about the constant 1
        gain = k / (n + k + 1e-12)
        est[got] += gain[got] * (msg[got] - est[got])
        n += k
        c = np.mean(1 / (n + 1))
        if t in (1_000, 3_000, 10_000, 30_000, 100_000):
            ts.append(t); cs.append(c); errs.append(np.mean((est - 1) ** 2))
    p = -np.polyfit(np.log(ts), np.log(cs), 1)[0]
    print(f"{mode:>8}: p = {p:.3f}   c(T) = " + " ".join(f"{x:.2e}" for x in cs) + "   MSE*n = " + " ".join(f"{e*nn:.2f}" for e, nn in zip(errs, [1/x for x in cs])))
