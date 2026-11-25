# C11 (H10): which step-size schedule does a self-adapting optimizer choose on a noisy problem?
# Noisy sphere f(x) = |x|^2/2 + noise, d = 10.  Measure the effective exponent p of the step size  eta_t ∝ t^-p  (late times).
# PRE-REGISTERED:
#   (a) Bayes-optimal (Kalman filter for a constant under noisy gradients): eta_t = 1/(lam t)  -> p = 1  (c t = const)
#   (b) AdaGrad (noise-dominated): eta ∝ (sum g^2)^-1/2 ∝ t^-1/2  -> p = 1/2  (w = 1, stiff)
#   (c) (1+1)-ES with the 1/5 success rule: at the noise floor success < 1/5 -> sigma shrinks geometrically -> p -> infinity
#       (log sigma linear in t, not in ln t): premature freezing = event horizon.
import numpy as np

rng = np.random.default_rng(11)
d, T, noise = 10, 20_000, 0.5
lam = 1.0
checkpoints = np.unique(np.logspace(2, np.log10(T), 9).astype(int))


def fit_p(ts, etas):
    ts, etas = np.array(ts, float), np.array(etas, float)
    return -np.polyfit(np.log(ts), np.log(etas), 1)[0]


# (a) Kalman: scalar per coordinate, prior var P0, gradient obs g = lam x + n, n~N(0,R)
R, P = noise ** 2, 1.0
etas = []
for t in range(1, T + 1):
    K = P * lam / (lam ** 2 * P + R)        # gain = effective step size
    P = P - K * lam * P
    if t in checkpoints:
        etas.append(K)
print(f"(a) Kalman-optimal step: p = {fit_p(checkpoints, etas):.4f}   (eta*lam*t at end = {etas[-1]*lam*T:.4f})")

# (b) AdaGrad with noisy gradients
x = np.ones(d) * 3
G = np.zeros(d)
etas = []
for t in range(1, T + 1):
    g = lam * x + noise * rng.standard_normal(d)
    G += g ** 2
    eta = 1.0 / np.sqrt(G)
    x -= eta * g
    if t in checkpoints:
        etas.append(eta.mean())
print(f"(b) AdaGrad: p = {fit_p(checkpoints, etas):.4f}")

# (c) (1+1)-ES with 1/5 success rule on noisy f
x = np.ones(d) * 3
sigma = 1.0
fx = 0.5 * x @ x + noise * rng.standard_normal()
sig_log = []
for t in range(1, T + 1):
    y = x + sigma * rng.standard_normal(d)
    fy = 0.5 * y @ y + noise * rng.standard_normal()
    if fy <= fx:
        x, fx = y, fy
        sigma *= np.exp(0.8 / d)            # success: grow
    else:
        sigma *= np.exp(-0.2 / d)           # failure: shrink (balanced at success prob 1/5)
    if t in checkpoints:
        sig_log.append(sigma)
sig_log = np.array(sig_log)
print("(c) 1/5-rule sigma at checkpoints:", " ".join(f"{s:.1e}" for s in sig_log))
lt, ls = np.log(checkpoints), np.log(np.maximum(sig_log, 1e-300))
print(f"    fit ln sigma vs ln t : slope {np.polyfit(lt[-5:], ls[-5:], 1)[0]:.2f}   |  fit ln sigma vs t : slope {np.polyfit(checkpoints[-5:], ls[-5:], 1)[0]:.2e}")
print(f"    final |x|^2/2 = {0.5*x@x:.3f}  (stale fx = {fx:.3f})")
