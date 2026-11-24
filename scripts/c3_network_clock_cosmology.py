# C3: a uniform, slowly changing network clock nu(T) is cosmic expansion.
# From the coincidence rule (notes §8): per tick, mass/clock frequency ∝ nu, hopping (light) ∝ nu^2 sites/tick,
# physical ruler = nu sites.  =>  proper time d tau = nu dT, scale factor a = 1/nu, locally measured c = 1.
# Network-unit light speed c_net = nu^2.  "c * t = const" in network units: c_net(T) * T = const.
# Family: c_net = T^-p  (p = step-size exponent of C2).
# PRE-REGISTERED: a ∝ tau^n with n = p/(2-p).  p=2/3 radiation (n=1/2), p=0.8 matter (n=2/3),
#                 p=1 coasting (n=1, c t = const), p=2 de Sitter (exponential).
# Also: horizon in sites  sum c_net  ~ T^(1-p): past-finite iff p<1 (horizon problem), future-finite iff p>1.
import numpy as np

T = np.logspace(0, 12, 200001)
for p in [0.5, 2/3, 0.8, 1.0, 1.5, 1.9]:
    nu = T ** (-p / 2)
    tau = np.concatenate([[0], np.cumsum(0.5 * (nu[1:] + nu[:-1]) * np.diff(T))])
    a = 1 / nu
    sel = slice(len(T) // 2, None)
    n_fit = np.polyfit(np.log(tau[sel] + 1e-300), np.log(a[sel]), 1)[0]
    print(f"p={p:.3f}: fitted n = {n_fit:.4f}, predicted p/(2-p) = {p/(2-p):.4f}")

p = 2.0
nu = T ** (-1.0)
tau = np.concatenate([[0], np.cumsum(0.5 * (nu[1:] + nu[:-1]) * np.diff(T))])
H = np.gradient(np.log(1 / nu), tau)
print(f"p=2: H at tau = {tau[50000]:.2f}, {tau[150000]:.2f}, {tau[-1]:.2f}: {H[50000]:.4f}, {H[150000]:.4f}, {H[-2]:.4f}  (constant => de Sitter)")

# LambdaCDM today in this dictionary: n = H t  -> p = 2n/(1+n)
for name, n in [("radiation", 0.5), ("matter", 2/3), ("LCDM today (n=0.9506)", 0.9506), ("H t = 1 epoch", 1.0)]:
    print(f"{name:>24}: effective step exponent p = {2*n/(1+n):.4f}")
