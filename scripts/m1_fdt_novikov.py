# M1: response of a Langevin system measured three ways (see es_methods_notes.html M0).
# dx = (-U'(x) + h(t) + F cos(w t)) dt + sqrt(2T) dW,  U = x^4/4 - x^2/2
import numpy as np

rng = np.random.default_rng(0)
T, dt = 0.5, 0.01
nsteps_resp = 300                    # response window: t - s = 0 .. 3.0
lag_idx = np.arange(0, nsteps_resp, 25)


def run(F, w, nwalk, nburn=3000, s_index=4000, h=0.0, noise=None, record_noise=False):
    """simulate nwalk independent walkers; a kick h applied at step s_index (one step), return x(t) after s and noise at s"""
    x = rng.normal(size=nwalk) if noise is None else np.zeros(nwalk)
    total = s_index + nsteps_resp
    if noise is None:
        noise = rng.normal(size=(total, nwalk))
    x = np.zeros(nwalk) + 0.0
    xs = np.zeros((nsteps_resp, nwalk))
    for n in range(total):
        t = n * dt
        f = -(x**3 - x) + F * np.cos(w * t)
        if n == s_index: f = f + h / dt          # delta-function kick of strength h at time s
        x = x + f * dt + np.sqrt(2 * T * dt) * noise[n]
        if n >= s_index: xs[n - s_index] = x
    return xs, noise


def responses(F, w, nwalk=40000, h=0.05):
    xs0, noise = run(F, w, nwalk)
    xs1, _ = run(F, w, nwalk, h=h, noise=noise)          # (a) common random numbers
    s_index = 4000
    Ra = (xs1 - xs0).mean(1) / h
    # (b) Novikov / ES: R(t,s) = <x(t) xi(s)> / (2T) with xi(s) dt-integrated noise -> noise[s]*sqrt(2T dt)/dt
    xi = noise[s_index] * np.sqrt(2 * T * dt) / dt
    prod = xs0 * xi[None, :] / (2 * T)
    Rb = prod.mean(1)                       # Girsanov/Novikov weight: x * dW / (2T dt)
    Rb_err = prod.std(1) / np.sqrt(nwalk)
    # (c) equilibrium FDT from autocorrelation of the unperturbed run: R = -(1/T) dC/dt, C(t) = <x(s) x(s+t)> - <x>^2
    x_s = xs0[0]
    C = np.array([np.mean(x_s * xs0[k]) - x_s.mean() * xs0[k].mean() for k in range(nsteps_resp)])
    Rc = np.full(nsteps_resp, np.nan); w10 = 10           # central difference over +-0.1 (dt-scale differences amplify noise)
    Rc[w10:-w10] = -(C[2*w10:] - C[:-2*w10]) / (2 * w10 * dt) / T
    return Ra, Rb, Rb_err, Rc, (xs1 - xs0).std(1) / h / np.sqrt(nwalk)


for label, F, w in [("equilibrium (F=0)", 0.0, 0.0), ("driven out of equilibrium (F=0.6, w=1.0)", 0.6, 1.0)]:
    Ra, Rb, Rbe, Rc, Rae = responses(F, w)
    print(f"\n=== {label} ===")
    print(" t-s    (a) direct kick      (b) ES/Novikov (no kick)      (c) equilibrium FDT")
    for k in lag_idx[1:-1]:
        print(f" {k*dt:4.2f}   {Ra[k]:+.4f} ({Rae[k]:.4f})    {Rb[k]:+.4f} ({Rbe[k]:.4f})            {Rc[k]:+.4f}")
    sel = lag_idx[1:-1]
    print(f" mean |b-a|/err_b = {np.mean(np.abs(Rb[sel]-Ra[sel])/Rbe[sel]):.2f} ;  mean |c-a| = {np.mean(np.abs(Rc[sel]-Ra[sel])):.4f} vs mean |a| = {np.mean(np.abs(Ra[sel])):.4f}")
    print(f" variance ratio (b error / a error) at t-s=0.5: {Rbe[50]/Rae[50]:.1f}")
