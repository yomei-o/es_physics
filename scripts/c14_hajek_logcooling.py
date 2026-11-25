# C14 (H13): Hajek's logarithmic-cooling theorem is the c*t = const boundary for escape rates.
# Two-state Arrhenius model: local well (barrier d1 = 1), global well (barrier d2 = 2), rates A exp(-d/T).
# Cooling T = c / ln t  ->  escape rate from the local well = A t^{-d1/c}: a schedule with exponent p = d1/c.
# Escape attempts ∫ A t^{-p} dt diverge iff p <= 1 iff c >= d1  (Hajek 1988: c >= depth of deepest non-global well).
# Integrate in log time s = ln t (the c t = const frame): dP/ds = a(s)(1-P) - b(s) P,
#   a = A e^{s(1 - d1/c)}, b = A e^{s(1 - d2/c)}; exact exponential update (stiff-safe).
# History: a first run (A = 1, explicit RK4) blew up for c >= 2 and gave poor contrast; rerun with A = 0.01 and this integrator.
# PRE-REGISTERED (for the rerun): c < 1: P_inf ≈ A c/(1-c) (small);  c = 1: 1 - P ≈ exp(-A s) -> P(200) ≈ 0.86;  c > 1: P -> 1.
import numpy as np
d1, d2, A = 1.0, 2.0, 0.01
print("   c     P(s=10)    P(s=50)    P(s=200)   prediction")
for c in [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 2.0, 3.0]:
    P, h = 0.0, 1e-3
    out = {}
    for n in range(1, 200_001):
        s = (n - 0.5) * h
        a = A * np.exp(s * (1 - d1 / c)); b = A * np.exp(s * (1 - d2 / c))
        Peq = a / (a + b)
        P = Peq + (P - Peq) * np.exp(-(a + b) * h)
        if n in (10_000, 50_000, 200_000): out[n // 1000] = P
    if c < 1: pred = f"A c/(1-c) = {A*c/(1-c):.4f}"
    elif c == 1: pred = f"1-exp(-A s) = {1-np.exp(-A*200):.4f} (s=200)"
    else: pred = "-> 1"
    print(f"{c:5.2f}  {out[10]:.6f}   {out[50]:.6f}   {out[200]:.6f}   {pred}")
