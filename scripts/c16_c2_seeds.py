# C16: robustness of C3 (scripts/c2_es_schedule.py): ES-BP noise at t = 1e5 over 5 seeds.
import sys, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("c2", "scripts/c2_es_schedule.py")
src = open("scripts/c2_es_schedule.py", encoding="utf8").read().split("print(f\"N={N}")[0]   # reuse definitions only
ns = {}
exec(src, ns)
run, reach = ns["run"], ns["reach"]
for p in [0.0, 0.5, 1.0, 1.5, 2.0]:
    bp = run(p, False)[100_000]
    r = reach(bp)
    vals = []
    for seed in range(5):
        es = ns["run"](p, True, seed=seed)[100_000]
        vals.append(np.sqrt(np.mean((es[:max(r,1)] - bp[:max(r,1)]) ** 2)))
    print(f"p={p}: reach {r}, noise mean {np.mean(vals):.4f} sd {np.std(vals):.4f}  values " + " ".join(f"{v:.4f}" for v in vals), flush=True)
