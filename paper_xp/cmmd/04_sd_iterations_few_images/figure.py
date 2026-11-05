"""results/smalln_sd.json -> data/smalln.csv (percentages with 95% Wilson intervals)"""
import json, os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{H}/results/smalln_sd.json"))


def wilson(k, n, z=1.96):
    ph, den = k / n, 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


os.makedirs(f"{H}/data", exist_ok=True)
with open(f"{H}/data/smalln.csv", "w") as f:
    cols = [f"{m}_{k}{s}" for m in ("fid", "cmmd", "msw") for k in ("perf", "last") for s in ("", "_ep", "_em")]
    f.write("N,reps," + ",".join(cols) + "\n")
    for N in sorted({r["N"] for r in R}):
        rows = [r for r in R if r["N"] == N]; n = len(rows); vals = []
        for m in ("fid", "cmmd", "msw"):
            for k in (sum(r[f"tau_{m}"] == 1.0 for r in rows), sum(r[m][4] > r[m][5] for r in rows)):
                lo, hi = wilson(k, n); ph = k / n
                vals += [100 * ph, 100 * (hi - ph), 100 * (ph - lo)]
        f.write(f"{N},{n}," + ",".join(f"{v:.3f}" for v in vals) + "\n")
