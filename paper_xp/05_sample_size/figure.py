"""results/sample_size.json -> data/sample_size.csv (mean, sd, and % of the value at the largest N)"""
import json, os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{H}/results/sample_size.json"))
Ns = sorted({r["N"] for r in R})
ref = {m: np.mean([r[m] for r in R if r["N"] == Ns[-1]]) for m in ("fid", "cmmd", "msw")}
os.makedirs(f"{H}/data", exist_ok=True)
with open(f"{H}/data/sample_size.csv", "w") as f:
    f.write("N," + ",".join(f"{m},{m}_sd,{m}_rel,{m}_rel_sd" for m in ("fid", "cmmd", "msw")) + "\n")
    for N in Ns:
        vals = []
        for m in ("fid", "cmmd", "msw"):
            v = np.array([r[m] for r in R if r["N"] == N]); rel = 100 * v / ref[m]
            sd = v.std(ddof=1) if len(v) > 1 else 0.0; rsd = rel.std(ddof=1) if len(v) > 1 else 0.0
            vals += [v.mean(), sd, rel.mean(), rsd]
        f.write(f"{N}," + ",".join(f"{x:.6g}" for x in vals) + "\n")
