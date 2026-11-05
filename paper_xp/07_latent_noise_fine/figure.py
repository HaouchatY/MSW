"""results/*.json -> data/comp_{coco,sd,levels}.csv (change from p = 0, in %; per-level change of u)"""
import json, os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{H}/data", exist_ok=True)
for key, pmax, name in (("cocoB_vq", 0.05, "comp_coco"), ("sdB_vq", 0.20, "comp_sd")):
    S = json.load(open(f"{H}/results/series_{key}.json"))
    rows = [dict(r, p=x) for x, r in zip(S["x"], S["rows"]) if x <= pmax + 1e-9]
    base = rows[0]
    with open(f"{H}/data/{name}.csv", "w") as f:
        f.write("p,fid,cmmd,msw\n")
        for r in rows:
            f.write(f"{r['p']}," + ",".join(f"{100 * (r[m] / base[m] - 1):.4f}" for m in ("fid", "cmmd", "msw")) + "\n")
L = json.load(open(f"{H}/results/levels.json"))
u0 = np.array(L["0.0"])
with open(f"{H}/data/comp_levels.csv", "w") as f:
    f.write("level,d001,d002,d005\n")
    for lvl in range(len(u0)):
        f.write(f"{lvl}," + ",".join(f"{1e5 * (L[k][lvl] - u0[lvl]):.4f}" for k in ("0.01", "0.02", "0.05")) + "\n")
