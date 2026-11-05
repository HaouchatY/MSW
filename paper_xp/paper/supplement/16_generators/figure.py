"""Real-generators figure from results/: deflection z = |mean_alt - mean_null| / sd_null
against the real-vs-real null at the same N, for every baseline (results/baselines.json,
10 repetitions) and for ours -- the test statistic T (50 repetitions) and
msw.distance (20 repetitions, ladder panels only).  Writes data/gen_ddpm.csv,
data/gen_gan.csv, data/gen_pairs128.csv (ours_rej = exact-test rejection rate at
0.05), the Kendall tau of every series against the ladder order, and, when the
pools are present under $MSW_XP_DATA/gen_pools, the sample tiles data/gen_*.png
(DDPM sample 1, DCGAN sample 0 of each pool, as in the paper).
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DATA
from protocol import rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS", os.path.join(HERE, "results"))
OUT = os.path.join(HERE, "data")
RUNGS = {"ddpm": [2, 5, 10, 20, 50], "gan": [200, 500, 1000, 2000, 5000, 10000]}
PAIRS = [(t, f"s{a}-s{b}") for t in RUNGS for a, b in zip(RUNGS[t][:-1], RUNGS[t][1:])]
BASE = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw"]
FLOOR = 0.05                                    # the log axis' floor


def z(null, alt):
    if not null or not alt:
        return float("nan")
    v0, v1 = np.asarray(null, float), np.asarray(alt, float)
    return max(abs(v1.mean() - v0.mean()) / v0.std(ddof=1), FLOOR)


def kendall(v):
    """tau of v against the ladder order (rung index); -1 = decreasing = perfect."""
    s = [np.sign(v[j] - v[i]) * np.sign(j - i) for i in range(len(v)) for j in range(i + 1, len(v))]
    return float(np.mean(s))


def main():
    base = json.load(open(os.path.join(RES, "baselines.json")))["cells"]
    ours = {}
    for t in RUNGS:
        p = os.path.join(RES, f"msw_{t}.json")
        if os.path.exists(p):
            ours.update(json.load(open(p)))
    T = lambda c: [x[1] for x in ours.get(c, [])]
    D = lambda c: [x[0] for x in ours.get(c + "|dist", [])]
    rej = lambda c: rejection_rate([x[0] for x in ours[c]]) if c in ours else float("nan")
    os.makedirs(OUT, exist_ok=True)
    hdr = BASE + ["ours", "ours_dist", "ours_rej"]
    taus = {}
    for t, rungs in RUNGS.items():
        rows = []
        for S in rungs:
            n, a = f"{t}/null@1024", f"{t}/s{S}@1024"
            rows.append([S] + [z(base[n][m], base[a][m]) for m in BASE]
                        + [z(T(n), T(a)), z(D(n), D(a)), rej(a)])
        write(f"gen_{t}.csv", ["x"] + hdr, rows)
        taus[t] = {m: kendall([r[1 + i] for r in rows]) for i, m in enumerate(hdr[:-1])}
    rows = []
    for i, (t, nm) in enumerate(PAIRS, 1):
        n, a = f"{t}/null@128", f"{t}/{nm}@128"
        rows.append([i] + [z(base[n][m], base[a][m]) for m in BASE] + [z(T(n), T(a)), rej(a)])
    write("gen_pairs128.csv", ["i"] + BASE + ["ours", "ours_rej"], rows)
    json.dump(taus, open(os.path.join(OUT, "kendall_tau.json"), "w"), indent=1)
    for t in taus:
        print(t, "Kendall tau vs ladder order:", {m: round(v, 3) for m, v in taus[t].items()})
    tiles()


def write(name, hdr, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(hdr)
        w.writerows([[r[0]] + ["%.4g" % v for v in r[1:]] for r in rows])
    print(open(os.path.join(OUT, name)).read())


def tiles():
    import torch
    from PIL import Image
    for t, i in [("ddpm", 1), ("gan", 0)]:
        for S in RUNGS[t]:
            p = os.path.join(DATA, "gen_pools", f"{t}_s{S}.pt")
            if os.path.exists(p):
                x = torch.load(p, map_location="cpu")[i].permute(1, 2, 0).numpy()
                Image.fromarray((x * 255).astype(np.uint8)).save(os.path.join(OUT, f"gen_{t}_{S}.png"))


if __name__ == "__main__":
    main()
