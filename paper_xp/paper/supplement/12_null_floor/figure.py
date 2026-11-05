"""fig_null from results/: offset of each metric's null reading from its ideal
value in units of its own scatter, |mean - ideal| / sd, against N.  Writes
data/null_{grf,celeba}.csv (baselines) and data/null_ours_<family>.csv (MSW).

Baselines: exp22, 15 null repetitions per N (seeds 31 r + 5); ideal 0 for the
distances, 1/2 for C2ST accuracy, Naeem et al.'s 1 - prod_i (N-i)/(2N-i) for
coverage.  MSW: the U-statistic u = sign(d) d^2 behind msw.distance d, 300
repetitions per N, ideal 0.  The MSW files also carry z = mean(u) / mcse(u),
the test of zero mean, and the jackknife interval's coverage of 0.  Values
below the axis floor are written at the floor (0.022).
"""
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR = 0.022
METRICS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw"]


def ideal(m, n):
    if m == "coverage":
        return 1 - math.prod((n - i) / (2 * n - i) for i in range(1, 6))
    return 0.5 if m == "c2st" else 0.0


def baselines(out):
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))
    for tb in ("grf", "celeba"):
        with open(os.path.join(out, f"null_{tb}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n"] + METRICS)
            for n, c in sorted(base[tb].items(), key=lambda kv: int(kv[0])):
                row = []
                for m in METRICS:
                    v = c.get(m)
                    ok = v is not None and v["mean"] == v["mean"] and v["sd"] > 0
                    row.append(f"{max(FLOOR, abs(v['mean'] - ideal(m, int(n))) / v['sd']):.6g}"
                               if ok else "nan")
                w.writerow([n] + row)


def ours(out, results):
    for fam in ("fields", "celeba32", "celeba32rs"):
        path = os.path.join(results, f"msw_{fam}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        rows = []
        for N in (8, 16, 64, 500):
            reps = d.get(f"{fam}|N{N}")
            if not reps:
                continue
            r = np.asarray(reps, dtype=float)
            u = r[:, 0] * np.abs(r[:, 0])
            sd = u.std(ddof=1)
            rows.append([N, len(u), f"{max(FLOOR, abs(u.mean()) / sd):.6g}",
                         f"{u.mean() / (sd / math.sqrt(len(u))):.4f}",
                         f"{u.mean():.6g}", f"{sd:.6g}",
                         f"{np.mean((r[:, 1] <= 0) & (r[:, 2] >= 0)):.4f}"])
        with open(os.path.join(out, f"null_ours_{fam}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "reps", "off", "z", "mean_u", "sd_u", "ci_covers0"])
            w.writerows(rows)
        print(fam, *(f"N={r[0]}: off {r[2]} z {r[3]} cover {r[6]}" for r in rows), sep="\n  ")


if __name__ == "__main__":
    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)
    baselines(out)
    ours(out, os.environ.get("NULL_RESULTS", os.path.join(HERE, "results")))
