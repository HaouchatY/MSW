"""Leaderboard figure from results/: the fraction of model pairs each metric
orders like the exact-W2 answer key, mean over the 15 repetitions; within family
(5 pairs) and across families (40 pairs).  A tie scores 0.5, the chance level.
Writes data/lead_within.csv and data/lead_cross.csv; series whose results are
missing (ours before the rerun, the classical columns without --classical) are
left empty.
"""
import csv
import glob
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [50, 100, 250, 1000]
COLS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw", "ours", "ours_dist",
        "spec", "pswd", "ps", "ours_flat"]


def merged(pattern):
    d = {}
    for f in sorted(glob.glob(os.path.join(HERE, "results", pattern))):
        d.update({k: v for k, v in json.load(open(f)).items() if k.startswith("N")})
    return d


def acc(v, pairs, tie=0.5):
    return float(np.mean([1.0 if v[j] > v[i] else tie if v[j] == v[i] else 0.0 for i, j in pairs]))


def main():
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))
    fams = [m["family"] for m in base["models"]]
    pairs = {"within": [p for p in itertools.combinations(range(10), 2) if fams[p[0]] == fams[p[1]]],
             "cross": [p for p in itertools.combinations(range(10), 2) if fams[p[0]] != fams[p[1]]]}
    ours, cls = merged("msw*.json"), merged("classical*.json")
    reps = base["reps"]
    for tag, P in pairs.items():
        with open(os.path.join(HERE, "data", f"lead_{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["n"] + COLS)
            for N in NS:
                runs = {m: base["scores"][str(N)][m] for m in COLS[:8]}
                cells = [ours.get(f"N{N}|rep{r}") for r in range(reps)]
                if all(cells):
                    runs["ours"], runs["ours_dist"] = [c["T"] for c in cells], [c["d"] for c in cells]
                    if all("d_flat" in c for c in cells):         # the 0.2.x distance (flat weights)
                        runs["ours_flat"] = [c["d_flat"] for c in cells]
                cells = [cls.get(f"N{N}|rep{r}") for r in range(reps)]
                if all(cells):
                    runs.update({m: [c[m] for c in cells] for m in ("spec", "pswd", "ps")})
                w.writerow([N] + [f"{np.mean([acc(v, P) for v in runs[m]]):.6f}" if m in runs else "nan"
                                  for m in COLS])
        print(open(os.path.join(HERE, "data", f"lead_{tag}.csv")).read())


if __name__ == "__main__":
    main()
