"""Table 3 from results/: how many of the 28 pipeline pairs each method resolves.

Baselines: a two-sided oracle threshold (2.5% / 97.5% quantiles) fitted on the
first half of the eight same-pipeline null cells pooled, the pair counted as
resolved when at least half its 30 repetitions fall outside it; and, where it
was run, an exact permutation test on the method's own features (199
relabellings).  MSW: msw.test at alpha = 0.05, no null drawn.  Writes table.csv.
"""
import csv
import glob
import itertools
import json
import os

import numpy as np

from prep import PIPELINES

HERE = os.path.dirname(os.path.abspath(__file__))
NS = (16, 64, 128)


def main():
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))
    vals, perm = base["values"], base["permutation"]
    ours = {}
    for f in glob.glob(os.path.join(HERE, "results", "msw_*.json")):
        ours.update(json.load(open(f)))
    pairs = list(itertools.combinations(PIPELINES, 2))
    out = {}
    for m in ["fid", "kid", "cmmd", "dino", "c2st", "spec", "pswd", "ps"]:
        r = {}
        for N in NS:
            nulls = [vals.get(f"null|{v}|{N}", []) for v in PIPELINES]
            if not all(x and m in x[0] for x in nulls):
                continue
            fit = sum(([d[m] for d in x[:len(x) // 2]] for x in nulls), [])
            lo, hi = np.percentile(fit, [2.5, 97.5])
            r[N] = sum(np.mean([(d[m] < lo) or (d[m] > hi) for d in vals[f"alt|{a}|{b}|{N}"]]) >= 0.5
                       for a, b in pairs)
        for N in (16, 64):
            if f"{m}|{N}" in perm:
                r[f"perm{N}"] = sum(a >= 0.5 for a in perm[f"{m}|{N}"]["alt"])
        out[m] = r
    out["msw"] = {N: sum(np.mean([p <= 0.05 for p, _ in ours[f"alt|{a}|{b}|{N}"]]) >= 0.5 for a, b in pairs)
                  for N in NS if all(f"alt|{a}|{b}|{N}" in ours for a, b in pairs)}
    out["msw_size"] = {N: float(np.mean([p <= 0.05 for v in PIPELINES for p, _ in ours.get(f"null|{v}|{N}", [])]
                                        or [np.nan])) for N in NS}
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["16", "64", "128", "perm16", "perm64"]
        w.writerow(["method"] + [f"N={c}" if c.isdigit() else c for c in cols])
        for m, r in out.items():
            if m != "msw_size":
                w.writerow([m] + [f"{r[int(c) if c.isdigit() else c]}/28" if (int(c) if c.isdigit() else c) in r
                                  else "-" for c in cols])
    print(open(os.path.join(HERE, "table.csv")).read(), "MSW size:", out["msw_size"])


if __name__ == "__main__":
    main()
