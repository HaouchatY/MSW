"""fig:modedrop data from results/: the deflection |z| of every method against N on
the three mode-dropping panels, and MSW's N* (first rung with power >= 0.8 that
stays there).  Writes data/{md_cifar,gender_stat,md_fields}.csv (baselines),
data/ours_md_{cifar,gender,fields}.csv and data/nstar.csv.

Baselines: |mean_alt - mean_null| / sd_null of their own null repetitions
(results/baselines.json).  MSW: (mean T_alt - mean T_null) / sd T_null of msw.test.
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import n_star, rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MD_OUT", os.path.join(HERE, "results"))
DATA = os.path.join(HERE, "data")
METRICS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw"]
PANELS = {"cifar": ("md_cifar", 170), "gender": ("gender_stat", 430), "fields": ("md_fields", 170)}


def ours(panel):
    d = {}
    for f in glob.glob(os.path.join(RES, f"msw_{panel}_N*.json")):
        d.update(json.load(open(f)))
    out = {}
    for k in d:
        _, n, side = k.split("|")
        if side != "alt" or f"{panel}|{n}|null" not in d:
            continue
        a, b = np.array(d[k], float), np.array(d[f"{panel}|{n}|null"], float)
        if len(a) < 2 or len(b) < 2:
            continue
        sd = b[:, 1].std(ddof=1)
        out[int(n[1:])] = dict(
            z=(a[:, 1].mean() - b[:, 1].mean()) / sd,
            se=math.sqrt(a[:, 1].var(ddof=1) / len(a) + b[:, 1].var(ddof=1) / len(b)) / sd,
            power=rejection_rate(a[:, 0]), size=rejection_rate(b[:, 0]), reps=len(a))
    return dict(sorted(out.items()))


def main():
    os.makedirs(DATA, exist_ok=True)
    base = json.load(open(os.path.join(RES, "baselines.json")))["cells"]
    rows = []
    for panel, (name, xmin) in PANELS.items():
        with open(os.path.join(DATA, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n"] + METRICS)
            for N in sorted(base[panel], key=int):
                c = base[panel][N]
                w.writerow([N] + [f"{c[m]['z']:.6g}" if m in c else "nan" for m in METRICS])
        o = ours(panel)
        if not o:
            continue
        with open(os.path.join(DATA, f"ours_md_{panel}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "ours", "se", "power", "size", "reps"])
            for N, r in o.items():
                if N >= xmin:
                    w.writerow([N, f"{r['z']:.4f}", f"{r['se']:.4f}", f"{r['power']:.4f}",
                                f"{r['size']:.4f}", r["reps"]])
        ns = n_star(list(o), lambda N: o[N]["power"])
        rows.append([panel, ns] + [f"{N}:{o[N]['power']:.3f}" for N in o])
        print(panel, "N* =", ns, " ".join(f"{N}:z={r['z']:.2f}/pw={r['power']:.2f}/size={r['size']:.3f}"
                                         for N, r in o.items()))
    with open(os.path.join(DATA, "nstar.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "n_star", "power_by_rung"])
        for r in rows:
            w.writerow(r[:2] + [" ".join(r[2:])])


if __name__ == "__main__":
    main()
