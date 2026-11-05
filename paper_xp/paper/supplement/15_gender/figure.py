"""Figure data from results/: per contrast and rung, the deflection |z| = |mean_alt -
mean_null| / sd_null of every method (ours on msw.test's T), and N*, the smallest
rung with power >= 0.8 that stays there.  Baselines detect beyond their own null's
95th percentile (coverage below the 5th); MSW when msw.test rejects at 0.05.

    python figure.py      # writes data/gender_{drop,pure}.csv and data/nstar.csv
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import Store, n_star, rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
RUNGS = [128, 256, 512, 1024, 2048, 4096]
BASE = ["fid", "kid", "cmmd", "c2st", "coverage", "sw", "csw"]
# pure stops at 2048: its comparison pool has 3369 males, so at 4096 the null puts the
# whole pool on both sides (degenerate for every method; the baselines' 4096 cell is dropped).
STOP = {"drop": 4096, "pure": 2048}
ZC = 2.4865                       # |z| giving power 0.8 at one-sided 0.05 (normal approximation)


def ours(contrast):
    path = os.path.join(HERE, "results", f"msw_{contrast}.json")
    st = Store(path).d if os.path.exists(path) else {}
    out = {}
    for N in RUNGS:
        n0, n1 = st.get(f"{contrast}|N{N}|null", []), st.get(f"{contrast}|N{N}|alt", [])
        if len(n0) > 1 and n1:
            T0, T1 = np.array([x[1] for x in n0]), np.array([x[1] for x in n1])
            out[N] = dict(z=(T1.mean() - T0.mean()) / T0.std(ddof=1),
                          power=rejection_rate([x[0] for x in n1]),
                          size=rejection_rate([x[0] for x in n0]))
    return out


if __name__ == "__main__":
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))["cells"]
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    ns = []
    for c in ("drop", "pure"):
        o, b = ours(c), base[c]
        rungs = [N for N in RUNGS if N <= STOP[c]]
        get = lambda m, N, f: b.get(str(N), {}).get(m, {}).get(f)
        with open(os.path.join(HERE, "data", f"gender_{c}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["n"] + BASE + ["ours", "ours_power", "ours_size"])
            for N in rungs:
                zs = [get(m, N, "z") for m in BASE]
                w.writerow([N] + ["nan" if z is None else f"{abs(z):.6g}" for z in zs]
                           + ([f"{abs(o[N]['z']):.6g}", f"{o[N]['power']:.3f}", f"{o[N]['size']:.3f}"]
                              if N in o else ["nan"] * 3))
        for m in BASE:
            ns.append([c, m, n_star(rungs, lambda N: get(m, N, "power")),
                       n_star(rungs, lambda N: None if get(m, N, "z") is None else abs(get(m, N, "z")),
                              level=ZC)])
        if o:
            ns.append([c, "msw", n_star(rungs, lambda N: o[N]["power"] if N in o else None),
                       n_star(rungs, lambda N: abs(o[N]["z"]) if N in o else None, level=ZC)])
    with open(os.path.join(HERE, "data", "nstar.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["contrast", "method", "nstar_power", "nstar_absz"])
        w.writerows(ns)
    print(open(os.path.join(HERE, "data", "nstar.csv")).read())
