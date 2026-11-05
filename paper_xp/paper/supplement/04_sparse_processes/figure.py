"""Figure fig:sspresults from results/: deflection |z| against N for the
near-integrator benchmark, 1D (top) and 2D (bottom), on Laplace, Student-t4,
Bernoulli-Laplace p = 0.05 and the operator control.  Writes
data/ssp_{d1,d2}_{laplace,student4,bernlap,opshift}.csv and data/ours_detail.csv.

Baselines (12 repetitions, results/baselines.json): z = (mean_alt - mean_null) /
sd_null of the metric's value; MSW (results/msw_near_*.json): the same on the
msw.test statistic T.  Plotted |z|, floored at 0.05 for the log axis; a
zero-variance coverage null (|z| > 1e4) is shown at the floor.

    python figure.py && pdflatex figure.tex
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = {"d1": ["spectrum", "sw", "fwd", "wmmd", "wc2st"],
        "d2": ["spectrum", "fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw"]}
PANELS = [("laplace", "laplace"), ("student4", "student4"), ("bernlap_0.05", "bernlap"),
          ("opshift", "opshift")]
NS = {"d1": [8, 32, 128], "d2": [8, 32, 128, 500]}


def z(alt, null):
    v0 = np.asarray(null, dtype=float)
    return (np.mean(alt) - v0.mean()) / max(v0.std(ddof=1), 1e-12)


def ours(op, dom, results=os.path.join(HERE, "results")):
    """{(contrast, N): (z, power, size, reps_alt, reps_null)} from msw_<op>_<dom>.json."""
    path = os.path.join(results, f"msw_{op}_{dom}.json")
    if not os.path.exists(path):
        return {}
    st, out = json.load(open(path)), {}
    # a repetition whose distance is NaN had a non-finite draw (an infinite Laplace
    # sample, fixed in ssp.py since) and is dropped; a NaN T with a valid p (very
    # sparse fields: msw.test's T skips the finite() guard its p applies) makes z
    # undefined but leaves the power intact
    st = {k: [r for r in v if np.isfinite(r[2])] for k, v in st.items()}
    for key, arr in st.items():
        c, N = key.split("|N")
        null = st.get(f"null|N{N}", [])
        if c == "null" or len(null) < 2 or not arr:
            continue
        T0, T1 = [r[1] for r in null], [r[1] for r in arr]
        zz = z(T1, T0) if np.all(np.isfinite(T1 + T0)) else float("nan")
        out[(c, int(N))] = (zz, np.mean([r[0] <= 0.05 for r in arr]),
                            np.mean([r[0] <= 0.05 for r in null]), len(arr), len(null))
    return out


if __name__ == "__main__":
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))["near"]
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    detail = []
    for dom in ["d1", "d2"]:
        mine = ours("near", dom)
        for c, f in PANELS:
            with open(os.path.join(HERE, "data", f"ssp_{dom}_{f}.csv"), "w") as fh:
                fh.write("n," + ",".join(COLS[dom]) + ",ours\n")
                for N in NS[dom]:
                    b, b0 = base[dom][c][str(N)], base[dom]["null"][str(N)]
                    vals = []
                    for m in COLS[dom]:
                        v = abs(z(b[m], b0[m]))
                        vals.append(f"{max(0.0 if m == 'coverage' and v > 1e4 else v, 0.05):.4g}")
                    o = mine.get((c, N))
                    vals.append(f"{max(abs(o[0]), 0.05):.4g}" if o and np.isfinite(o[0]) else "nan")
                    fh.write(f"{N}," + ",".join(vals) + "\n")
        detail += [[dom, c, N, f"{v[0]:.3f}", f"{v[1]:.3f}", f"{v[2]:.3f}", v[3], v[4]]
                   for (c, N), v in sorted(mine.items(), key=lambda t: (t[0][0], t[0][1]))]
    with open(os.path.join(HERE, "data", "ours_detail.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dim", "contrast", "N", "z", "exact_power", "null_size", "reps_alt", "reps_null"])
        w.writerows(detail)
    print(open(os.path.join(HERE, "data", "ours_detail.csv")).read())
