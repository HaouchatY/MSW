"""Table tab:time from results/times.json: median wall-clock of one evaluation
(one pair of sets) per metric and N.  Writes table.csv and table.tex; cells
not yet run print as --.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = [("fid", "FID"), ("kid", "KID"), ("cmmd", "CMMD"), ("dino", "DINOv2"), ("c2st", "C2ST"),
        ("coverage", "coverage"), ("sw", "SW"), ("csw", "CSW"), ("pswd", "pSWD"),
        ("spec", "spectrum"), ("ps", "P--S"), ("msw_distance", "MSW distance (ours)"),
        ("msw_test", "MSW test (ours)")]
NS = [200, 500, 1000]


def fmt(t):
    if t is None:
        return "--"
    return f"{t:#.3g}" if t >= 0.1 else f"{t:.2g}" if t >= 0.01 else f"{t:.1g}"


if __name__ == "__main__":
    path = os.path.join(HERE, "results", "times.json")
    d = json.load(open(path)) if os.path.exists(path) else {}
    med = {(m, N): (float(np.median(d[c])) if (c := f"{m}|N{N}") in d and d[c] else None)
           for m, _ in ROWS for N in NS}
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + [f"N{N}_median_s" for N in NS])
        for m, _ in ROWS:
            w.writerow([m] + ["" if med[m, N] is None else f"{med[m, N]:.5g}" for N in NS])
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\nMetric & $N{=}200$ & $N{=}500$ & $N{=}1000$ \\\\\n"
                "\\midrule\n")
        for m, name in ROWS:
            if m == "msw_distance":
                f.write("\\midrule\n")
            f.write(name + " & " + " & ".join(fmt(med[m, N]) + ("\\,s" if med[m, N] else "")
                                              for N in NS) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    if "env" in d:
        print(d["env"])
    print(open(os.path.join(HERE, "table.csv")).read())
