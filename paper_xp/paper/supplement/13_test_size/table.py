"""Size of msw.test from results/: per draw family and N, the rejection rate at
alpha = 0.05 over the null repetitions with its 95% Wilson interval, then the
pooled rate over every family and N.  Writes table.csv and table.tex; missing
results are skipped.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import ALPHA, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MSW_XP_OUT", os.path.join(HERE, "results"))
FAMILIES = [("fields", "Gaussian fields"), ("t3", "t3 fields"), ("celeba32", "CelebA, 32 px crops"),
            ("celeba64", "CelebA, 64 px"), ("cifar", "CIFAR-10")]


def row(name, N, k, n):
    lo, hi = wilson(k, n)
    return [name, N, n, k, f"{k / n:.4f}", f"{lo:.4f}", f"{hi:.4f}"]


if __name__ == "__main__":
    rows, K, n_all = [], 0, 0
    for fam, name in FAMILIES:
        p = os.path.join(RES, f"msw_{fam}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for cell in sorted(d, key=lambda c: int(c.split("|")[1][1:])):
            ps = [r[0] for r in d[cell]]
            k = sum(x <= ALPHA for x in ps)
            rows.append(row(name, int(cell.split("|")[1][1:]), k, len(ps)))
            K, n_all = K + k, n_all + len(ps)
    if not rows:
        sys.exit("no results")
    rows.append(row("pooled", "all", K, n_all))
    head = ["family", "N", "reps", "rejections", "rate", "wilson_lo", "wilson_hi"]
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrc}\n\\toprule\nFamily & $N$ & reps & rejections & "
                "size [95\\% CI] \\\\\n\\midrule\n")
        for r in rows:
            if r[0] == "pooled":
                f.write("\\midrule\n")
            f.write(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} [{r[5]}, {r[6]}] \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())
