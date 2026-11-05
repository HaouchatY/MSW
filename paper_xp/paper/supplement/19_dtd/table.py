"""Held-out textures (DTD) from results/: power of every metric in every cell of
the four designs (table.csv), and a summary (table.tex): designs A and B as the
largest power over all their cells, design C as eps_0.8 at N = 50, design D as
power per image count.

Baselines detect above the 95th percentile of the cell's own 12 null
repetitions (coverage below the 5th); MSW when msw.test rejects at 0.05.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import Store, eps_08, rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw", "msw"]
HEAD = ["FID", "KID", "CMMD", "DINOv2", "C2ST", "coverage", "SW", "CSW", "MSW"]
GRID = {"noise": [0.005, 0.01, 0.02, 0.04, 0.08], "blur": [0.5, 0.75, 1.0, 1.5]}


def results():
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))["power"]
    ours = {}
    for d in "ABCD":
        p = os.path.join(HERE, "results", f"msw_{d}.json")
        if os.path.exists(p):
            ours.update({k: rejection_rate([x[0] for x in v]) for k, v in Store(p).d.items() if "|null|" not in k})
    return {c: dict(v, msw=ours.get(c)) for c, v in base.items()}


def fmt(v):
    return "n/a" if v is None else (v if isinstance(v, str) else f"{v:.2f}")


if __name__ == "__main__":
    res = results()
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell"] + COLS)
        for c, v in res.items():
            w.writerow([c] + [fmt(v[m]) for m in COLS])
    rows = []
    for d, label in [("A", "A: 47-category mix, max power"), ("B", "B: single category, max power")]:
        cells = [v for c, v in res.items() if c.startswith(d)]
        rows.append((label, [fmt(None if any(v[m] is None for v in cells) else max(v[m] for v in cells))
                             for m in COLS]))
    for k in ["noise", "blur"]:
        e = {m: eps_08(GRID[k], lambda x, m=m: res[f"C|woven|{k}{x:g}|N50"][m]) for m in COLS}
        rows.append((f"C: woven {k}, $\\varepsilon_{{0.8}}$ ($N{{=}}50$)",
                     ["n/a" if res[f"C|woven|{k}{GRID[k][0]:g}|N50"][m] is None else
                      "--" if e[m] is None else f"{e[m]:g}" for m in COLS]))
    for k, eps in [("noise", 0.02), ("blur", 1.0)]:
        for N in [8, 16, 25]:
            rows.append((f"D: woven {k} {eps:g}, $N{{=}}{N}$", [fmt(res[f"D|woven|{k}|N{N}"][m]) for m in COLS]))
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(COLS) + "}\n\\toprule\nDTD & " + " & ".join(HEAD)
                + " \\\\\n\\midrule\n")
        for label, cells in rows:
            f.write(f"{label} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.tex")).read())
