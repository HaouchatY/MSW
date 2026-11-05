"""Table 1 from results/: eps_0.8, the smallest perturbation detected with power
>= 0.8 at N = 500, for every method.  Writes table.csv and table.tex.

Baselines detect above the 95th percentile of their own null (coverage, which
falls under the alternative, below the 5th); MSW when msw.test rejects at 0.05.
"""
import csv
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import BASE, ROWS, grid, key
from draws import NAMES
from protocol import Store, eps_08, rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw", "pswd", "spec", "ps", "ps_star", "msw"]
HEAD = ["FID", "KID", "CMMD", "DINOv2", "C2ST", "coverage", "SW", "CSW", "pSWD", "spec.", "P--S",
        "P--S$^\\star$", "MSW"]


def fmt(e, row):
    if e is None:
        return "--"
    if row == "fields_marginal":
        return f"1/{round(1 / e)}"
    return f"{e:g}"


def results():
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))
    out = {}
    for row in ROWS:
        pw = base[row]["power"]
        r = {m: eps_08(BASE[row], lambda e, m=m: pw[m].get(key(e))) for m in COLS[:-1]}
        st = Store(os.path.join(HERE, "results", f"msw_{row}.json"))
        r["msw"] = eps_08(grid(row), lambda e: (rejection_rate([p for p, _ in st.d[c]])
                                                if (c := f"{row}|N500|eps{key(e)}") in st.d else None))
        r["msw_size"] = rejection_rate([p for p, _ in st.d.get(f"{row}|N500|null", [[1, 0]])])
        out[row] = r
    return out


if __name__ == "__main__":
    res = results()
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting"] + COLS + ["msw_size"])
        for row in ROWS:
            w.writerow([row] + [fmt(res[row][m], row) for m in COLS] + [f"{res[row]['msw_size']:.3f}"])
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(COLS) + "}\n\\toprule\nSetting ($N{=}500$) & "
                + " & ".join(HEAD) + " \\\\\n\\midrule\n")
        for row in ROWS:
            vals = [res[row][m] for m in COLS]
            best = min(v for v in vals if v is not None)
            cells = [f"$\\mathbf{{{fmt(v, row)}}}$" if v == best else fmt(v, row) for v in vals]
            f.write(f"{NAMES[row]} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())
