"""Representation-or-inference table from results/: eps_0.8 (smallest rung rejected
at 0.05 in >= 80% of the repetitions, N = 500) of the P-S vector inside MSW's
machinery (plain and per-coordinate standardised) and of MSW's transport rows on
the same draws.  Writes table.csv and table.tex.
"""
import csv
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import Store, eps_08, rejection_rate
from run import GRIDS

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"fields_slope": "fields, slope", "fields_marginal": "fields, marginal",
         "celeba_noise": "CelebA, noise", "celeba_shading": "CelebA, shading",
         "cifar_checker": "CIFAR, checker"}
COLS = ["ps_plain", "ps_std", "msw_T"]


def fmt(e, row):
    if e is None:
        return "--"
    if isinstance(e, str):
        return e
    if row == "fields_marginal":
        f = Fraction(e).limit_denominator(1000)
        return f"{f.numerator}/{f.denominator}"
    return f"{e:g}"


def results():
    out = {}
    for row, grid in GRIDS.items():
        p = os.path.join(HERE, "results", f"{row}.json")
        if not os.path.exists(p):
            continue
        st = Store(p)
        r = {}
        for i, m in enumerate(COLS):
            pw = lambda e, i=i: (rejection_rate([x[i] for x in st.d[c]])
                                 if (c := f"eps{e:.6g}") in st.d and st.d[c] else None)
            e8 = eps_08(grid, pw)
            done = all(pw(e) is not None for e in grid)
            r[m] = e8 if e8 is not None else (f">{fmt(grid[-1], row)}" if done else None)
        out[row] = r
    return out


def ratio(r):
    """eps_0.8 of P-S (standardised) over MSW transport; '>x' when P-S never reached 0.8."""
    a, b = r["ps_std"], r["msw_T"]
    if not isinstance(b, float):
        return None
    if isinstance(a, float):
        return f"{a / b:.2g}"
    if isinstance(a, str) and a.startswith(">"):
        top = float(Fraction(a[1:]))
        return f">{top / b:.2g}"
    return None


if __name__ == "__main__":
    res = results()
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting"] + COLS + ["ratio_std_over_T"])
        for row, r in res.items():
            w.writerow([row] + [fmt(r[m], row) for m in COLS] + [ratio(r) or ""])
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\nsetting & P--S, plain & P--S, standardised"
                " & MSW transport only & ratio \\\\\n\\midrule\n")
        for row, r in res.items():
            f.write(f"{NAMES[row]} & " + " & ".join(f"${fmt(r[m], row)}$" for m in COLS)
                    + (f" & ${ratio(r)}\\times$" if ratio(r) else " & ---") + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())
