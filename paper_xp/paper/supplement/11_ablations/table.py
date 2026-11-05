"""Table 'banks' and the component-deletion numbers from results/.

banks  eps_0.8 per spatial bank: the smallest rung whose power, and that of every
       larger rung, is >= 0.8 (exact test at alpha = 0.05, 24 reps per rung).
       Writes table.csv, table.tex and table_power.csv (every rung).
loo    eps_0.8 as in Table 1 (first rung with power >= 0.8) of CORE, of the
       transport rows alone, and of CORE minus each group, with the factor lost
       relative to CORE.  Writes table_loo.csv.
Missing results are skipped.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import ROWS, grid, key
from draws import NAMES
from protocol import eps_08, n_star, rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MSW_XP_OUT", os.path.join(HERE, "results"))
BANKS = ["dct", "gauss", "ortho", "pca"]
HEAD = ["DCT", "random", "orthonormal", "patch PCA"]
GRID = {"fields_slope": [0.002, 0.004, 0.008, 0.016, 0.032, 0.064],
        "celeba_noise": [0.000125, 0.00025, 0.0005, 0.00125, 0.0025, 0.005],
        "celeba_blur": [0.033, 0.05, 0.067, 0.1, 0.15, 0.22, 0.33],
        "celeba_shading": [0.005, 0.01, 0.02, 0.04, 0.08],
        "cifar_checker": [0.00015625, 0.0003125, 0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02]}
LOO = ["CORE", "T_all", "CORE-T", "CORE-TL", "CORE-PX", "CORE-J"]


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(e):
    return "--" if e is None else f"{e:g}"


def banks():
    rows, power = [], []
    for s, g in GRID.items():
        st = load(f"banks_{s}.json")
        if st is None:
            continue
        for port in (["CORE", "CORE-J"] if s == "cifar_checker" else ["CORE"]):
            pw = {}
            for b in BANKS:
                for e in g:
                    c = st.get(f"{b}|eps{key(e)}")
                    if c:
                        pw[b, e] = rejection_rate([r[port] for r in c])
                        power.append([s, port, b, f"{e:g}", len(c), f"{pw[b, e]:.3f}"])
            name = NAMES[s] + ("" if port == "CORE" else ", product slices removed")
            rows.append([name] + [n_star(g, lambda e, b=b: pw.get((b, e))) for b in BANKS])
    if not rows:
        return
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting"] + BANKS)
        w.writerows([[r[0]] + [fmt(v) for v in r[1:]] for r in rows])
    with open(os.path.join(HERE, "table_power.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "portfolio", "bank", "eps", "reps", "power"])
        w.writerows(power)
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\nSetting & " + " & ".join(HEAD)
                + " \\\\\n\\midrule\n")
        for r in rows:
            vals = [v for v in r[1:] if v is not None]
            best = min(vals) if vals else None
            win = best is not None and any(v is not None and v > best for v in r[1:])
            name = ("\\quad the same, product slices removed" if r[0].endswith("removed") else r[0])
            f.write(name + " & "
                    + " & ".join(f"$\\mathbf{{{fmt(v)}}}$" if win and v == best else fmt(v)
                                 for v in r[1:]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())


def loo():
    out = []
    for row in ROWS:
        st = load(f"loo_{row}.json")
        if st is None:
            continue
        e = {k: eps_08(grid(row), lambda x, k=k: (rejection_rate([r[k] for r in st[c]])
                                                   if (c := f"eps{key(x)}") in st else None))
             for k in LOO}
        out.append([row] + [fmt(e[k]) for k in LOO] +
                   [f"{e[k] / e['CORE']:.3g}" if e[k] and e["CORE"] else "--" for k in LOO[1:]])
    if not out:
        return
    with open(os.path.join(HERE, "table_loo.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting"] + [f"eps_{k}" for k in LOO] + [f"factor_{k}" for k in LOO[1:]])
        w.writerows(out)
    print(open(os.path.join(HERE, "table_loo.csv")).read())


if __name__ == "__main__":
    banks()
    loo()
