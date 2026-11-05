"""Table 2 from results/: N*, the fewest images per set at which each method
reaches power 0.8 and keeps it at every larger rung.  Writes table.csv/.tex.
'>' means the method never reaches power 0.8 on the ladder.  The baselines ran
from 16 images, MSW from 8."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import CAPS, LADDER, LADDER_EPS, ROWS, key
from draws import NAMES
from protocol import Store, n_star, rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw", "pswd", "spec", "ps", "ps_star", "msw"]
HEAD = ["FID", "KID", "CMMD", "DINOv2", "C2ST", "coverage", "SW", "CSW", "pSWD", "spec.", "P--S",
        "P--S$^\\star$", "MSW"]


def results():
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))
    out = {}
    for row in ROWS:
        rungs = LADDER + ([CAPS[row]] if row in CAPS else [])
        pw = base[row]["power"]
        r = {m: n_star(rungs, lambda N, m=m: pw[m].get(str(N))) for m in COLS[:-1]}
        st = Store(os.path.join(HERE, "results", f"msw_{row}.json"))
        cell = lambda N: f"{row}|N{N}|eps{key(LADDER_EPS[row])}"
        r["msw"] = n_star(rungs, lambda N: rejection_rate([p for p, _ in st.d[cell(N)]])
                          if cell(N) in st.d else None)
        r["msw_power8"] = rejection_rate([p for p, _ in st.d.get(cell(8), [[1, 0]])])
        out[row] = r
    return out


if __name__ == "__main__":
    res = results()
    fmt = lambda v: ">" if v is None else str(v)
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting"] + COLS + ["msw_power_at_8"])
        for row in ROWS:
            w.writerow([row] + [fmt(res[row][m]) for m in COLS] + [f"{res[row]['msw_power8']:.2f}"])
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(COLS) + "}\n\\toprule\nSetting & " + " & ".join(HEAD)
                + " \\\\\n\\midrule\n")
        for row in ROWS:
            vals = [res[row][m] for m in COLS]
            best = min(v for v in vals if v is not None)
            cells = [f"$\\mathbf{{{fmt(v)}}}$" if v == best else ("$>$" if v is None else fmt(v)) for v in vals]
            f.write(f"{NAMES[row]} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())
