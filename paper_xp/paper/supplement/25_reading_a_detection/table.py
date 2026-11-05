"""Reading-a-detection table from results/: msw.test's per-level transport z by
slice family (level 0 = finest), mean over the 12 repetitions, for each contrast
and for the null.  Bold: the largest cell of each contrast.
Writes table.csv and table.tex.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAMS = ["per-channel DCT", "opponent luminance", "opponent chroma"]
ROWS = [("noise", "noise $0.0025$"), ("shading", "shading $0.08$"), ("null", "null")]

if __name__ == "__main__":
    p = os.path.join(HERE, "results", "msw.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    res = {}
    for k, _ in ROWS:
        if d.get(k):
            res[k] = {fam: np.mean([r["per_level"][fam] for r in d[k]], 0) for fam in FAMS}
    nl = max((len(v[FAMS[0]]) for v in res.values()), default=4)
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "family"] + [f"level{l}" for l in range(nl)] + ["power", "reps"])
        for k, _ in ROWS:
            if k not in res:
                continue
            pw = float(np.mean([r["p"] <= 0.05 for r in d[k]]))
            for fam in FAMS:
                w.writerow([k, fam] + [f"{v:.2f}" for v in res[k][fam]] + [f"{pw:.2f}", len(d[k])])
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "c" * nl + "}\n\\toprule\n & "
                + " & ".join(f"level {l}" for l in range(nl)) + " \\\\\n\\midrule\n")
        for k, label in ROWS:
            if k not in res:
                continue
            top = max(abs(v) for fam in FAMS for v in res[k][fam])
            for fam in FAMS:
                cells = [f"$\\mathbf{{{v:.1f}}}$" if abs(v) == top and k != "null" else f"${v:.1f}$"
                         for v in res[k][fam]]
                f.write(f"{label}, {fam} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())
