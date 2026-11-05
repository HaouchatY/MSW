"""In-the-wild table from results/: deflection z at N = 16/64/256 for the baselines
(archived, 12 repetitions) and for MSW's T (200 repetitions), certified at
z >= 2.4865 (bold).  MSW's exact-test power and size go to table.csv.
Writes table.csv and table.tex.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
ZC = 2.4865
NS = [16, 64, 256]
TASKS = {"resize": "bicubic vs.\\ bilinear", "jpeg": "JPEG $q{=}95$ vs.\\ $q{=}90$"}
SHOWN = ["fid", "kid", "cmmd", "dino", "c2st"]
HEAD = ["FID", "KID", "CMMD", "DINOv2", "C2ST", "MSW (ours)"]


def ours(task, N):
    p = os.path.join(HERE, "results", f"msw_{task}.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    a, n = d.get(f"{task}|N{N}|alt", []), d.get(f"{task}|N{N}|null", [])
    if len(a) < 2 or len(n) < 2:
        return None
    Ta, Tn = np.array([t for _, t in a]), np.array([t for _, t in n])
    return {"z": float((Ta.mean() - Tn.mean()) / Tn.std(ddof=1)),
            "power": rejection_rate([p for p, _ in a]), "size": rejection_rate([p for p, _ in n]),
            "reps": len(a)}


if __name__ == "__main__":
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "N"] + base["metrics"] + ["msw_z", "msw_power", "msw_size", "msw_reps"])
        for t in TASKS:
            for N in NS:
                o = ours(t, N) or {}
                w.writerow([t, N] + [f"{base['z'][f'{t}|N{N}'][m]:+.2f}" for m in base["metrics"]]
                           + [f"{o[k]:.3f}" if k in o else "" for k in ("z", "power", "size")]
                           + [o.get("reps", 0)])
    cell = lambda v: f"\\mathbf{{{v:+.1f}}}" if v >= ZC else f"{v:+.1f}"
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(HEAD) + "}\n\\toprule\nContrast & "
                + " & ".join(HEAD) + " \\\\\n\\midrule\n")
        for t, label in TASKS.items():
            cells = ["$" + "/".join(cell(base["z"][f"{t}|N{N}"][m]) for N in NS) + "$" for m in SHOWN]
            o = [ours(t, N) for N in NS]
            cells.append("$" + "/".join(cell(x["z"]) if x else "\\cdot" for x in o) + "$")
            f.write(f"{label} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())
