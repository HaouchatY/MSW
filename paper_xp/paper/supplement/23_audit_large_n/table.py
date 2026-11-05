"""Large-N audit table from results/: per pipeline pair at N = 800, the baselines'
shift against the mean of the two same-pipeline nulls (archived) next to MSW's
exact-test p, statistic T and distance with its 95% interval.  table.csv holds
all 28 pairs and the 8 nulls; table.tex the five pairs of the paper.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHOWN = ["pil_bilinear|pil_nearest", "pil_lanczos|pil_nearest", "pil_bicubic|pil_bicubic_jpeg90",
         "pil_nearest|pil_box", "pil_lanczos|tv_bicubic_aa"]
LABEL = {"pil_bicubic": "bicubic", "pil_bilinear": "bilinear", "pil_lanczos": "lanczos",
         "pil_nearest": "nearest", "pil_box": "box", "tv_bicubic_aa": "torchvision AA",
         "tv_bicubic_noaa": "torchvision no-AA", "pil_bicubic_jpeg90": "JPEG\\,$q90$"}


def load(name):
    p = os.path.join(HERE, "results", name)
    return json.load(open(p)) if os.path.exists(p) else {}


if __name__ == "__main__":
    base, ours = load("baselines.json"), load("msw.json")
    keys = list(base["pairs"]) + [f"null|{v}" for v in base["null"]]
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "dfid", "dkid", "dcmmd", "ddino", "msw_p", "msw_T", "msw_d", "msw_lo", "msw_hi"])
        for k in keys:
            b = base["pairs"].get(k) or base["null"][k.split("|")[1]]
            o = ours.get(k, [None] * 5)
            w.writerow([k] + [f"{b[m]:.6g}" for m in ("fid", "kid", "cmmd", "dino")]
                       + ["" if v is None else f"{v:.6g}" for v in o])
    nulls = [ours[k][0] for k in ours if k.startswith("null|")]
    print(f"FID null spread (population std, 8 pipelines): "
          f"{np.std([v['fid'] for v in base['null'].values()]):.3f};  MSW nulls rejected at 0.05: "
          f"{sum(p <= 0.05 for p in nulls)}/{len(nulls)}")
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n& $\\Delta$FID & $\\Delta$DINOv2-FD & "
                "$\\Delta$CMMD & MSW $T$ & MSW $d$ \\\\\n\\midrule\n")
        for k in SHOWN:
            b, o = base["pairs"][k], ours.get(k)
            i, j = k.split("|")
            ms = ["", ""] if o is None else [f"${o[1]:.1f}$", f"${o[2]:+.4f}$"]
            f.write(f"{LABEL[i]} vs.\\ {LABEL[j]} & ${b['fid']:+.3f}$ & ${b['dino']:+.2f}$ & "
                    f"${b['cmmd']:+.3f}$ & " + " & ".join(ms) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())
