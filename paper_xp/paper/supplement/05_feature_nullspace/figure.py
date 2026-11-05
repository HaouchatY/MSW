"""Figure (feature nullspace) from results/: the deflection |z| = |E v_alt - E v_null| /
sd(v_null) of every baseline value v (results/baselines.json, 10 + 10 repetitions)
and of MSW's T (results/msw_*.json), plotted as max(|z|, 0.05) on the log axis.
Also data/ours_detail.csv: MSW's exact-test rejection rates at alpha = 0.05.

    python figure.py            # writes data/*.csv
    python figure.py --tiles    # also the four image tiles (needs imagenet256_ref.pt)
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
NS = [64, 128, 500, 2000]
METRICS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw"]
FLOOR = 0.05


def baselines():
    return json.load(open(os.path.join(HERE, "results", "baselines.json")))["cells"]


def ours():
    """cell -> (p array, T array), from every results/msw_*.json present."""
    out = {}
    for f in glob.glob(os.path.join(HERE, "results", "msw_*.json")):
        for k, v in json.load(open(f)).items():
            if v:
                a = np.asarray(v, dtype=float)
                out[k] = (a[:, 0], a[:, 1])
    return out


def zdef(cells, null, alt, m):
    if null not in cells or alt not in cells:
        return None
    v0, v1 = np.asarray(cells[null][m], float), np.asarray(cells[alt][m], float)
    return (v1.mean() - v0.mean()) / max(v0.std(ddof=1), 1e-30)


def plotted(z):
    return math.nan if z is None else max(abs(z), FLOOR)


def ours_z(o, null, alt):
    if null not in o or alt not in o:
        return None
    return zdef({null: {"T": o[null][1]}, alt: {"T": o[alt][1]}}, null, alt, "T")


def tiles():
    """pool[7] whole, and its central 64 px clean / at eps 0.02 / at eps 0.04."""
    import torch
    from PIL import Image
    sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
    from draws import DATA
    x = torch.load(os.path.join(DATA, "imagenet256_ref.pt"), map_location="cpu", mmap=True)[7]
    r = torch.arange(256)
    pat = ((-1.0) ** (r[:, None] + r[None, :])).view(1, 256, 256) * \
        torch.tensor([1.0, -1.0, 0.0]).view(3, 1, 1) / math.sqrt(2)

    def save(t, name):
        a = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(a).save(os.path.join(OUT, f"{name}.png"))
    save(x, "chk_clean")
    save(x[:, 96:160, 96:160], "chk_clean_zoom")
    save((x + 0.02 * pat)[:, 96:160, 96:160], "chk_e02_zoom")
    save((x + 0.04 * pat)[:, 96:160, 96:160], "chk_e04_zoom")


def main():
    os.makedirs(OUT, exist_ok=True)
    b, o = baselines(), ours()
    fmt = lambda v: "nan" if math.isnan(v) else "%.4g" % v
    with open(os.path.join(OUT, "killer_nullspace.csv"), "w") as f:
        f.write("n," + ",".join(METRICS) + ",ours\n")
        for N in NS:
            nk, ak = f"null|N{N}", f"chroma0.02|N{N}"
            vals = [plotted(zdef(b, nk, ak, m)) for m in METRICS] + [plotted(ours_z(o, nk, ak))]
            f.write(f"{N}," + ",".join(fmt(v) for v in vals) + "\n")
    with open(os.path.join(OUT, "killer_nullctrl.csv"), "w") as f:
        f.write("i,metric,chroma,blur\n")
        for i, m in enumerate(METRICS + ["ours"], 1):
            z = (lambda alt: ours_z(o, "null|N500", alt)) if m == "ours" else \
                (lambda alt: zdef(b, "null|N500", alt, m))
            f.write(f"{i},{m},{fmt(plotted(z('chroma0.02|N500')))},{fmt(plotted(z('blur0.5|N500')))}\n")
    rows = [(f"null|N{N}", f"chroma0.02|N{N}") for N in NS] + [("null|N500", "blur0.5|N500")]
    with open(os.path.join(OUT, "ours_detail.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alt_cell", "reps_null", "reps_alt", "z", "power_a05", "size_a05"])
        for nk, ak in rows:
            if nk in o and ak in o:
                w.writerow([ak, len(o[nk][0]), len(o[ak][0]), "%.4g" % ours_z(o, nk, ak),
                            "%.3f" % np.mean(o[ak][0] <= 0.05), "%.3f" % np.mean(o[nk][0] <= 0.05)])
    if "--tiles" in sys.argv:
        tiles()
    print(open(os.path.join(OUT, "killer_nullspace.csv")).read())
    print(open(os.path.join(OUT, "killer_nullctrl.csv")).read())


if __name__ == "__main__":
    main()
