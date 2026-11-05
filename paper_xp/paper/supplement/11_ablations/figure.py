"""Figures 'ablation' and 'depth' from results/: the deflection
z = (mean_alt T - mean_null T) / sd_null T of every series, floored at 0.055 for
the log axis, and per cell the exact-test power and null size (*_detail.csv).
Series whose results are missing are skipped.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import key
from protocol import rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MSW_XP_OUT", os.path.join(HERE, "results"))
OUT = os.path.join(HERE, "data")
FLOOR = 0.055
DELTAS = [0.04, 0.08, 0.16, 0.32]
GRIDS = {"celeba_noise": [0.00025, 0.0005, 0.00125, 0.0025],
         "cifar_checker": [0.0025, 0.005, 0.01, 0.02]}
FAMS = ["dct", "chroma", "pixel", "product", "CORE"]


def stats(T0, T1, p0=None, p1=None):
    T0, T1 = np.asarray(T0, float), np.asarray(T1, float)
    s0 = T0.std(ddof=1)
    return {"z": float((T1.mean() - T0.mean()) / s0), "mT0": T0.mean(), "sT0": s0,
            "mT1": T1.mean(), "sT1": T1.std(ddof=1), "n0": len(T0), "n1": len(T1),
            "power": rejection_rate(p1) if p1 is not None else float("nan"),
            "size": rejection_rate(p0) if p0 is not None else float("nan")}


def write(name, head, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


DHEAD = ["x", "series", "null_reps", "alt_reps", "mean_T_null", "sd_T_null", "mean_T_alt",
         "sd_T_alt", "z", "null_size_at_0.05", "exact_power_at_0.05"]


def drow(x, name, s):
    return [x, name, s["n0"], s["n1"]] + [f"{s[k]:.4f}" for k in ("mT0", "sT0", "mT1", "sT1", "z")] + \
        [f"{s['size']:.3f}", f"{s['power']:.3f}"]


def fl(z):
    return f"{max(z, FLOOR):.6g}"


def levels():
    path = os.path.join(RES, "levels.json")
    if not os.path.exists(path):
        return
    st = json.load(open(path))
    null = st.get(f"delta{key(0.0)}", [])
    if len(null) < 2:
        return
    nl = len(null[0]["lev_T"])
    ab, dl, de, det = [], [], [], []
    for d in DELTAS:
        alt = st.get(f"delta{key(d)}", [])
        if not alt:
            continue
        lev = [stats([r["lev_T"][l] for r in null], [r["lev_T"][l] for r in alt],
                     [r["lev_p"][l] for r in null], [r["lev_p"][l] for r in alt]) for l in range(nl)]
        core = stats([r["full:CORE"] for r in null], [r["full:CORE"] for r in alt],
                     [r["full:CORE:p"] for r in null], [r["full:CORE:p"] for r in alt])
        ab.append([d] + [fl(s["z"]) for s in lev + [core]])
        det += [drow(d, f"ablation L{l}", s) for l, s in enumerate(lev)] + [drow(d, "CORE", core)]
        pl = [stats([r["lev_z"][l] for r in null], [r["lev_z"][l] for r in alt]) for l in range(nl)]
        dl.append([d] + [fl(s["z"]) for s in pl])
        det += [drow(d, f"depth per-level L{l}", s) for l, s in enumerate(pl)]
        ee = []
        for lab, k in (("scaleaware", "full:b_T_ml"), ("onescale", "one:b_T_ml"),
                       ("default", "full:b_T_iv"), ("core", "full:CORE")):
            s = stats([r[k] for r in null], [r[k] for r in alt],
                      [r[k + ":p"] for r in null], [r[k + ":p"] for r in alt])
            ee.append(fl(s["z"]))
            det.append(drow(d, f"depth {lab} ({k})", s))
        de.append([d] + ee)
    lv = [f"L{l}" for l in range(nl)]
    write("abl_levels.csv", ["delta"] + lv + ["core"], ab)
    write("depth_perlevel.csv", ["delta"] + lv, dl)
    write("depth_endtoend.csv", ["delta", "scaleaware", "onescale", "default", "core"], de)
    write("levels_detail.csv", DHEAD, det)


def families():
    for panel, grid in GRIDS.items():
        path = os.path.join(RES, f"families_{panel}.json")
        if not os.path.exists(path):
            continue
        st = json.load(open(path))
        null = st.get(f"eps{key(0.0)}", [])
        if len(null) < 2:
            continue
        rows, det = [], []
        for e in grid:
            alt = st.get(f"eps{key(e)}", [])
            if not alt:
                continue
            s = {f: stats([r[f][0] for r in null], [r[f][0] for r in alt],
                          [r[f][1] for r in null], [r[f][1] for r in alt]) for f in FAMS}
            rows.append([e] + [fl(s[f]["z"]) for f in FAMS])
            det += [drow(e, f, s[f]) for f in FAMS]
        fn = {"celeba_noise": "abl_noise", "cifar_checker": "abl_checker"}[panel]
        write(f"{fn}.csv", ["eps", "dct", "chroma", "pixel", "product", "portfolio"], rows)
        write(f"{fn}_detail.csv", DHEAD, det)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    levels()
    families()
    for f in sorted(os.listdir(OUT)):
        if not f.endswith("_detail.csv"):
            print(f"== {f}\n" + open(os.path.join(OUT, f)).read())
