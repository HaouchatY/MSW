"""fig:camouflage from results/: deflection z = (mean_alt - mean_null) / sd_null
at N = 500 per df for every method (coverage sign-flipped: it falls under the
alternative; values below 0.055 floored for the log axis), the frontier ladder
(power, erasure of the statistic's excess over the null mean relative to df = 4),
and the 5th / 95th contrast-percentile tiles.  Writes data/*.csv, data/*.png.

    python figure.py [--res results]
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.stats import chi2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import rejection_rate, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
DFS = ["4", "8", "16", "48"]
DEEP = ["fid", "kid", "cmmd", "c2st", "coverage", "sw", "csw"]
CLASSICAL = ["spec", "ps", "pswd"]
FLOOR = 0.055


def zs(v0, v1):
    v0 = np.asarray(v0, float)
    return (np.mean(v1) - v0.mean()) / v0.std(ddof=1)


def baselines():
    b = json.load(open(os.path.join(HERE, "results", "baselines.json")))
    det, out = b["detection"]["N500"], {}
    for df in DFS:
        r = {m: (-1 if m == "coverage" else 1) * (det[df][m][0] - det["null"][m][0]) / det["null"][m][1]
             for m in DEEP}
        r["dino"] = zs(b["dino"]["null"], b["dino"][df])
        for m in CLASSICAL:
            c = b["classical"]
            r[m] = zs([x[m] for x in c["null"]], [x[m] for x in c[df]]) if df in c else float("nan")
        out[df] = r
    return out


def ours(res):
    path = os.path.join(res, "msw_camo.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if "null" not in d or len(d["null"]) < 2:
        return None
    T0 = np.array([x[1] for x in d["null"]])
    rows = {}
    for k in sorted((k for k in d if k.startswith("df")), key=lambda k: float(k[2:])):
        p, T = np.array([x[0] for x in d[k]]), np.array([x[1] for x in d[k]])
        rows[k[2:]] = dict(reps=len(T), meanT=T.mean(), sdT=T.std(ddof=1), z=zs(T0, T),
                           power=rejection_rate(p), k=int((p <= 0.05).sum()),
                           dist=float(np.mean([x[2] for x in d[k]])))
    ref = rows.get("4", {}).get("meanT")
    for r in rows.values():
        r["erased"] = 1 - (r["meanT"] - T0.mean()) / (ref - T0.mean()) if ref else float("nan")
    return dict(T0=T0.mean(), sd0=T0.std(ddof=1), n0=len(T0),
                size=rejection_rate([x[0] for x in d["null"]]), rows=rows)


def tiles(seed=3):
    """One field at the 5th and 95th percentile of nu for each df (Gauss: nu = 1)."""
    from draws import S0, sample_field
    g = torch.Generator(device=S0.device).manual_seed(seed)
    x = sample_field(S0, 1, generator=g)[0, 0].cpu().numpy()
    for name, df in [("gauss", None), ("t48", 48), ("t16", 16), ("t8", 8), ("t4", 4)]:
        for tag, q in [("a", 0.95), ("b", 0.05)]:          # chi2 upper tail = nu lower tail
            nu = 1.0 if df is None else math.sqrt((df - 2) / chi2.ppf(q, df))
            im = np.clip((nu * x + 3) / 6, 0, 1)
            Image.fromarray((im * 255).round().astype(np.uint8)).resize((128, 128), Image.NEAREST) \
                .save(os.path.join(OUT, f"camo_{name}_{tag}.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    B, O = baselines(), ours(a.res)
    cols = DEEP[:3] + ["dino"] + DEEP[3:] + CLASSICAL + (["ours"] if O else [])
    with open(os.path.join(OUT, "camo_n500.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["df"] + cols)
        for df in DFS:
            v = [O["rows"].get(df, {}).get("z", float("nan")) if m == "ours" else B[df][m] for m in cols]
            w.writerow([df] + ["nan" if math.isnan(x) else f"{max(x, FLOOR):.6g}" for x in v])
    detail = os.path.join(OUT, "camo_ours_detail.csv")
    if not O and os.path.exists(detail):
        os.remove(detail)
    if O:
        with open(detail, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["df", "null_reps", "mean_T_null", "sd_T_null", "null_size", "alt_reps",
                        "mean_T_alt", "sd_T_alt", "z", "power", "power_lo", "power_hi", "erased",
                        "mean_distance"])
            for df, r in O["rows"].items():
                lo, hi = wilson(r["k"], r["reps"])
                w.writerow([df, O["n0"], f"{O['T0']:.4f}", f"{O['sd0']:.4f}", f"{O['size']:.3f}",
                            r["reps"], f"{r['meanT']:.4f}", f"{r['sdT']:.4f}", f"{r['z']:.4f}",
                            f"{r['power']:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{r['erased']:.3f}",
                            f"{r['dist']:.5g}"])
                print(f"df {df:>5s}  z {r['z']:7.2f}  power {r['power']:.3f}  erased {100 * r['erased']:4.0f}%")
        dfs = sorted(O["rows"], key=float)
        cross = [(d0, d1) for d0, d1 in zip(dfs, dfs[1:])
                 if O["rows"][d0]["power"] >= 0.8 > O["rows"][d1]["power"]]
        if cross:
            d0, d1 = cross[0]
            p0, p1 = O["rows"][d0]["power"], O["rows"][d1]["power"]
            fr = math.exp(math.log(float(d0)) + (p0 - 0.8) / (p0 - p1) * math.log(float(d1) / float(d0)))
            print(f"power-0.8 frontier between df {d0} and {d1} (log-linear: df ~ {fr:.0f}); "
                  f"null size {O['size']:.3f} over {O['n0']} runs")
    tiles()
    print(open(os.path.join(OUT, "camo_n500.csv")).read())
