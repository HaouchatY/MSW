"""fig_edm2 from results/: data/edm_rungs.csv (z vs real), data/edm_pairs.csv
(adjacent snapshots, unpaired), data/edm_paired.csv (paired seeds), and the five
thumbnails data/edm_<pct>.png (image 2 of each snapshot, the same scene).

Every series is a deflection z = (mean alt - mean null) / sd null against its
null cell, signed so that positive means "further from the null in the
direction of the alternative" (coverage falls under the alternative, so its
sign is flipped).  The log axis floors z at 0.06; a wrong-direction deflection
plots at the floor.  Ours: z of the msw.test statistic T, its rejection rate at
0.05 (power), and msw.distance through the same transform (rungs only).
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DATA
from protocol import rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MSW_XP_OUT", os.path.join(HERE, "results"))
OUT = os.path.join(HERE, "data")
PCT = [2, 6, 25, 50, 100]
PAIRS = list(zip(PCT[:-1], PCT[1:]))
M = ["fid", "kid", "cmmd", "c2st", "coverage", "sw", "csw"]
FLOOR = 0.06


def deflect(null, alt, sign=1.0):
    x, y = np.asarray(null, float), np.asarray(alt, float)
    return sign * (y.mean() - x.mean()) / x.std(ddof=1)


def fl(v):
    return "" if v is None else f"{max(v, FLOOR):.4g}"


def ours(cell):
    p = os.path.join(RES, f"msw_{cell}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def write(name, head, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(head)
        w.writerows(rows)


def thumbnails(idx=2, px=96):
    from PIL import Image
    import torch
    for p, t in zip(PCT, ["0033554", "0134217", "0536870", "1073741", "2147483"]):
        f = os.path.join(DATA, "sd_pools", f"edm_k{t}.pt")
        if not os.path.exists(f):
            return
        x = torch.load(f, mmap=True)[idx].permute(1, 2, 0).numpy()
        Image.fromarray((x * 255).round().astype(np.uint8)).resize((px, px), Image.LANCZOS) \
            .save(os.path.join(OUT, f"edm_{p}.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    B = json.load(open(os.path.join(RES, "baselines.json")))
    C = B["cells"]
    sg = {m: (-1.0 if m == "coverage" else 1.0) for m in M}

    def ours_cols(null, alt, with_dist):
        n, a = ours(null).get(null, []), ours(alt).get(alt, [])
        if len(n) < 2 or not a:
            return [""] * (3 if with_dist else 2)
        z = deflect([t for _, t in n], [t for _, t in a])
        cols = [fl(z), f"{rejection_rate([p for p, _ in a]):.3f}"]
        if with_dist:
            dn, da = ours(null).get(f"{null}|dist", []), ours(alt).get(f"{alt}|dist", [])
            cols.append(fl(deflect([d[0] for d in dn], [d[0] for d in da]))
                        if len(dn) > 1 and da else "")
        return cols

    rows = []
    for p in PCT:
        rows.append([p] + [fl(deflect(C["null"][m], C[f"rung_{p}"][m], sg[m])) for m in M]
                    + ours_cols("null", f"rung_{p}", True))
    write("edm_rungs.csv", ["pct"] + M + ["ours", "ours_power", "ours_dist"], rows)

    rows = []
    for i, (a, b) in enumerate(PAIRS, 1):
        rows.append([i] + [fl(deflect(C[f"self_{a}"][m], C[f"pair_{a}-{b}"][m], sg[m])) for m in M]
                    + ours_cols(f"self_{a}", f"pair_{a}-{b}", False))
    write("edm_pairs.csv", ["i"] + M + ["ours", "ours_power"], rows)

    pp = os.path.join(RES, "msw_paired.json")
    P = json.load(open(pp)) if os.path.exists(pp) else {}
    rows = []
    for i, (a, b) in enumerate(PAIRS, 1):
        k = f"{a}_{b}"
        rows.append([i, f"{a}:{b}", fl(P[k]["z"]) if k in P else "",
                     fl(B["paired"]["clip"][k]["z"]), fl(B["paired"]["incep"][k]["z"])])
    write("edm_paired.csv", ["i", "pair", "ours", "clip", "incep"], rows)
    thumbnails()


if __name__ == "__main__":
    main()
