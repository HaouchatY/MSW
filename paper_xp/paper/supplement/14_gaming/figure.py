"""Figure data from results/: writes data/gaming_z.csv, data/gaming_power.csv and
data/gaming_components.csv, and prints the share of each metric's signal the
attacks erase.

Every metric, ours included, is scored by its deflection z = (mean over the
attacked cell - mean over the null cell) / null sd (coverage, which falls under
the alternative, with the sign flipped); "erased" is 1 - z_matched / z_random.
MSW is also scored by its decision: the rejection rate of msw.test at 0.05 with
a 95% Wilson interval.  The per-component profile standardizes every CORE
component of msw.test by its null mean and sd.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import rejection_rate, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
RES, DATA = os.path.join(HERE, "results"), os.path.join(HERE, "data")
METRICS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw"]
ATTACKS = [("select", "selection", "random", "matched"),
           ("hbnoise", "HB from noise", "before", "after"),
           ("hbblur", "HB from blur", "before", "after")]


def z(v, null, sign=1):
    n = np.asarray(null, dtype=float)
    return sign * (np.mean(v) - n.mean()) / n.std(ddof=1)


def main():
    os.makedirs(DATA, exist_ok=True)
    base = json.load(open(os.path.join(RES, "baselines.json")))["values"]
    rows = []
    for m in METRICS:
        v, s = base[m], (-1 if m == "coverage" else 1)
        rows.append((m, z(v["select|random"], v["null"], s), z(v["select|matched"], v["null"], s)))
    path = os.path.join(RES, "scores.json")
    S = json.load(open(path)) if os.path.exists(path) else {}
    T = {k: [r["T"] for r in v] for k, v in S.items()}
    if {"null", "select|random", "select|matched"} <= set(T):
        rows.append(("ours", z(T["select|random"], T["null"]), z(T["select|matched"], T["null"])))
    with open(os.path.join(DATA, "gaming_z.csv"), "w") as f:
        f.write("i,metric,rand,gamed\n")
        for i, (m, a, b) in enumerate(rows, 1):
            f.write(f"{i},{m},{a:.4g},{b:.4g}\n")
            print(f"selection  {m:9s} z {a:8.4g} -> {b:8.4g}   erased {100 * (1 - b / a):+6.1f}%")
    if "null" not in S:
        return
    size = rejection_rate([r["p"] for r in S["null"]])
    with open(os.path.join(DATA, "gaming_power.csv"), "w") as f:
        f.write("i,label,size,before,before_ep,before_em,after,after_ep,after_em,"
                "z_before,z_after,erased,worst_p_after,n\n")
        for i, (tag, lab, b, a) in enumerate(ATTACKS, 1):
            cb, ca = S.get(f"{tag}|{b}"), S.get(f"{tag}|{a}")
            if not cb or not ca:
                continue
            c = []
            for cell in (cb, ca):
                k, n = sum(r["p"] <= 0.05 for r in cell), len(cell)
                lo, hi = wilson(k, n)
                c += [f"{k / n:.4f}", f"{hi - k / n:.4f}", f"{k / n - lo:.4f}"]
            zb, za = z([r["T"] for r in cb], T["null"]), z([r["T"] for r in ca], T["null"])
            worst = max(r["p"] for r in ca)
            f.write(f"{i},{lab},{size:.4f}," + ",".join(c)
                    + f",{zb:.4g},{za:.4g},{100 * (1 - za / zb):.2f},{worst:.4g},{len(ca)}\n")
            print(f"{lab:14s} rejection {c[0]} -> {c[3]} (size {size:.3f})  z {zb:.4g} -> {za:.4g}"
                  f"  erased {100 * (1 - za / zb):+.1f}%  worst p after {worst:.3g}")
    if "hbnoise|after" in S:                     # which components the HB attack reaches
        names = list(S["null"][0]["comp"])
        C = {k: np.array([[r["comp"][c] for c in names] for r in S[k]])
             for k in ("null", "hbnoise|before", "hbnoise|after", "hbblur|before", "hbblur|after") if k in S}
        mu, sd = C["null"].mean(0), np.maximum(C["null"].std(0, ddof=1), 1e-30)
        prof = {k: ((v - mu) / sd).mean(0) for k, v in C.items() if k != "null"}
        with open(os.path.join(DATA, "gaming_components.csv"), "w") as f:
            f.write("component," + ",".join(k.replace("|", "_") for k in prof) + ",surv_noise\n")
            for j, c in enumerate(names):
                sv = prof["hbnoise|after"][j] / prof["hbnoise|before"][j]
                f.write(f"{c}," + ",".join(f"{prof[k][j]:.4g}" for k in prof) + f",{sv:.4g}\n")


if __name__ == "__main__":
    main()
