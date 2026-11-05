"""Global color cast table from results/: per amplitude, every baseline's deflection
z (12 repetitions; certified at z >= 2.4865) and MSW's power with and without the
raw-pixel row (exact test at 0.05).  Last row: MSW's size on the null.  Also checks that, without the pixel row, every
repetition returns a bit-identical p at every amplitude.
Writes table.csv and table.tex.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
ZC = 2.4865
BASE = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw"]
HEAD = ["FID", "KID", "CMMD", "DINOv2", "C2ST", "coverage", "SW", "CSW", "MSW", "MSW$-$PX"]


def load(name):
    p = os.path.join(HERE, "results", name)
    return json.load(open(p)) if os.path.exists(p) else {}


if __name__ == "__main__":
    base, ours = load("baselines.json"), load("msw.json")
    eps = sorted({float(e) for e in base.get("z", {})} |
                 {float(k[3:]) for k in ours if k.startswith("eps")})
    rows = []
    for e in eps:
        z = base.get("z", {}).get(f"{e:g}", {})
        arr = ours.get(f"eps{e:g}", [])
        pw = {k: rejection_rate([r[k][0] for r in arr]) if arr else None for k in ("CORE", "CORE-PX")}
        rows.append((f"{e:g}", [z.get(m) for m in BASE], [pw["CORE"], pw["CORE-PX"]], len(arr)))
    null = ours.get("null", [])
    size = [rejection_rate([r[k][0] for r in null]) if null else None for k in ("CORE", "CORE-PX")]
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps"] + [f"z_{m}" for m in BASE] + ["power_msw", "power_msw_nopx", "msw_reps"])
        for e, zs, pw, n in rows:
            w.writerow([e] + ["" if v is None else f"{v:+.2f}" for v in zs]
                       + ["" if v is None else f"{v:.2f}" for v in pw] + [n])
        w.writerow(["null"] + [""] * len(BASE) + ["" if v is None else f"{v:.3f}" for v in size]
                   + [len(null)])
    zc = lambda v: "" if v is None else (f"$\\mathbf{{{v:+.1f}}}$" if v >= ZC else f"${v:+.1f}$")
    pc = lambda v: "" if v is None else (f"$\\mathbf{{{v:.2f}}}$" if v >= 0.8 else f"${v:.2f}$")
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(HEAD) + "}\n\\toprule\n$\\varepsilon$ & "
                + " & ".join(HEAD) + " \\\\\n\\midrule\n")
        for e, zs, pw, _ in rows:
            f.write(f"${e}$ & " + " & ".join([zc(v) for v in zs] + [pc(v) for v in pw]) + " \\\\\n")
        f.write("\\midrule\nnull (size) & " + " & ".join([""] * len(BASE) + [pc(v) for v in size])
                + " \\\\\n\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())
    # the bank's blindness: without the pixel row, repetition r should give the same p at
    # every eps.  It does up to float32 rounding (filters zero-mean to ~1e-6, x + eps c
    # rounded): report the fraction of bit-identical repetitions and the largest move.
    cols = [[r["CORE-PX"][0] for r in ours[k]] for k in ours if k.startswith("eps")]
    n = min(map(len, cols), default=0)
    if n:
        same = sum(len({c[i] for c in cols}) == 1 for i in range(n))
        dmax = max(max(c[i] for c in cols) - min(c[i] for c in cols) for i in range(n))
        print(f"MSW without the pixel row, {len(cols)} amplitudes x {n} repetitions: p bit-identical "
              f"across amplitudes in {same}/{n} repetitions; largest spread of p {dmax:.2e}")
