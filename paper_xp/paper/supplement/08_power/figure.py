"""Figure data from results/: power against images per set, one CSV per panel
(data/power_<setting>.csv).  Baseline columns are the archived exp26 powers
(threshold mean + 1.645 sd of 20 null reps); ours is the rejection rate of
msw.test at 0.05 with its Wilson interval, plus its empirical null size.
Missing MSW results give nan (no curve).
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from protocol import ALPHA, Store, rejection_rate, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = ["fields_slope", "celeba_blur"]
NS = [25, 50, 100, 200, 500, 1000]
BASE = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw"]

if __name__ == "__main__":
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))["power"]
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    for s in PANELS:
        path = os.path.join(HERE, "results", f"msw_{s}.json")
        st = Store(path).d if os.path.exists(path) else {}
        head = ["n"] + BASE + ["ours", "ours_lo", "ours_hi", "ours_reps", "ours_size"]
        with open(os.path.join(HERE, "data", f"power_{s}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            for N in NS:
                row = [N] + [base[s][m][str(N)] for m in BASE]
                p = [r[0] for r in st.get(f"{s}|N{N}|alt", [])]
                if not p:
                    row += ["nan", "nan", "nan", 0, "nan"]
                else:
                    lo, hi = wilson(sum(x <= ALPHA for x in p), len(p))
                    p0 = [r[0] for r in st.get(f"{s}|N{N}|null", [])]
                    size = f"{rejection_rate(p0):.4f}" if p0 else "nan"
                    row += [f"{rejection_rate(p):.4f}", f"{lo:.4f}", f"{hi:.4f}", len(p), size]
                w.writerow(row)
        print(open(os.path.join(HERE, "data", f"power_{s}.csv")).read())
