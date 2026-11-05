"""The t3 row with and without per-image standardization: eps_0.8 at N = 500
for every metric.  Writes table.csv and table.tex.

standardized   Table 1's row (main/02_detection_thresholds/results)
raw            MSW from results/msw_raw.json (msw.test rejects at 0.05);
               baselines from results/baselines_raw_rerun.json when it exists
               (95th percentile of their own nulls), otherwise the archived
               exp33 run (results/baselines.json: means only, so the original
               mean-z >= 2.4865 criterion; no DINOv2, pSWD, spec. or P-S there)
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import BASE, grid, key
from protocol import Store, eps_08, power, q95, rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "..", "main", "02_detection_thresholds", "results")
COLS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw", "pswd", "spec", "ps", "msw"]
HEAD = ["FID", "KID", "CMMD", "DINOv2", "C2ST", "coverage", "SW", "CSW", "pSWD", "spec.", "P--S", "MSW"]


def msw_eps(path, prefix):
    if not os.path.exists(path):
        return None
    st = Store(path)
    return eps_08(grid("t3_slope"), lambda e: (rejection_rate([p for p, _ in st.d[c]])
                                               if (c := f"{prefix}|N500|eps{key(e)}") in st.d else None))


def standardized():
    pw = json.load(open(os.path.join(MAIN, "baselines.json")))["t3_slope"]["power"]
    r = {m: eps_08(BASE["t3_slope"], lambda e, m=m: pw[m].get(key(e))) for m in COLS[:-1]}
    r["msw"] = msw_eps(os.path.join(MAIN, "msw_t3_slope.json"), "t3_slope")
    return r


def raw():
    rerun = os.path.join(HERE, "results", "baselines_raw_rerun.json")
    if os.path.exists(rerun):
        d = json.load(open(rerun))
        null = d["t3raw|N500|null"]
        r, src = {}, "rerun"
        for m in COLS[:-1]:
            v0 = [x[m] for x in null]
            lower = m == "coverage"                     # falls under the alternative
            thr = q95([-v for v in v0]) if lower else q95(v0)

            def pw_at(e, m=m, thr=thr, lower=lower):
                c = f"t3raw|N500|eps{key(e)}"
                if c not in d:
                    return None
                return power([-x[m] for x in d[c]] if lower else [x[m] for x in d[c]], thr)
            r[m] = eps_08(BASE["t3_slope"], pw_at)
    else:
        b = json.load(open(os.path.join(HERE, "results", "baselines.json")))
        r = {m: (float(b["eps_08_meanz"][m]) if b["eps_08_meanz"].get(m) else None) if m in b["z"]
             else "n/a" for m in COLS[:-1]}
        src = "exp33, mean-z"
    r["msw"] = msw_eps(os.path.join(HERE, "results", "msw_raw.json"), "t3raw")
    return r, src


def fmt(v):
    return v if isinstance(v, str) else ("--" if v is None else f"{v:g}")


if __name__ == "__main__":
    s = standardized()
    r, src = raw()
    if not os.path.exists(os.path.join(HERE, "results", "msw_raw.json")):
        r["msw"] = "n/a"
    rows = [("standardized", s), (f"raw ({src})", r)]
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inputs"] + COLS)
        for name, v in rows:
            w.writerow([name] + [fmt(v[m]) for m in COLS])
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(COLS) + "}\n\\toprule\n$t_3$ slope ($N{=}500$) & "
                + " & ".join(HEAD) + " \\\\\n\\midrule\n")
        for name, v in rows:
            nums = [v[m] for m in COLS if isinstance(v[m], float)]
            best = min(nums) if nums else None
            cells = [f"$\\mathbf{{{fmt(v[m])}}}$" if v[m] == best else fmt(v[m]) for m in COLS]
            f.write(f"{name.split(' (')[0]} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())
