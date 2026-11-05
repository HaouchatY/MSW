"""Figure 3 data from results/: writes data/killer_key_v2.csv, killer_key_mid.csv,
calib_struct.csv, calib_null.csv and calib_summary.csv.

Left panel: each method's response to the seven points, normalised to its
response at the largest one; MSW is msw.distance normalised per repetition, mean
and standard error over 48 repetitions; the mid-band point is drawn separately.
Right panel: each method against the exact W2 over the 55 model pairs, scaled by
its least-squares slope on the structured pairs (identity, contrast, blur)."""
import csv
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
SEVEN = ["nyq025", "nyq050", "fine075", "low100", "low140", "con200", "con280"]


def key():
    k = json.load(open(os.path.join(HERE, "results", "key.json")))
    D = {p: np.array([r[0] for r in k[p]]) for p in SEVEN + ["mid120"]}
    norm = {p: D[p] / D["con280"] for p in D}
    base = list(csv.DictReader(open(os.path.join(HERE, "results", "baselines_key.csv"))))
    w = csv.writer(open(os.path.join(OUT, "killer_key_v2.csv"), "w", newline=""))
    cols = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw"]
    w.writerow(["w2"] + cols + ["ours", "ours_err", "truth", "spec", "pswd", "ps"])
    for p, b in zip(SEVEN, base):
        v = norm[p]
        w.writerow([b["w2"]] + [b[c] for c in cols] + [f"{v.mean():.4f}", f"{v.std(ddof=1) / len(v) ** .5:.4f}",
                   f"{k['w2'][p] / k['w2']['con280']:.4f}"] + [b[c] for c in ("spec", "pswd", "ps")])
    b, v = base[-1], norm["mid120"]
    w = csv.writer(open(os.path.join(OUT, "killer_key_mid.csv"), "w", newline=""))
    w.writerow(["w2", "ours", "ours_err", "truth", "spec", "pswd", "ps"])
    w.writerow([b["w2"], f"{v.mean():.4f}", f"{v.std(ddof=1) / len(v) ** .5:.4f}",
                f"{k['w2']['mid120'] / k['w2']['con280']:.4f}", b["spec"], b["pswd"], b["ps"]])


def calib():
    c = json.load(open(os.path.join(HERE, "results", "calib.json")))
    base = json.load(open(os.path.join(HERE, "results", "baselines_calib.json")))
    W, fam = np.array(c["w2"]), [m.rstrip("0123456789.") for m in c["models"]]
    iu = list(itertools.combinations(range(len(fam)), 2))
    x = np.array([W[i, j] for i, j in iu])
    S = {m: np.array(v) for m, v in base["pairs"].items()}
    S["msw"] = np.array([np.mean([r[f"{i},{j}"] for r in c["reps"]]) for i, j in iu])
    struct = np.array([k for k, (i, j) in enumerate(iu) if {fam[i], fam[j]} <= {"id", "contrast", "blur"}])
    exact = np.array([k for k, (i, j) in enumerate(iu) if "id" in (fam[i], fam[j]) or fam[i] == fam[j]])
    cols = ["fid", "kid", "cmmd", "dino", "spec", "pswd", "ps", "msw"]
    slope = {m: (x[struct] * S[m][struct]).sum() / (x[struct] ** 2).sum() for m in cols}
    for name, idx in (("calib_struct", struct), ("calib_null", np.setdiff1d(np.arange(len(x)), struct))):
        w = csv.writer(open(os.path.join(OUT, f"{name}.csv"), "w", newline=""))
        w.writerow(["w2"] + cols)
        for k in idx:
            w.writerow([f"{x[k]:.6f}"] + [f"{S[m][k] / slope[m]:.6f}" for m in cols])

    def r2(m, idx):
        s = (x[idx] * S[m][idx]).sum() / (x[idx] ** 2).sum()
        return 1 - ((S[m][idx] - s * x[idx]) ** 2).sum() / ((S[m][idx] - S[m][idx].mean()) ** 2).sum()
    rank = lambda v: np.argsort(np.argsort(v))
    w = csv.writer(open(os.path.join(OUT, "calib_summary.csv"), "w", newline=""))
    w.writerow(["method", "r2_struct", "r2_all", "r2_exact", "rho"])
    for m in cols:
        w.writerow([m, f"{r2(m, struct):.3f}", f"{r2(m, np.arange(len(x))):.3f}", f"{r2(m, exact):.3f}",
                    f"{np.corrcoef(rank(x), rank(S[m]))[0, 1]:.3f}"])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    key()
    calib()
