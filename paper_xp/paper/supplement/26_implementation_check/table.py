"""Implementation-check table from results/xval.json: each quantity, ours, the
official package and the relative gap.  Also prints the DINOv2-FD signal excess
(A-Bn minus A-B) under both resizes.  Writes table.csv and table.tex.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = [("FID (null pair)", "ours_fid_AB", "ptfid_AB", "pytorch-fid"),
        ("FID (noise pair)", "ours_fid_ABn", "ptfid_ABn", "pytorch-fid"),
        ("density", "ours_density_AB", "prdc_density_AB", "prdc"),
        ("coverage", "ours_coverage_AB", "prdc_coverage_AB", "prdc"),
        ("FID (null pair)", "ours_fid_AB", "tf_fid_AB", "torch-fidelity"),
        ("FID (noise pair)", "ours_fid_ABn", "tf_fid_ABn", "torch-fidelity"),
        ("KID (null pair)", "ours_kid_AB", "tf_kid_AB", "torch-fidelity"),
        ("KID (noise pair)", "ours_kid_ABn", "tf_kid_ABn", "torch-fidelity"),
        ("FID (null pair)", "ours_fid_AB", "cleanfid_AB", "clean-fid")]

if __name__ == "__main__":
    r = json.load(open(os.path.join(HERE, "results", "xval.json")))
    rows = []
    for q, o, ref, pkg in ROWS:
        if o in r and ref in r:
            gap = abs(r[o] - r[ref]) / max(abs(r[ref]), 1e-300)
            rows.append((q, r[o], r[ref], pkg, gap))
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "ours", "official", "package", "rel_gap"])
        for q, a, b, pkg, gap in rows:
            w.writerow([q, f"{a:.9g}", f"{b:.9g}", pkg, f"{gap:.3g}"])
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{llll}\n\\toprule\nquantity & ours & official & gap \\\\\n\\midrule\n")
        for q, a, b, pkg, gap in rows:
            g = "exact" if gap == 0 else f"${gap:.1e}$"
            f.write(f"{q} & {a:.9g} & {b:.9g}~(\\texttt{{{pkg}}}) & {g} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for t in ("straight", "cropref"):
        if f"dino_{t}_fd_AB" in r:
            print(f"DINOv2-FD {t}: null {r[f'dino_{t}_fd_AB']:.3f}  noise {r[f'dino_{t}_fd_ABn']:.3f}  "
                  f"signal excess {r[f'dino_{t}_fd_ABn'] - r[f'dino_{t}_fd_AB']:+.2f}")
    print(open(os.path.join(HERE, "table.csv")).read())
