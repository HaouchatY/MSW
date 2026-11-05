"""Table tab:edm2paired from results/: the paired-seed differential z on the four
adjacent EDM2 transitions for our per-image family (results/msw_paired.json),
CLIP and Inception (results/baselines.json, the original exp60 run), and the
rolled-pairing control.  Writes table.csv and table.tex.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MSW_XP_OUT", os.path.join(HERE, "results"))
PAIRS = [(2, 6), (6, 25), (25, 50), (50, 100)]
ROWS = [("ours", "MSW slices (paired)"), ("clip", "CLIP (paired)"), ("incep", "Inception (paired)")]


def main():
    B = json.load(open(os.path.join(RES, "baselines.json")))["paired"]
    pp = os.path.join(RES, "msw_paired.json")
    T = {"ours": json.load(open(pp)) if os.path.exists(pp) else {}, **B}
    keys = [f"{a}_{b}" for a, b in PAIRS] + ["control_rolled"]
    z = {fam: [T[fam][k]["z"] if k in T[fam] else None for k in keys] for fam, _ in ROWS}
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["family"] + keys + ["p_" + k for k in keys])
        for fam, _ in ROWS:
            w.writerow([fam] + ["" if v is None else f"{v:.2f}" for v in z[fam]]
                       + [f"{T[fam][k]['p']:.4g}" if "p" in T[fam].get(k, {}) else "" for k in keys])
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{lccccc}\n\\toprule\ntransition & "
                + " & ".join(f"${a}\\%\\!\\to\\!{b}\\%$" for a, b in PAIRS)
                + " & rolled control \\\\\n\\midrule\n")
        for fam, name in ROWS:
            cells = []
            for j in range(len(keys)):
                v = z[fam][j]
                best = j < len(PAIRS) and v is not None and v == max(
                    z[g][j] for g, _ in ROWS if z[g][j] is not None)
                s = "--" if v is None else f"{v:+.1f}"
                cells.append(f"$\\mathbf{{{s}}}$" if best else f"${s}$")
            f.write(f"{name} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(HERE, "table.csv")).read())


if __name__ == "__main__":
    main()
