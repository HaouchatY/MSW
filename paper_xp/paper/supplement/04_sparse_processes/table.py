"""Table tab:ssp1d and the numbers of app:sspsmooth from results/.

table.csv / table.tex   1D near-integrator benchmark: signed deflections
    z = (mean_alt - mean_null) / sd_null of every baseline (12 repetitions) and
    of MSW's T (50 repetitions, 64 x 64 raster), plus MSW's exact-test
    rejections at alpha = 0.05.
sspsmooth.csv   every (operator, dimension, contrast, N) cell: the same signed
    deflection for every metric and MSW, and MSW's rejection rate.

    python table.py
"""
import csv
import json
import os

from figure import HERE, ours, z

ROWS = [("laplace", "Laplace"), ("student4", "Student-$t_4$"),
        ("bernlap_0.05", "Bern.--Lap.\\ $p{=}0.05$"), ("bernlap_0.005", "Bern.--Lap.\\ $p{=}0.005$"),
        ("opshift", "operator control")]
COLS = ["spectrum", "sw", "fwd", "wmmd", "wc2st"]
HEAD = ["spectrum", "SW", "F-w2v", "MMD-w2v", "C2ST-w2v", "MSW", "MSW rej."]


def cells(base, op, dom):
    """[(contrast, N, {metric: z}, (msw z, power, reps) or None)] for one operator/dimension."""
    b, mine, out = base[op][dom], ours(op, dom), []
    for c in b:
        if c == "null":
            continue
        for N in sorted(b[c], key=int):
            zs = {m: z(b[c][N][m], b["null"][N][m]) for m in b[c][N]}
            o = mine.get((c, int(N)))
            out.append((c, int(N), zs, (o[0], o[1], o[3]) if o else None))
    return out


if __name__ == "__main__":
    base = json.load(open(os.path.join(HERE, "results", "baselines.json")))
    one = {(c, N): (zs, o) for c, N, zs, o in cells(base, "near", "d1")}
    rows = []
    for c, name in ROWS:
        for N in [8, 32, 128]:
            zs, o = one[(c, N)]
            rows.append((c, name, N, [zs[m] for m in COLS] + ([o[0]] if o else [None]),
                         f"{round(o[1] * o[2])}/{o[2]}" if o else "--"))
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "N"] + COLS + ["msw", "msw_rejections"])
        for c, _, N, v, rej in rows:
            w.writerow([c, N] + ["" if x is None or x != x else f"{x:.2f}" for x in v] + [rej])
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{llccccccc}\n\\toprule\ncontrast & $N$ & " + " & ".join(HEAD)
                + " \\\\\n\\midrule\n")
        for i, (c, name, N, v, rej) in enumerate(rows):
            v = [None if x is None or x != x else x for x in v]
            best = max(x for x in v if x is not None)
            cell = lambda x: "--" if x is None else (f"$\\mathbf{{{x:+.1f}}}$" if x == best else f"${x:+.1f}$")
            f.write(f"{name if i % 3 == 0 else ''} & {N} & " + " & ".join(cell(x) for x in v)
                    + f" & {rej} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(HERE, "sspsmooth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operator", "dim", "contrast", "N", "metric", "z", "msw_power", "reps"])
        for op in ["near", "smooth", "axis"]:
            for dom in ["d1", "d2"]:
                for c, N, zs, o in cells(base, op, dom):
                    w.writerows([op, dom, c, N, m, f"{v:.2f}", "", 12] for m, v in zs.items())
                    if o:
                        w.writerow([op, dom, c, N, "msw", "" if o[0] != o[0] else f"{o[0]:.2f}", f"{o[1]:.3f}", o[2]])
    print(open(os.path.join(HERE, "table.csv")).read())
