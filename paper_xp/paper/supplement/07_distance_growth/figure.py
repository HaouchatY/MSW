"""Figures S5-S6 from results/: data/s5_*.csv (mean normalised curves, six
families), data/s6_<N>_<method>.csv (single runs, CelebA noise), the two TikZ
bodies, and data/summary.json with every number the prose quotes.

Normalisation, per run and identical for every method:
y = (v(eps) - v(0)) / (v(eps_max) - v(0)); coverage enters with its sign flipped
(it falls under the alternative).  A run is monotone when no step falls back by
more than 1% of that run's own total rise.  MSW's resolution limit is the
smallest rung at which most runs' 95% interval clears zero.  Families or methods
without results are skipped.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import FAMILIES, key  # noqa: E402

RES = os.environ.get("RESULTS", os.path.join(HERE, "results"))
OUT = os.environ.get("FIGURE_OUT", HERE)          # smoke tests write elsewhere
DATA = os.path.join(OUT, "data")
S5 = ["grf_slope", "celeba_noise", "boost", "cifar_blur", "cifar_checker", "shading"]
TITLES = {"grf_slope": "fields, slope", "celeba_noise": "CelebA, noise",
          "boost": "fields, confined boost", "cifar_blur": "CIFAR, blur",
          "cifar_checker": "CIFAR, checker", "shading": "CelebA, shading"}
METHODS = [("fid", "FID", "cfid", "cbVerm"), ("kid", "KID", "ckid", "cbGreen"),
           ("cmmd", "CMMD", "ccmmd", "cbBlue"), ("dino", "DINOv2", "cdino", "cbOrange"),
           ("c2st", "C2ST", "cc2st", "cbPurple"), ("coverage", "coverage", "ccov", None),
           ("sw", "SW", "csw", "black!45"), ("csw", "CSW", "ccsw", "cbSky"),
           ("pswd", "pSWD", "cpswd", "cbVerm!60!black"), ("ours", "MSW (ours)", "cours", "black")]
SGN = {"coverage": -1}


def norm(a):
    v0, sp = a[:, :1], a[:, -1:] - a[:, :1]
    return (a - v0) / np.where(np.abs(sp) < 1e-30, 1.0, sp)


def score(a, tol=0.01):
    """(mean Kendall tau over runs, runs that never fall back by > tol of their rise, runs)."""
    t = np.array([kendalltau(range(a.shape[1] - 1), r[1:]).statistic for r in a])
    t = t[np.isfinite(t)]
    v = a[:, 1:]
    sp = np.abs(v[:, -1] - v[:, 0])
    drop = np.maximum(0.0, -np.diff(v, axis=1)).max(axis=1) / np.where(sp < 1e-30, 1.0, sp)
    return float(t.mean()), int((drop <= tol).sum()), len(t)


def curves(tag):
    """{method: (reps, rungs) array}; ours as (value, lo, hi) arrays."""
    base = json.load(open(os.path.join(RES, "baselines.json")))[tag]
    ks = [key(e) for e in [0.0] + FAMILIES[tag][2]]
    out = {m: SGN.get(m, 1) * np.array([base["values"][m][k] for k in ks]).T
           for m, *_ in METHODS if m in base["values"]}
    p = os.path.join(RES, f"msw_{tag}.json")
    if os.path.exists(p):
        st = json.load(open(p))
        cells = [st.get(f"{tag}|eps{k}", []) for k in ks]
        n = min(len(c) for c in cells)
        if n >= 2:
            a = np.array([c[:n] for c in cells])           # (rungs, reps, 3)
            out["ours"] = a[:, :, 0].T
            out["ours_lo"], out["ours_hi"] = a[:, :, 1].T, a[:, :, 2].T
    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def sweep(summary):
    body = []
    for tag in S5:
        grid, c = FAMILIES[tag][2], curves(tag)
        eps = [0.0] + grid
        cols = {m: norm(c[m]).mean(0) for m, *_ in METHODS if m in c}
        st = {m: score(c[m]) for m in cols}
        if "ours" in c:
            v0, sp = c["ours"][:, :1], c["ours"][:, -1:] - c["ours"][:, :1]
            sp = np.where(np.abs(sp) < 1e-30, 1.0, sp)
            cols["ours_lo"] = ((c["ours_lo"] - v0) / sp).mean(0)
            cols["ours_hi"] = ((c["ours_hi"] - v0) / sp).mean(0)
            frac = (c["ours_lo"][:, 1:] > 0).mean(0)
            idx = np.where(frac >= 0.5)[0]
            st["_res_eps"] = float(grid[idx[0]]) if len(idx) else None
            st["_res_frac"] = [round(float(x), 3) for x in frac]
        # how far each mean (and single) normalised curve leaves [0, 1]
        st["_range"] = {m: [float(cols[m].min()), float(cols[m].max()),
                            float(norm(c[m]).min()), float(norm(c[m]).max())] for m in st if m in cols}
        ks = [m for m in [m for m, *_ in METHODS] + ["ours_lo", "ours_hi"] if m in cols]
        write_csv(os.path.join(DATA, f"s5_{tag}.csv"), ["eps"] + ks,
                  [[key(e)] + [f"{cols[k][j]:.5f}" for k in ks] for j, e in enumerate(eps)])
        summary["s5"][tag] = st
        body.append((tag, [m for m, *_ in METHODS if m in cols], st.get("_res_eps")))
    return body


def singlerun(summary):
    rows = {}
    for tag in ["noise_n1000", "noise_n200", "noise_n25"]:
        N, c = FAMILIES[tag][1], curves(tag)
        eps = [0.0] + FAMILIES[tag][2]
        rows[N] = []
        for m, *_ in METHODS:
            if m == "coverage" or m not in c:
                continue
            a = norm(c[m])
            write_csv(os.path.join(DATA, f"s6_{N}_{m}.csv"), ["eps"] + [f"r{i + 1}" for i in range(len(a))],
                      [[key(e)] + [f"{a[i, j]:.5f}" for i in range(len(a))] for j, e in enumerate(eps)])
            summary["s6"][f"{N}/{m}"] = score(c[m])
            rows[N].append(m)
    return rows


def agg(summary, m):
    v = [summary["s5"][t][m] for t in S5 if m in summary["s5"][t]]
    return sum(x[1] for x in v), sum(x[2] for x in v), float(np.mean([x[0] for x in v]))


SWEEP_HEAD = r"""% Generated by figure.py.  fig:sweep: mean over runs of the per-run normalised
% distance (v(eps) - v(0)) / (v(eps_max) - v(0)), paired design; curves that
% overshoot leave the panel (summary.json, s5/*/_range).  Dotted rule: MSW's
% resolution limit.
\begin{tikzpicture}
\begin{groupplot}[group style={group size=3 by 2, horizontal sep=14pt, vertical sep=30pt,
    ylabels at=edge left, xlabels at=edge bottom},
  mswbase, xmode=log, ymode=normal, width=2.06in, height=3.9cm,
  ymin=-0.12, ymax=1.14, ytick={0,0.25,0.5,0.75,1}, yticklabels={0,,.5,,1},
  ylabel={normalised distance}, xlabel={$\varepsilon$},
  every axis title/.append style={font=\scriptsize, yshift=-1pt}]"""

SR_HEAD = r"""% Generated by figure.py.  fig:singlerunnoise: every run of CelebA noise at
% N = 1000, 200, 25, normalised by its own rise; the count is runs that never
% fall back by more than 1% of that rise.
\begin{tikzpicture}
\begin{groupplot}[group style={group size=NCOL by 3, horizontal sep=3pt, vertical sep=13pt,
  ylabels at=edge left, xlabels at=edge bottom, xticklabels at=edge bottom},
  mswbase, width=1.31in, height=3.5cm, xmode=log,
  xmin=0.00042, xmax=0.048, ymin=-0.28, ymax=1.22,
  ytick={0,0.5,1}, yticklabels={0,.5,1}, xlabel={noise $\sigma$},
  xtick={0.001,0.01}, xticklabels={.001,.01}, minor xtick={0.0005,0.0025,0.005,0.02,0.04}]"""


def tex(body, rows, summary):
    out = [SWEEP_HEAD]
    style = {m: s for m, _, s, _ in METHODS}
    for i, (tag, ms, res) in enumerate(body):
        opts = [f"title={{{TITLES[tag]}}}"]
        if i == len(body) - 1:
            opts.append("legend to name=growlegend, legend columns=5, legend style={font=\\scriptsize},\n"
                        "  legend entries={" + ", ".join(lab for m, lab, *_ in METHODS if m in ms) + "}")
        out.append(f"\\nextgroupplot[{', '.join(opts)}]")
        if res:
            out.append(f"\\addplot[black!55, densely dotted, thin, no marks, forget plot] "
                       f"coordinates {{({res:g},-0.12)({res:g},1.14)}};")
        for m in ms:
            out.append(f"\\addplot[{style[m]}] table[x=eps, y={m}, col sep=comma]{{data/s5_{tag}.csv}};")
    out += [r"\end{groupplot}",
            r"\node[anchor=north] at ($(group c2r2.south)+(0,-26pt)$) {\pgfplotslegendfromname{growlegend}};",
            r"\end{tikzpicture}"]
    open(os.path.join(OUT, "figure_body.tex"), "w").write("\n".join(out) + "\n")

    ncol = max(len(v) for v in rows.values())
    out = [SR_HEAD.replace("NCOL", str(ncol))]
    color = {m: c for m, _, _, c in METHODS}
    label = {m: lab for m, lab, *_ in METHODS}
    for i, N in enumerate(rows):
        for j, m in enumerate(rows[N]):
            t, k, n = summary["s6"][f"{N}/{m}"]
            opts = ([f"title={{{label[m]}}}"] if i == 0 else []) + (["yticklabels={}"] if j else [])
            out.append(f"\\nextgroupplot[{', '.join(opts)}]")
            out.append("\\addplot[black!70, thin, no marks, forget plot] coordinates {(0.0005,0)(0.04,0)};")
            out.append(f"\\pgfplotsinvokeforeach{{1,...,{n}}}{{\\addplot[{color[m]}, opacity=0.55, thin, no marks]"
                       f" table[x=eps, y=r#1, col sep=comma]{{data/s6_{N}_{m}.csv}};}}")
            out.append(f"\\node[anchor=south east, font=\\tiny, inner sep=1pt] at (rel axis cs:0.96,0.07)"
                       f" {{{k}/{n}}};")
    out.append(r"\end{groupplot}")
    for i, N in enumerate(rows):
        out.append(rf"\node[rotate=90, font=\scriptsize] at ($(group c1r{i + 1}.west)+(-26pt,0)$) {{$N={N}$}};")
    out += [r"\node[rotate=90, font=\scriptsize] at ($(group c1r2.west)+(-40pt,0)$) {normalised distance, one run};",
            r"\end{tikzpicture}"]
    open(os.path.join(OUT, "figure_singlerun_body.tex"), "w").write("\n".join(out) + "\n")


def statements(summary):
    """The numbers the prose of app:sweeps quotes, from the data."""
    s = []
    for m in ["ours", "fid", "cmmd", "kid", "dino", "c2st", "coverage", "sw", "csw", "pswd"]:
        g, n, t = agg(summary, m)
        if n:
            s.append(f"S5 {m:8s} monotone {g}/{n}  mean tau {t:.3f}")
    r = summary["s5"]["grf_slope"]["_range"]
    s.append(f"fields slope: mean CSW curve peaks at {r['csw'][1]:.2f} (one run to {r['csw'][3]:.1f}),"
             f" mean SW curve dips to {r['sw'][0]:.2f} (one run to {r['sw'][2]:.1f})")
    for tag in S5:
        if "_res_eps" in summary["s5"][tag]:
            s.append(f"resolution limit {tag}: {summary['s5'][tag]['_res_eps']}")
    for k, (t, g, n) in summary["s6"].items():
        s.append(f"S6 {k:14s} monotone {g}/{n}  tau {t:.3f}")
    return s


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    summary = {"s5": {}, "s6": {}}
    body = sweep(summary)
    rows = singlerun(summary)
    tex(body, rows, summary)
    summary["statements"] = statements(summary)
    json.dump(summary, open(os.path.join(DATA, "summary.json"), "w"), indent=1)
    print("\n".join(summary["statements"]))
