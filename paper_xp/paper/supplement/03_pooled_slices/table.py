"""Tables pooled and pooled-supp from results/.  Writes table.csv (eps_0.8 per
setting, budget and method), table.tex (both tables), and data/pooled_{power_grid,
paperrows,size,clampcontrol}.csv.

eps_0.8 is the smallest rung with power >= 0.8 that keeps power >= 0.8 at every
larger rung (protocol.n_star applied to the eps grid; the rule of the original
assembly).  Power = rate of p <= 0.05 over the 24 repetitions.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, HERE)
from protocol import n_star
from run import CLAMP_EPS, GRID, K512, NULLS, PAPER, SETTINGS
from bench import key

RIV = ["ks", "mmd", "energy", "w2"]
COLS = RIV + ["MSW_T", "MSW_CORE"]
HYB = [m + "+TLPXJ" for m in RIV]
LBL = {"fields_slope": "Fields, slope", "fields_marginal": "Fields, marginal",
       "celeba_noise": "CelebA, noise", "celeba_blur": "CelebA, blur",
       "celeba_shading": "CelebA, shading", "cifar_checker": "CIFAR-10, checker"}
PAPER500 = {"celeba_blur": 0.33, "celeba_shading": 0.08, "cifar_checker": 0.01}


def load(name, res):
    p = os.path.join(res, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def pw(v, m):
    return sum(r[m][0] <= 0.05 for r in v) / len(v)


def eps80(cells, grid, m):
    e = n_star(grid, lambda x: pw(cells[x], m) if x in cells else None)
    return e


def _inner(s, e):
    if s == "fields_marginal":
        return f"1/{round(1 / e)}"
    if e < 1e-3:
        return f"{e * 1e4:.4g}\\!\\cdot\\!10^{{-4}}"
    return f"{e:.3g}"


def fx(s, e, bold=False):
    if e is None:
        return r"$>$grid"
    return f"$\\mathbf{{{_inner(s, e)}}}$" if bold else f"${_inner(s, e)}$"


def main(res=os.path.join(HERE, "results"), out=HERE):
    """res: the results directory; out: where table.* and data/ go."""
    E, Pc, Sz, cellsG = {}, {}, {}, {}
    for s in SETTINGS:
        g, k = load(f"grid_{s}.json", res), load(f"k512_{s}.json", res)
        G = {e: g[c] for e in GRID[s] if (c := f"{s}|N500|eps{key(e)}") in g and g[c]}
        K = {e: k[c] for e in K512[s] if (c := f"{s}|N500|eps{key(e)}") in k and k[c]}
        cellsG[s] = (G, K)
        if G:
            E[(s, 2048)] = {m: eps80(G, GRID[s], m) for m in COLS + HYB}
        if K:
            E[(s, 8192)] = {m: eps80(K, K512[s], m) for m in COLS}
        eps, Ns = PAPER.get(s, (PAPER500.get(s), []))
        rows = [(N, g[c]) for N in Ns if (c := f"{s}|N{N}|eps{key(eps)}") in g and g[c]]
        if s in PAPER500 and PAPER500[s] in G:
            rows.append((500, G[PAPER500[s]]))
        if rows:
            Pc[s] = (eps, rows)
        Sz[s] = [(N, g[c]) for N, _ in NULLS.get(s, []) if (c := f"{s}|N{N}|null") in g and g[c]]

    def ratio(s, budget):
        rv = [E[(s, budget)][m] for m in RIV if E[(s, budget)][m] is not None]
        c = E.get((s, 2048), {}).get("MSW_CORE")
        return min(rv) / c if rv and c else None

    fmt = lambda e: "" if e is None else f"{e:.6g}"
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    with open(os.path.join(out, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "budget"] + COLS + HYB + ["best_rival_over_core"])
        for s in SETTINGS:
            for b in (2048, 8192):
                if (s, b) in E:
                    r = ratio(s, b)
                    w.writerow([s, b] + [fmt(E[(s, b)].get(m)) for m in COLS + HYB]
                               + ["" if r is None else f"{r:.4g}"])
    with open(os.path.join(out, "data", "pooled_power_grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "budget", "eps", "reps"] + COLS + HYB)
        for s in SETTINGS:
            G, K = cellsG[s]
            for e in sorted(G):
                w.writerow([s, 2048, f"{e:.6g}", len(G[e])] + [f"{pw(G[e], m):.3f}" for m in COLS + HYB])
            for e in sorted(K):
                w.writerow([s, 8192, f"{e:.6g}", len(K[e])] + [f"{pw(K[e], m):.3f}" for m in COLS]
                           + [""] * len(HYB))
    with open(os.path.join(out, "data", "pooled_paperrows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "eps", "N", "reps"] + COLS + HYB)
        for s, (eps, rows) in Pc.items():
            for N, v in rows:
                if N != 500:
                    w.writerow([s, f"{eps:.6g}", N, len(v)] + [f"{pw(v, m):.3f}" for m in COLS + HYB])
    with open(os.path.join(out, "data", "pooled_size.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "N", "nulls"] + COLS + HYB)
        for s in SETTINGS:
            for N, v in Sz[s]:
                w.writerow([s, N, len(v)] + [f"{pw(v, m):.4f}" for m in COLS + HYB])
    cl = load("clamp.json", res)
    with open(os.path.join(out, "data", "pooled_clampcontrol.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "eps", "reps", "ks", "mmd", "energy", "w2", "MSW_T", "MSW_CORE"])
        for v in ("noclamp", "paper"):
            for e in CLAMP_EPS:
                if (c := f"{v}|N500|eps{key(e)}") in cl and cl[c]:
                    w.writerow([v, e, len(cl[c])] + [f"{pw(cl[c], m):.3f}" for m in COLS])

    # ---------------------------------------------------------------- tex
    L = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r" & \multicolumn{4}{c}{generic two-sample test on the pooled slices} & "
         r"\multicolumn{2}{c}{ours} & best generic \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
         r"Setting & KS-max & MMD & energy & $W_2$ & transport only & MSW (CORE) & $/$ MSW \\"]
    heads = {2048: r"(a) matched budget: $2048$ subsampled points per side per slice, as in the previous version",
             8192: r"(b) the generic tests at $4\times$ the budget ($8192$ points per side per slice); ours unchanged"}
    for b in (2048, 8192):
        L += [r"\midrule", r"\multicolumn{8}{l}{\emph{" + heads[b] + r"}}\\"]
        for s in SETTINGS:
            if (s, b) not in E:
                continue
            v = E[(s, b)]
            best = min([v[m] for m in RIV if v[m] is not None], default=None)
            cells = [fx(s, v[m], bold=m in RIV and best is not None and v[m] is not None and v[m] <= best)
                     for m in COLS]
            if b == 8192:
                cells[4] = cells[5] = r"\textemdash"
            r = ratio(s, b)
            cells.append(f"${r:.3g}\\times$" if r else r"\textemdash")
            L.append(f"{LBL[s]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", "",
          r"\begin{tabular}{lcccccc}", r"\toprule",
          r"Setting & KS-max & MMD & energy & $W_2$ & transport only & MSW (CORE) \\", r"\midrule",
          r"\multicolumn{7}{l}{\emph{(c) $\varepsilon_{0.8}$ of CORE with only its transport rows "
          r"swapped (TL$+$PX$+$J kept)}}\\"]
    for s in SETTINGS:
        if (s, 2048) in E:
            v = E[(s, 2048)]
            L.append(f"{LBL[s]} & " + " & ".join([fx(s, v[m + "+TLPXJ"]) for m in RIV]
                                                 + [r"\textemdash", fx(s, v["MSW_CORE"])]) + r" \\")
    L += [r"\midrule", r"\multicolumn{7}{l}{\emph{(d) power at $\alpha=0.05$ on the cells of the "
                       r"previous version of this table}}\\"]
    for s, (eps, rows) in Pc.items():
        e_ = f"1/{round(1 / eps)}" if s == "fields_marginal" else f"{eps:.3g}"
        ns = "/".join(str(N) for N, _ in rows)
        cells = ["$" + "/".join(f"{pw(v, m):.2f}".lstrip("0") for _, v in rows) + "$" for m in COLS]
        L.append(f"{LBL[s]} ${e_}$, $N{{=}}{ns}$ & " + " & ".join(cells) + r" \\")
    L += [r"\midrule", r"\multicolumn{7}{l}{\emph{(e) size at $\alpha=0.05$ under $H_0$}}\\"]
    for s in SETTINGS:
        for N, v in Sz[s]:
            L.append(f"{LBL[s]}, $N{{=}}{N}$ $[{len(v)}]$ & "
                     + " & ".join("$" + f"{pw(v, m):.3f}".lstrip("0") + "$" for m in COLS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table.tex"), "w").write("\n".join(L) + "\n")
    print(open(os.path.join(out, "table.csv")).read())


if __name__ == "__main__":
    main(*sys.argv[1:3])
