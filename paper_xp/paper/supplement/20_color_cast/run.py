"""Appendix 'Global color cast', MSW: power of msw.test against a spatially
constant chromatic cast eps (1, -1, 0)/sqrt2 added to every CelebA 32-px crop
of set B, with and without the raw-pixel row (portfolio CORE and CORE-PX).

House draws (celeba32 random crops); repetition r draws both sets from one
generator, seeded 1000 r + 7 (alternative) or 70000 + r (null).  N = 500,
eps in {0.0025, ..., 0.08} as in the original rerun plus 0.0008, so the grid spans
the 100x amplitude sweep the text cites; 100 alternative repetitions per rung and
300 null repetitions.  One orbit table per repetition gives
both portfolios on the same draws; saved per repetition: [p, T] for each.

    python run.py              # ~25 GPU-minutes
"""
import argparse
import math
import os
import sys
import time

import msw
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, alt_seed, gen, null_seed, setting
from protocol import Store, rejection_rate

EPS = [0.0008, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
PORTFOLIOS = ("CORE", "CORE-PX")
CAST = (torch.tensor([1.0, -1.0, 0.0], device=DEV) / math.sqrt(2.0)).view(1, 3, 1, 1)
_P = {}


@torch.no_grad()
def tests(a, b):
    """{portfolio: [p, T]} from one orbit table; 'CORE' is exactly msw.test(a, b)."""
    key = (a.shape[1], a.shape[-1])
    if key not in _P:
        _P[key] = msw.Portfolio(*key)
    P = _P[key]
    A = P.table(a, b)
    idx = msw.portfolios(P.names)
    out = {}
    for k in PORTFOLIOS:
        S = A[:, idx[k]]
        out[k] = [float(msw.components.p_maxz(S)),
                  float(((S[0] - S.mean(0)) / msw.components.std_floor(S)).max())]
    return out


def fill(store, cell, reps, pair):
    arr = store.cell(cell)
    t0, n0 = time.time(), len(arr)
    while len(arr) < reps:
        arr.append(tests(*pair(len(arr))))
        if len(arr) % 10 == 0 or len(arr) == reps:
            store.save()
    if len(arr) > n0:
        print(f"{cell:14s} reps={len(arr):4d}  " + "  ".join(
            f"{k}={rejection_rate([r[k][0] for r in arr]):.3f}" for k in PORTFOLIOS)
            + f"  ({(time.time() - t0) / (len(arr) - n0):.2f} s/rep)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--nulls", type=int, default=300)
    ap.add_argument("--eps", type=float, nargs="*", default=EPS)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    draw, _ = setting("celeba_blur")                      # its null draw: random 32-px crops
    store = Store(os.path.join(a.out, "msw.json"))

    def null_pair(r):
        g = gen(null_seed(r))
        return draw(a.N, g), draw(a.N, g)

    for e in a.eps:
        def alt_pair(r, e=e):
            g = gen(alt_seed(r))
            return draw(a.N, g), draw(a.N, g) + e * CAST
        fill(store, f"eps{e:g}", a.reps, alt_pair)
    fill(store, "null", a.nulls, null_pair)
