"""Sparse stochastic processes (Figure fig:sspresults, Table tab:ssp1d,
Appendix app:sspsmooth), MSW column: msw.test and msw.distance between a
Gaussian-innovation reference and each matched-spectrum innovation law or the
operator-change control (see ssp.py), for three whitening operators in 1D and 2D.

Protocol: N in {8, 32, 128} (1D) and {8, 32, 128, 500} (2D); per (operator,
dimension, N) 50 null repetitions (Gaussian vs Gaussian, generator seeded
70000 + r) and 50 repetitions per contrast (generator 1000 r + 7), both sets
from that one generator.  Saved per repetition: p, T of msw.test and the
msw.distance value with its interval.  The deflection is
(mean_alt T - mean_null T) / sd_null T.  msw has no one-dimensional entry
point, so a length-4096 signal is read as its row-major 64 x 64 raster.

    python run.py                          # all, about 1.5 GPU-hours
    python run.py --ops near --doms d2     # one (operator, dimension) per process
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from draws import DEV, alt_seed, gen, null_seed
from ours import dist, test
from protocol import Store, rejection_rate
from ssp import CONTRASTS, NS, Process

ap = argparse.ArgumentParser()
ap.add_argument("--ops", nargs="*", default=["near", "smooth", "axis"])
ap.add_argument("--doms", nargs="*", default=["d1", "d2"])
ap.add_argument("--N", type=int, nargs="*", default=None)
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--nulls", type=int, default=50)
ap.add_argument("--out", default=os.path.join(HERE, "results"))
a = ap.parse_args()


@torch.no_grad()
def rep(proc, contrast, N, seed):
    x, y = proc.pair(contrast, N, gen(seed))
    return list(test(x, y)) + list(dist(x, y))          # p, T, d, lo, hi


for op in a.ops:
    for dom in a.doms:
        proc = Process(op, dom, DEV)
        store = Store(os.path.join(a.out, f"msw_{op}_{dom}.json"))
        for N in a.N or NS[dom]:
            for c in ["null"] + CONTRASTS[op]:
                arr, need = store.cell(f"{c}|N{N}"), a.nulls if c == "null" else a.reps
                seed = null_seed if c == "null" else alt_seed
                t0, n0 = time.time(), len(arr)
                while len(arr) < need:
                    arr.append(rep(proc, c, N, seed(len(arr))))
                    if len(arr) % 10 == 0 or len(arr) == need:
                        store.save()
                if len(arr) > n0:
                    print(f"{op} {dom} {c:14s} N={N:4d} reps={len(arr)} "
                          f"rej={rejection_rate([r[0] for r in arr]):.2f} "
                          f"({(time.time() - t0) / (len(arr) - n0):.2f} s/rep)", flush=True)
