"""Supplementary Table (tab:time): wall-clock of one evaluation -- one pair of
sets of N images -- for every metric of the paper, at N = 200, 500, 1000.

Every row is timed the same way on the same data: two sets of random 32-px
3-channel CelebA crops (generator seeded N), each metric called as the
benchmark calls it (common/baselines.py, msw), feature extraction of BOTH sets
included; one-time setup is not (loading a network, drawing SW directions or
CSW kernels, the P-S pool z-scoring, building msw's cached portfolio).  After
2 warm-up calls, 25 timed calls with torch.cuda.synchronize before and after
each; the table reports the median.  All 25 times per cell are saved.

    python run.py                                 # all rows; timed calls total 3 min on an H200, plus model loading
    python run.py --metrics cmmd dino --N 1000    # a subset; runs can be split
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import baselines as B
import msw
from draws import DEV, celeba, gen, random_crops
from protocol import Store

METRICS = ["fid", "kid", "cmmd", "dino", "c2st", "coverage", "sw", "csw", "pswd", "spec", "ps",
           "msw_distance", "msw_test"]

ap = argparse.ArgumentParser()
ap.add_argument("--metrics", nargs="*", default=METRICS)
ap.add_argument("--N", type=int, nargs="*", default=[200, 500, 1000])
ap.add_argument("--reps", type=int, default=25)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
a = ap.parse_args()


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, reps, warm):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t0)
    return out


def calls(a, b):
    """metric -> zero-argument call evaluating it on (a, b), setup done here."""
    draw = lambda n, g: random_crops(celeba(), n, gen=g)
    pool = draw(500, gen(1))                       # the benchmark's fixed map to [0, 1]
    mu, sd = float(pool.mean()), float(pool.std())
    to01 = lambda t: ((t - (mu - 4 * sd)) / (8 * sd)).clamp(0, 1)
    sw, csw, psd = B.SW(3, 32, 8192, seed=0), B.CSW(3, 32, 64, 32, seed=0), B.PSDist(draw)
    inc = lambda: (B.inception(to01(a)), B.inception(to01(b)))

    def with_inc(head):
        def f():
            fa, fb = inc()
            return head(fa, fb)
        return f
    m = min(500, a.shape[0])
    return {
        "fid": with_inc(B.fid),
        "kid": with_inc(lambda fa, fb: B.kid(fa, fb, 0, subsets=20, subset_size=m)),
        "cmmd": lambda: B.cmmd(B.clip(to01(a)), B.clip(to01(b))),
        "dino": lambda: B.fid(B.dino(to01(a)), B.dino(to01(b))),
        "c2st": with_inc(lambda fa, fb: B.c2st(fa, fb, 0)),
        "coverage": with_inc(lambda fa, fb: B.coverage(fa, fb, k=5)),
        "sw": lambda: sw.statistic(a, b),
        "csw": lambda: csw.statistic(a, b),
        "pswd": lambda: B.karras_swd(to01(a), to01(b), 0),
        "spec": lambda: B.spec_dist(a, b),
        "ps": lambda: psd(a, b),
        "msw_distance": lambda: msw.distance(a, b),
        "msw_test": lambda: msw.test(a, b),
    }


warnings.simplefilter("ignore")                     # msw's stationarity warning on crops
store = Store(os.path.join(a.out, "times.json"))
store.d["env"] = {"device": torch.cuda.get_device_name() if DEV.type == "cuda" else "cpu",
                  "torch": torch.__version__, "msw": msw.__version__}
for N in a.N:
    g = gen(N)
    x, y = random_crops(celeba(), N, gen=g), random_crops(celeba(), N, gen=g)
    fns = calls(x, y)
    for name in a.metrics:
        cell = store.cell(f"{name}|N{N}")
        if len(cell) >= a.reps:
            continue
        cell[:] = timed(fns[name], a.reps, a.warm)
        store.save()
        print(f"N={N:5d}  {name:13s} median {np.median(cell):8.4f} s  "
              f"(min {min(cell):.4f}, max {max(cell):.4f})", flush=True)
