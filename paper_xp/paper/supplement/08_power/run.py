"""Figure (power against sample count), MSW curve: the power of the exact test
at a fixed perturbation as the number of images per set grows.

Two panels with the house draws: fields, slope 0.032, and CelebA 32-px crops,
blur 0.33.  N in {25, 50, 100, 200, 500, 1000}; 100 alternative repetitions per
rung (seed 1000 r + 7) and 50 null repetitions (seed 70000 + r), the null giving
the empirical size.  Power is the fraction of alternative repetitions with
p <= 0.05 -- no simulated null is needed for it.  Saved per repetition:
[p, T, distance, lo, hi] (msw.test, msw.distance on the same two sets).

    python run.py                          # both panels, ~20 GPU-minutes
    python run.py --settings celeba_blur   # one panel; panels can run in parallel
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import alt_seed, gen, null_seed, setting
from ours import dist, test
from protocol import Store, rejection_rate

PANELS = {"fields_slope": 0.032, "celeba_blur": 0.33}
NS = [25, 50, 100, 200, 500, 1000]


@torch.no_grad()
def rep(a, b):
    return list(test(a, b)) + list(dist(a, b))


def fill(store, cell, reps, draw_pair):
    arr = store.cell(cell)
    t0, n0 = time.time(), len(arr)
    while len(arr) < reps:
        arr.append(rep(*draw_pair(len(arr))))
        if len(arr) % 10 == 0 or len(arr) == reps:
            store.save()
    if len(arr) > n0:
        print(f"{cell:32s} reps={len(arr):4d}  rejection={rejection_rate([r[0] for r in arr]):.3f}"
              f"  ({(time.time() - t0) / (len(arr) - n0):.2f} s/rep)", flush=True)


def null_pair(draw, N):
    def f(r):
        g = gen(null_seed(r))
        return draw(N, g), draw(N, g)
    return f


def alt_pair(draw, alt_fn, N):
    def f(r):
        g = gen(alt_seed(r))
        return draw(N, g), alt_fn(N, g)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="*", default=list(PANELS))
    ap.add_argument("--N", type=int, nargs="*", default=NS)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--nulls", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    for s in a.settings:
        draw, alt = setting(s)
        store = Store(os.path.join(a.out, f"msw_{s}.json"))
        for N in a.N:
            fill(store, f"{s}|N{N}|alt", a.reps, alt_pair(draw, alt(PANELS[s]), N))
            fill(store, f"{s}|N{N}|null", a.nulls, null_pair(draw, N))
