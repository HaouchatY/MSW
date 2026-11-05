"""Figure (app:gender), MSW column: aligned CelebA gender, two contrasts.

drop   reference = natural mixture, comparison = females only (males dropped)
pure   males against females (the same difference without mixture dilution)

Protocol of exp51: the 16,000 aligned 64-px CelebA images are split by index into
half 1 (0-7999, reference pool) and half 2 (8000-15999, comparison pool; 4631
females, 3369 males).  drop: null half1 vs half2, alternative half1 vs females of
half2.  pure: null males vs males of half2, alternative males vs females of half2.
Each set is the first N of a randperm of its pool, both drawn from one generator
seeded 70000 + r (null) or 1000 r + 7 (alternative), first set first.  50 null and
30 alternative repetitions per rung; saved per repetition: msw.test (p, T) and
msw.distance (value).  pure stops at 2048: at 4096 the null would take the whole
male pool on both sides.

    python run.py                                  # all rungs (about 3 GPU-hours)
    python run.py --contrasts drop --N 128 256     # one shard; shards run in parallel
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, celeba, celeba_male, gen
from ours import dist, test
from protocol import Store, rejection_rate

RUNGS = {"drop": [128, 256, 512, 1024, 2048, 4096], "pure": [128, 256, 512, 1024, 2048]}


def pools():
    male = celeba_male()
    h = male.shape[0] // 2
    half1 = torch.arange(0, h, device=DEV)
    half2 = torch.arange(h, male.shape[0], device=DEV)
    return {"half1": half1, "half2": half2, "fem2": half2[~male[half2]], "mal2": half2[male[half2]]}


ARMS = {("drop", "null"): ("half1", "half2"), ("drop", "alt"): ("half1", "fem2"),
        ("pure", "null"): ("mal2", "mal2"), ("pure", "alt"): ("mal2", "fem2")}


@torch.no_grad()
def rep(cel, P, contrast, arm, N, r):
    g = gen(70000 + r if arm == "null" else 1000 * r + 7)
    take = lambda pool: cel[P[pool][torch.randperm(P[pool].shape[0], device=DEV, generator=g)[:N]]]
    pa, pb = ARMS[(contrast, arm)]
    a = take(pa)
    b = take(pb)
    return list(test(a, b)) + [dist(a, b)[0]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrasts", nargs="*", default=list(RUNGS))
    ap.add_argument("--N", type=int, nargs="*")
    ap.add_argument("--nulls", type=int, default=50)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    cel, P = celeba(), pools()
    for c in a.contrasts:
        store = Store(os.path.join(a.out, f"msw_{c}.json"))
        for N in [n for n in RUNGS[c] if not a.N or n in a.N]:
            for arm, reps in (("null", a.nulls), ("alt", a.reps)):
                arr, t0 = store.cell(f"{c}|N{N}|{arm}"), time.time()
                n0 = len(arr)
                while len(arr) < reps:                   # [p, T, distance]
                    arr.append(rep(cel, P, c, arm, N, len(arr)))
                    store.save()
                if len(arr) > n0:
                    print(f"{c} N={N} {arm}: reps={len(arr)} rejection={rejection_rate([x[0] for x in arr]):.3f}"
                          f" ({(time.time() - t0) / (len(arr) - n0):.2f} s/rep)", flush=True)
