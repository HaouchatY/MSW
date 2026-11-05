"""Real generators: MSW along two quality ladders and between adjacent checkpoints.

Pools (4096 samples per rung, generated once by SW_project/exp/exp57_generators.py):
DDPM on CIFAR-10 (google/ddpm-cifar10-32) sampled with DDIM at S in {2, 5, 10, 20,
50} steps, against CIFAR-10 train; DCGAN trained on CelebA-64 images 0-7999,
checkpoints at {200, 500, 1000, 2000, 5000, 10000} steps, against the held-out
CelebA images 8000-15999.  Draws replay exp57 seed for seed (CUDA generator,
`randperm(pool)[:N]`): null repetition r draws the two halves of the real pool
from one generator seeded 70000 + r; ladder rung r draws real then generated,
adjacent pair r the earlier then the later checkpoint, from one generator seeded
1000 r + 7.  CUDA randperm differs across torch versions: the baselines' draws
are exactly these under the original torch 2.10 (cu128).  Cells: real-vs-real
null and ladder rungs at N = 1024, null and the nine adjacent pairs at N = 128;
50 repetitions each (the baselines ran 10).
Saved per repetition: msw.test (p, T) in '<cell>', and msw.distance (value, interval)
in '<cell>|dist' on the first 20 repetitions of the N = 1024 null and ladder cells.

    python run.py                    # both tracks, ~40 GPU-minutes (DCGAN 64 px dominates)
    python run.py --tracks gan       # one track; tracks can run in parallel

Pools on the cluster: $MSW_XP_DATA/gen_pools/{ddpm_s*,gan_s*}.pt (1.46 GB).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import run_cell
from draws import DATA, DEV, celeba, cifar, gen
from ours import dist, test
from protocol import Store

HERE = os.path.dirname(os.path.abspath(__file__))
RUNGS = {"ddpm": [2, 5, 10, 20, 50], "gan": [200, 500, 1000, 2000, 5000, 10000]}
_POOLS = {}


def real(track):
    return cifar() if track == "ddpm" else celeba()[8000:]


def pool(track, S):
    k = f"{track}_s{S}"
    if k not in _POOLS:
        _POOLS[k] = torch.load(os.path.join(DATA, "gen_pools", f"{k}.pt"), map_location="cpu").to(DEV)
    return _POOLS[k]


def draw(x, N, g):
    return x[torch.randperm(x.shape[0], device=DEV, generator=g)[:N]]


def pair(track, name, N, r):
    """The two image sets of exp57 cell `name` ('null', 'sX' or 'sX-sY'), repetition r."""
    if name == "null":
        x, g = real(track), gen(70000 + r)
        h = x.shape[0] // 2
        return draw(x[:h], N, g), draw(x[h:], N, g)
    g = gen(1000 * r + 7)
    if "-" in name:
        p, q = name.split("-")
        return draw(pool(track, p[1:]), N, g), draw(pool(track, q[1:]), N, g)
    return draw(real(track), N, g), draw(pool(track, name[1:]), N, g)


def cells(track):
    s = RUNGS[track]
    return ([("null", 1024)] + [(f"s{S}", 1024) for S in s] + [("null", 128)]
            + [(f"s{a}-s{b}", 128) for a, b in zip(s[:-1], s[1:])])


@torch.no_grad()
def test_rep(track, name, N, r):
    return test(*pair(track, name, N, r))


@torch.no_grad()
def run_dist(store, track, name, N, reps):
    """store['<cell>|dist'] <- msw.distance (value, lo, hi) on the cell's first reps draws."""
    arr = store.cell(f"{track}/{name}@{N}|dist")
    while len(arr) < reps:
        arr.append(list(dist(*pair(track, name, N, len(arr)))))
        store.save()
    print(f"{track}/{name}@{N}|dist  reps={len(arr)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="*", default=list(RUNGS))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--dist-reps", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    for track in a.tracks:
        store = Store(os.path.join(a.out, f"msw_{track}.json"))
        for name, N in cells(track):
            run_cell(store, f"{track}/{name}@{N}", a.reps,
                     lambda r: test_rep(track, name, N, r))
            if N == 1024:
                run_dist(store, track, name, N, a.dist_reps)
