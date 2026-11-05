"""Figure (feature nullspace), MSW series: a green-magenta Nyquist checker on the
R-G opponent axis (amplitude eps = 0.02, clamped to [0, 1]) added to ImageNet-256
images, which the feature extractors' input resizes erase; control: Gaussian blur
of 0.5 px at N = 500.

Protocol of the original run (exp63): imagenet256_ref.pt (4096 images, 256 px) is
split into disjoint halves H1 = [:2048], H2 = [2048:].  Null repetition r draws N
images from each half with one CPU generator seeded 70000 + r; alternative
repetition r draws from H1 and H2 with a CPU generator seeded 1000 r + 7 and
perturbs the H2 set (CPU generators, so the draws are those of the baselines,
bit for bit).  Per cell, msw.test gives (p, T); 50 null and 30 alternative
repetitions at N in {64, 128, 500, 2000}, plus the blur control at N = 500.

    python run.py                                   # all cells, ~2 GPU-hours
    python run.py --cells null@2000 chroma0.02@2000 # a subset; cells can run in parallel
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import run_cell
from draws import DATA, DEV, blur
from ours import test
from protocol import Store

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [64, 128, 500, 2000]
CELLS = ([f"null@{N}" for N in NS] + [f"chroma0.02@{N}" for N in NS] + ["blur0.5@500"])
REPS = {"null": 50, "chroma0.02": 30, "blur0.5": 30}

_POOL = {}


def pool():
    """H1, H2 and the checker pattern (unit-norm R-G axis times (-1)^(i+j))."""
    if not _POOL:
        x = torch.load(os.path.join(DATA, "imagenet256_ref.pt"), map_location="cpu", mmap=True)
        r = torch.arange(x.shape[-1], device=DEV)
        s = ((-1.0) ** (r[:, None] + r[None, :])).view(1, 1, x.shape[-1], x.shape[-1])
        v = torch.tensor([1.0, -1.0, 0.0], device=DEV).view(1, 3, 1, 1) / math.sqrt(2)
        _POOL.update(H1=x[:x.shape[0] // 2], H2=x[x.shape[0] // 2:], PAT=s * v)
    return _POOL["H1"], _POOL["H2"], _POOL["PAT"]


def sample(h, N, g):
    return h[torch.randperm(h.shape[0], generator=g)[:N]].to(DEV)


def pair(name, N, r):
    H1, H2, PAT = pool()
    if name == "null":
        g = torch.Generator().manual_seed(70000 + r)
        return sample(H1, N, g), sample(H2, N, g)
    g = torch.Generator().manual_seed(1000 * r + 7)
    a, b = sample(H1, N, g), sample(H2, N, g)
    if name.startswith("chroma"):
        return a, (b + float(name[6:]) * PAT).clamp(0, 1)
    if name.startswith("blur"):
        return a, blur(b, float(name[4:]))
    raise ValueError(name)


@torch.no_grad()
def rep(name, N, r):
    a, b = pair(name, N, r)
    out = test(a, b)
    del a, b
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--reps", type=int, default=None, help="override every cell's count")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    for cell in a.cells:
        name, N = cell.split("@")
        store = Store(os.path.join(a.out, f"msw_{name}_N{N}.json"))   # one file per cell
        run_cell(store, f"{name}|N{N}", a.reps or REPS[name],
                 lambda r, name=name, N=int(N): rep(name, N, r))
