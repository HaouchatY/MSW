"""Table 3, MSW column: which pairs of preprocessing pipelines does msw.test
tell apart?  For each pipeline pair and N in {16, 64, 128}: 30 repetitions,
each drawing N images from the first half of the 1600 COCO images through
pipeline i and N different images from the second half through pipeline j;
60 null repetitions per pipeline (i = j).  A pair counts as resolved when the
test rejects at alpha = 0.05 in at least half the repetitions.

    python prep.py && python run.py        # ~2 GPU-hours
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV
from ours import test
from prep import OUT, PIPELINES
from protocol import Store

NS, NULLS, REPS = (16, 64, 128), 60, 30
PIX = {v: np.load(os.path.join(OUT, f"{v}.npy"), mmap_mode="r") for v in PIPELINES}
HALF = PIX[PIPELINES[0]].shape[0] // 2


def images(v, idx):
    return torch.from_numpy(np.ascontiguousarray(PIX[v][idx])).permute(
        0, 3, 1, 2).float().div_(255).to(DEV)


def rep(vi, vj, N, r, seed):
    g = torch.Generator().manual_seed(seed + 977 * r)
    ia = torch.randperm(HALF, generator=g)[:N].numpy()
    ib = torch.randperm(HALF, generator=g)[:N].numpy() + HALF
    return list(test(images(vi, ia), images(vj, ib)))


if __name__ == "__main__":
    shard, nsh = (int(x) for x in (sys.argv[1:] or ["0", "1"]))
    store = Store(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                               f"msw_{shard}.json"))
    jobs = []
    for N in NS:
        jobs += [(f"null|{v}|{N}", v, v, N, NULLS, 70000) for v in PIPELINES]
        jobs += [(f"alt|{a}|{b}|{N}", a, b, N, REPS, 1000)
                 for a, b in itertools.combinations(PIPELINES, 2)]
    for k, (cell, vi, vj, N, reps, seed) in enumerate(jobs):
        if k % nsh != shard:
            continue
        arr = store.cell(cell)
        while len(arr) < reps:
            arr.append(rep(vi, vj, N, len(arr), seed))
            store.save()
        print(cell, f"reject {np.mean([p <= 0.05 for p, _ in arr]):.2f}", flush=True)
