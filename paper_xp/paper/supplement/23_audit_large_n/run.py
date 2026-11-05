"""Preprocessing audit at large N, MSW column: msw.test and msw.distance between
pipelines on identical content.  For every pipeline pair (i, j) of the COCO grid
(../../main/05_pipeline_audit/prep.py, 1600 images per pipeline), set A is the
first 800 images through pipeline i and set B the last 800 through pipeline j;
the eight same-pipeline nulls (i = j) use the same split.  This is the split of
the archived baseline shifts (results/baselines.json), one evaluation per cell.
Saved per cell: [p, T, distance, lo, hi].

    python ../../main/05_pipeline_audit/prep.py && python run.py    # ~30 GPU-minutes
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "main", "05_pipeline_audit"))
from draws import DEV
from ours import dist, test
from prep import OUT, PIPELINES
from protocol import Store


def images(v, idx):
    x = np.load(os.path.join(OUT, f"{v}.npy"), mmap_mode="r")
    return torch.from_numpy(np.ascontiguousarray(x[idx])).permute(0, 3, 1, 2).float().div_(255).to(DEV)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    store = Store(os.path.join(a.out, "msw.json"))
    ia, ib = np.arange(a.N), np.arange(a.N) + 1600 // 2
    cells = [(v, v) for v in PIPELINES] + list(itertools.combinations(PIPELINES, 2))
    for vi, vj in cells:
        key = f"{vi}|{vj}" if vi != vj else f"null|{vi}"
        if key in store.d:
            continue
        with torch.no_grad():
            A, B = images(vi, ia), images(vj, ib)
            store.d[key] = list(test(A, B)) + list(dist(A, B))
        store.save()
        p, T, d = store.d[key][:3]
        print(f"{key:40s} p={p:.4g}  T={T:6.2f}  d={d:+.5f}", flush=True)
        del A, B
