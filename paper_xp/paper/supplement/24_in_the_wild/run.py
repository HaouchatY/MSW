"""In-the-wild table (CelebA version), MSW column: two real pipeline differences,
no injected corruption.

  resize  PIL BICUBIC vs BILINEAR resize 64 -> 32 px
  jpeg    PIL JPEG quality 95 vs 90 at 64 px
Pools (as in the original exp78): CelebA images 0..6143 (A side) and
8000..14143 (B side), uint8-quantised and processed with PIL.  A null
repetition draws N images of the A pool and N of the B pool through the same
pipeline (generator 70000 + r); an alternative one, B through the other
pipeline (generator 1000 r + 7).  N in {16, 64, 256} (the table's columns;
8, 32 and 128 were also run), 200 repetitions per cell and side.  Saved per
repetition: [p, T].  The table reports the deflection of T,
(mean_alt - mean_null) / sd_null, and the exact-test power.

    python run.py                          # both tasks, ~40 GPU-minutes
    python run.py --tasks jpeg --N 8 32 128
"""
import argparse
import io
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import run_cell
from draws import DEV, alt_seed, celeba, gen, null_seed
from ours import test
from protocol import Store

NP = 6144


def _pil(x, f):
    out = []
    for im in (x.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8):
        out.append(torch.from_numpy(np.array(f(Image.fromarray(im)))).permute(2, 0, 1))
    return torch.stack(out).float().div_(255.0).to(DEV)


def jpeg(x, q):
    def f(im):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf)
    return _pil(x, f)


def pools(task):
    """(A, B_null, B_alt) of one task."""
    cel = celeba().cpu()
    h1, h2 = cel[:NP], cel[cel.shape[0] // 2:cel.shape[0] // 2 + NP]
    if task == "jpeg":
        return jpeg(h1, 95), jpeg(h2, 95), jpeg(h2, 90)
    rz = lambda x, m: _pil(x, lambda im: im.resize((32, 32), m))
    return rz(h1, Image.BICUBIC), rz(h2, Image.BICUBIC), rz(h2, Image.BILINEAR)


def take(p, N, g):
    return p[torch.randperm(p.shape[0], device=DEV, generator=g)[:N]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["resize", "jpeg"])
    ap.add_argument("--N", type=int, nargs="*", default=[16, 64, 256])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    for task in a.tasks:
        A, B0, B1 = pools(task)
        store = Store(os.path.join(a.out, f"msw_{task}.json"))
        for N in a.N:
            def null(r, N=N):
                g = gen(null_seed(r))
                return test(take(A, N, g), take(B0, N, g))

            def alt(r, N=N):
                g = gen(alt_seed(r))
                return test(take(A, N, g), take(B1, N, g))
            with torch.no_grad():
                run_cell(store, f"{task}|N{N}|alt", a.reps, alt)
                run_cell(store, f"{task}|N{N}|null", a.reps, null)
