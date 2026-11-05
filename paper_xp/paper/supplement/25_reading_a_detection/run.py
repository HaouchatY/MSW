"""Reading a detection: where in scale and colour axis msw.test locates a
difference.  msw.test returns, on every call, the per-level transport z of each
slice family (per-channel DCT, opponent luminance, opponent chroma; level 0 =
finest) as `per_level`; this records it for two CelebA contrasts.

  noise    32-px crops + white noise of std 0.0025
  shading  32-px crops + eps * (unit-std random field in the band 1-3 cycles
           per 32 px), eps = 0.08 -- the 32-px variant the original panel used
House draws: alternative generator 1000 r + 7 (draw, then perturbed draw),
null 70000 + r (two draws).  N = 500, 12 repetitions per side as in the
original.  Saved per repetition: {p, T, per_level}.

    python run.py          # ~3 GPU-minutes
"""
import argparse
import os
import sys
import warnings

import msw
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, alt_seed, celeba, freq_grid, gen, null_seed, random_crops
from protocol import Store

K32 = freq_grid(32)
BAND = ((K32 >= 1 / 32) & (K32 < 3 / 32)).to(DEV)


def noise(b, g):
    return b + 0.0025 * torch.randn(b.shape, device=DEV, generator=g)


def shading(b, g):
    z = torch.randn(b.shape[0], 1, 32, 32, device=DEV, generator=g)
    s = torch.fft.ifft2(torch.fft.fft2(z) * BAND).real
    return b + 0.08 * s / s.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def rep(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = msw.test(a, b)
    return {"p": float(r.p), "T": float(r.T), "per_level": r.per_level}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    cel = celeba()
    store = Store(os.path.join(a.out, "msw.json"))
    for name, f in [("noise", noise), ("shading", shading), ("null", None)]:
        arr = store.cell(name)
        while len(arr) < a.reps:
            r = len(arr)
            g = gen(null_seed(r) if f is None else alt_seed(r))
            x, y = random_crops(cel, a.N, gen=g), random_crops(cel, a.N, gen=g)
            arr.append(rep(x, y if f is None else f(y, g)))
            store.save()
        print(name, "done", flush=True)
