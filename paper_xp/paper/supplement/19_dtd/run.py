"""Held-out textures (DTD, Cimpoi et al. 2014): MSW on 32-px crops of a dataset
never used in development, in the four designs of the paper's appendix.  Two
disjoint pools of 2000 crops each (h1, h2) are cut from disjoint image sets; a
null repetition compares h1 with h2, an alternative compares h1 with corrupted
h2 (added noise of std eps or exact Gaussian blur of std eps pixels).

  A   pools mixing all 47 categories (first 2000 / next 2000 shuffled images, one
      crop each); noise 0.0025, blur 0.33; N in {8, 16, 25, 50, 100}
  B   single-category pools (woven, marbled, honeycombed; each half of the
      category's images, 2000 random crops per half); same corruptions;
      N in {8, 16, 25, 50}
  C   woven, N = 50: noise eps in {0.005 ... 0.08}, blur eps in {0.5 ... 1.5}
  D   woven, noise 0.02 and blur 1.0 at N in {8, 16, 25}

12 repetitions per cell (house seeds: null 70000 + r, alternative 1000 r + 7),
as in the original exp67 runs.  Saved per repetition: msw.test (p, T) and
msw.distance (value, 95% interval).

    python run.py              # all four designs (~10 GPU-minutes)
    python run.py --designs C D

Data: DTD r1.0.1 extracted at $MSW_XP_DATA/dtd (dtd/images/<category>/*.jpg, 5640
images, https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DATA, DEV, blur, gen
from ours import dist, test
from protocol import Store, rejection_rate

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(DATA, "dtd", "images")
CATS = ["woven", "marbled", "honeycombed"]


def _crop(f, rng):
    from PIL import Image
    im = Image.open(f).convert("RGB")
    w, h = im.size
    x0 = rng.integers(0, w - 32)
    y0 = rng.integers(0, h - 32)
    return torch.from_numpy(np.array(im.crop((x0, y0, x0 + 32, y0 + 32)))).permute(2, 0, 1)


def pools(cat=None):
    """(h1, h2) in [0, 1], 2000 crops each, exactly as exp67 cut them."""
    files = sorted(glob.glob(os.path.join(IMAGES, cat or "*", "*.jpg")))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    out = []
    if cat is None:                       # one crop per image, disjoint image sets
        for fs in (files[:2000], files[2000:4000]):
            out.append(torch.stack([_crop(f, rng) for f in fs]))
    else:                                 # 2000 crops of random images of each half
        h = len(files) // 2
        for fs in (files[:h], files[h:2 * h]):
            out.append(torch.stack([_crop(fs[int(rng.integers(0, len(fs)))], rng) for _ in range(2000)]))
    return [x.float().div(255.0).to(DEV) for x in out]


def pick(h, n, g):
    return h[torch.randperm(h.shape[0], device=DEV, generator=g)[:n]]


@torch.no_grad()
def rep(H1, H2, N, kind, eps, r):
    if kind == "null":
        g = gen(70000 + r)
        a, b = pick(H1, N, g), pick(H2, N, g)
    else:                                 # h2 draw and its corruption first, then h1 (exp67 order)
        g = gen(1000 * r + 7)
        b = pick(H2, N, g)
        b = b + eps * torch.randn(b.shape, device=DEV, generator=g) if kind == "noise" else blur(b, eps)
        a = pick(H1, N, g)
    return list(test(a, b)) + list(dist(a, b))


def cell(store, name, reps, H, N, kind, eps=0.0):
    arr = store.cell(name)
    while len(arr) < reps:
        arr.append(rep(*H, N, kind, eps, len(arr)))
        store.save()
    print(f"{name:28s} reps={len(arr):3d}  rejection={rejection_rate([x[0] for x in arr]):.3f}", flush=True)


DESIGNS = {
    "A": [("mix", None, N, k, e) for N in [8, 16, 25, 50, 100] for k, e in [("noise", 0.0025), ("blur", 0.33)]],
    "B": [(c, c, N, k, e) for c in CATS for N in [8, 16, 25, 50] for k, e in [("noise", 0.0025), ("blur", 0.33)]],
    "C": [("woven", "woven", 50, k, e) for k, g in [("noise", [0.005, 0.01, 0.02, 0.04, 0.08]),
                                                   ("blur", [0.5, 0.75, 1.0, 1.5])] for e in g],
    "D": [("woven", "woven", N, k, e) for N in [8, 16, 25] for k, e in [("noise", 0.02), ("blur", 1.0)]],
}


def name(design, tag, N, kind, eps):
    return f"{design}|{tag}|{kind}{'' if design != 'C' else f'{eps:g}'}|N{N}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="*", default=list(DESIGNS))
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    built = {}
    for d in a.designs:
        store = Store(os.path.join(a.out, f"msw_{d}.json"))
        for tag, cat, N, kind, eps in DESIGNS[d]:
            if tag not in built:
                built[tag] = pools(cat)
            null = f"{d}|{tag}|null|N{N}"
            if null not in store.d or len(store.d[null]) < a.reps:
                cell(store, null, a.reps, built[tag], N, "null")
            cell(store, name(d, tag, N, kind, eps), a.reps, built[tag], N, kind, eps)
