"""Tables pooled and pooled-supp: KS-max, kernel MMD, energy distance and a
subsampled W2 on MSW's own pooled slices, each given MSW's L = 16 matched
blocks, pair group (4096-element parity subgroup) and max-z exact test, against
MSW's transport rows alone and the shipped CORE portfolio (= msw.test).

Six settings of the house benchmark, N = 500, 24 repetitions per rung (house
seeds: alternative 1000 r + 7, null 70000 + r).  Every repetition stores
(p, T) of every method (pooled.Bench).

  grid   K = 128 points per block per slice (2048 per side): every rung of the
         extended grids, the rivals also with TL + PX + J kept (hybrids); the
         previous table's cells at N = 16 / 64 / 256; null sizes (fields slope
         N = 500 and 64, 400 each; CelebA noise and CIFAR checker N = 500, 700)
  k512   K = 512 (8192 per side) on the rungs that can move a rival's eps_0.8
  clamp  CIFAR checker without the [0, 1] clamp, against the clamped alternative

    python run.py grid                           # ~5 GPU-hours, all six settings
    python run.py grid --settings celeba_blur    # one setting per process
    python run.py k512                           # ~2 GPU-hours
    python run.py clamp                          # ~15 GPU-minutes
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, HERE)
from bench import key
from draws import DEV, alt_seed, cifar, gen, null_seed, setting
from pooled import Bench
from protocol import Store

SETTINGS = ["fields_slope", "fields_marginal", "celeba_noise", "celeba_blur",
            "celeba_shading", "cifar_checker"]
GRID = {"fields_slope": [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256],
        "fields_marginal": [1 / 384, 1 / 192, 1 / 96, 1 / 48, 1 / 24, 1 / 12, 1 / 6, 1 / 3],
        "celeba_noise": [0.0000625, 0.000125, 0.00025, 0.0005, 0.00125, 0.0025, 0.005, 0.01,
                         0.02, 0.04],
        "celeba_blur": [0.033, 0.05, 0.067, 0.1, 0.15, 0.22, 0.33, 0.5, 0.75],
        "celeba_shading": [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
        "cifar_checker": [0.00015625, 0.0003125, 0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02]}
PAPER = {"fields_slope": (0.064, [16, 64, 256]), "fields_marginal": (1 / 12, [16, 64, 256]),
         "celeba_noise": (0.0025, [16, 64, 256])}
NULLS = {"fields_slope": [(500, 400), (64, 400)], "celeba_noise": [(500, 700)],
         "cifar_checker": [(500, 700)]}
K512 = {"fields_slope": [0.008, 0.016, 0.032, 0.064],
        "fields_marginal": [1 / 192, 1 / 96, 1 / 48, 1 / 24, 1 / 12, 1 / 6],
        "celeba_noise": [0.00025, 0.0005, 0.00125, 0.0025],
        "celeba_blur": [0.067, 0.1, 0.15, 0.22, 0.33],
        "celeba_shading": [0.02, 0.04, 0.08, 0.16],
        "cifar_checker": [0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02]}
CLAMP_EPS = [0.0, 0.000625, 0.00125, 0.0025]


def fill(store, cell, reps, fn):
    arr = store.cell(cell)
    while len(arr) < reps:
        arr.append(fn(len(arr)))
        store.save()
    print(f"{cell:44s} reps={len(arr)}  MSW={sum(r['MSW_CORE'][0] <= 0.05 for r in arr) / len(arr):.2f}"
          f"  ks={sum(r['ks'][0] <= 0.05 for r in arr) / len(arr):.2f}", flush=True)


def shape(draw):
    x = draw(2, gen(0))
    return x.shape[1], x.shape[-1]


@torch.no_grad()
def phase_grid(s, reps, out, nulls=None):
    draw, alt = setting(s)
    bn = Bench(*shape(draw), Ksub=128)
    st = Store(os.path.join(out, f"grid_{s}.json"))
    pair = lambda N, f: lambda r: (lambda g: bn.run(draw(N, g), f(N, g)))(gen(alt_seed(r)))
    for e in GRID[s]:
        fill(st, f"{s}|N500|eps{key(e)}", reps, pair(500, alt(e)))
    eps, Ns = PAPER.get(s, (None, []))
    for N in Ns:
        fill(st, f"{s}|N{N}|eps{key(eps)}", reps, pair(N, alt(eps)))
    for N, R in NULLS.get(s, []):
        fill(st, f"{s}|N{N}|null", nulls or R,
             lambda r, N=N: (lambda g: bn.run(draw(N, g), draw(N, g)))(gen(null_seed(r))))


@torch.no_grad()
def phase_k512(s, reps, out):
    draw, alt = setting(s)
    bn = Bench(*shape(draw), Ksub=512)
    st = Store(os.path.join(out, f"k512_{s}.json"))
    for e in K512[s]:
        fill(st, f"{s}|N500|eps{key(e)}", reps,
             lambda r, f=alt(e): (lambda g: bn.run(draw(500, g), f(500, g), hybrids=False))(gen(alt_seed(r))))


@torch.no_grad()
def phase_clamp(reps, out):
    draw, _ = setting("cifar_checker")
    cifar()
    r_ = torch.arange(32, device=DEV)
    P = ((-1.0) ** (r_[:, None] + r_[None, :])).view(1, 1, 32, 32)
    alt = {"paper": lambda e: lambda n, g: (draw(n, g) + e * P).clamp(0, 1),
           "noclamp": lambda e: lambda n, g: draw(n, g) + e * P}
    bn = Bench(3, 32, Ksub=128)
    st = Store(os.path.join(out, "clamp.json"))
    for v in ("paper", "noclamp"):
        for e in CLAMP_EPS:
            f = draw if e == 0 else alt[v](e)
            fill(st, f"{v}|N500|eps{key(e)}", reps,
                 lambda r, f=f: (lambda g: bn.run(draw(500, g), f(500, g), hybrids=False))(gen(alt_seed(r))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["grid", "k512", "clamp"])
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--nulls", type=int, default=None, help="override the null counts")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    if a.phase == "clamp":
        phase_clamp(a.reps, a.out)
    else:
        for s in a.settings:
            if a.phase == "grid":
                phase_grid(s, a.reps, a.out, a.nulls)
            else:
                phase_k512(s, a.reps, a.out)
