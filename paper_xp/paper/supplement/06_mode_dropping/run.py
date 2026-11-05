"""Mode dropping (supplementary figure fig:modedrop), MSW column: msw.test on an
image-count ladder for three ways of dropping half the modes of a mixture.

cifar   CIFAR-10 train against itself with 5 of the 10 classes removed (a random
        5 per repetition, generator 5000 + r on the device)
gender  stationarized CelebA 64 px: disjoint halves of the 16,000 images, every
        image circularly shifted by an independent uniform offset; the second
        set keeps only the women of its half
fields  a uniform mixture of the ten Gaussian field models of the leaderboard
        (32 px, five families x two strengths around the |k|^-3 reference)
        against the same mixture with 5 models removed (CPU generator 5000 + r)

House seeds: null 70000 + r, alternative 1000 r + 7.  200 null and 200 alternative
repetitions per rung (gender: 120 at N = 2048, 60 at N = 4096), as in the original
run.  Saved per repetition: (p, T).  The figure plots the deflection
(mean T_alt - mean T_null) / sd T_null; the text quotes N*, the first rung with
power >= 0.8 that stays there.

    python run.py                                   # everything, ~3 GPU-hours
    python run.py --panels gender --rungs 4096      # one shard; shards run in parallel
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import run_cell
from draws import DEV, celeba, celeba_male, cifar, cifar_labels, gen, null_seed, alt_seed
from ours import test
from protocol import Store
from msw.spectral import freq_grid, powerlaw_spectrum, sample_field

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MD_OUT", os.path.join(HERE, "results"))
RUNGS = {"cifar": [16, 32, 64, 128, 200, 256, 500, 512, 1024, 2000],
         "gender": [64, 128, 256, 512, 1024, 2048, 4096],
         "fields": [16, 32, 64, 128, 200, 256, 500, 512, 1000, 2000]}
REPS = {("gender", 2048): 120, ("gender", 4096): 60}
K = 5                                                    # modes dropped of 10

# the ten leaderboard field models (SW_project/results/exp35_models.json)
MODELS = [("slope", 0.03199996612966046), ("noise", 0.07995424778185137),
          ("aniso", 1.0153060555458073), ("blur", 0.18719060239144827),
          ("contrast", 0.030961747776150534), ("slope", 0.11728913709521256),
          ("noise", 0.16176694011476106), ("aniso", 1.056147089424328),
          ("blur", 0.39317978265354175), ("contrast", 0.11127086954505398)]


def field_spec(family, p, size=32):
    S0 = powerlaw_spectrum(size, 3.0, total_power=1.0, device=DEV)
    KK = freq_grid(size, device=DEV)
    if family == "slope":
        s = S0 * KK.clamp_min(1e-8).pow(-p)
        s[0, 0] = 0.0
        return s * (S0.sum() / s.sum())
    if family == "blur":
        return S0 * torch.exp(-2 * (torch.pi * p * KK) ** 2) ** 2
    if family == "noise":
        return S0 + p ** 2
    if family == "contrast":
        return S0 * (1.0 + p) ** 2
    f = torch.fft.fftfreq(size, device=DEV)
    KX, KY = torch.meshgrid(f, f, indexing="ij")
    s = torch.sqrt((p * KX) ** 2 + (KY / p) ** 2).clamp_min(1e-8).pow(-3.0)
    s[0, 0] = 0.0
    return s * (S0.sum() / s.sum())


def take(x, pool, N, g):
    return x[pool[torch.randperm(pool.shape[0], device=DEV, generator=g)[:N]]]


def cifar_panel():
    x, y = cifar(), cifar_labels()
    allx = torch.arange(x.shape[0], device=DEV)

    def null(N, r):
        g = gen(null_seed(r))
        return take(x, allx, N, g), take(x, allx, N, g)

    def alt(N, r):
        g = gen(alt_seed(r))
        a = take(x, allx, N, g)
        keep = torch.ones(10, dtype=torch.bool, device=DEV)
        keep[torch.randperm(10, device=DEV, generator=gen(5000 + r))[:K]] = False
        return a, take(x, allx[keep[y]], N, g)
    return null, alt


def rshift(t, g):
    """An independent uniform circular shift of every image."""
    n, c, S = t.shape[0], t.shape[1], t.shape[-1]
    sy = torch.randint(0, S, (n,), device=DEV, generator=g)
    sx = torch.randint(0, S, (n,), device=DEV, generator=g)
    ar = torch.arange(S, device=DEV)
    ri, ci = (ar[None, :] - sy[:, None]) % S, (ar[None, :] - sx[:, None]) % S
    t = t.gather(2, ri[:, None, :, None].expand(n, c, S, S))
    return t.gather(3, ci[:, None, None, :].expand(n, c, S, S))


def gender_panel():
    cel, male = celeba(), celeba_male()
    h = cel.shape[0] // 2
    half1, half2 = torch.arange(0, h, device=DEV), torch.arange(h, cel.shape[0], device=DEV)
    fem2 = half2[~male[half2]]

    def null(N, r):
        g = gen(null_seed(r))
        return rshift(take(cel, half1, N, g), g), rshift(take(cel, half2, N, g), g)

    def alt(N, r):
        g = gen(alt_seed(r))
        return rshift(take(cel, half1, N, g), g), rshift(take(cel, fem2, N, g), g)
    return null, alt


def fields_panel():
    specs = [field_spec(f, p) for f, p in MODELS]

    def mixture(sp, n, g):
        assign = torch.randint(0, len(sp), (n,), device=DEV, generator=g)
        out = torch.empty(n, 1, 32, 32, device=DEV)
        for i, s in enumerate(sp):
            m = assign == i
            if int(m.sum()):
                out[m] = sample_field(s, int(m.sum()), generator=g)
        return out

    def null(N, r):
        g = gen(null_seed(r))
        return mixture(specs, N, g), mixture(specs, N, g)

    def alt(N, r):
        g = gen(alt_seed(r))
        a = mixture(specs, N, g)
        drop = set(torch.randperm(10, generator=torch.Generator().manual_seed(5000 + r))[:K].tolist())
        return a, mixture([s for i, s in enumerate(specs) if i not in drop], N, g)
    return null, alt


PANELS = {"cifar": cifar_panel, "gender": gender_panel, "fields": fields_panel}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=list(PANELS))
    ap.add_argument("--rungs", type=int, nargs="*", help="subset of the ladder")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    for panel in a.panels:
        null, alt = PANELS[panel]()
        for N in RUNGS[panel]:
            if a.rungs and N not in a.rungs:
                continue
            store = Store(os.path.join(OUT, f"msw_{panel}_N{N}.json"))     # one file per shard
            reps = min(a.reps, REPS.get((panel, N), a.reps))
            with torch.no_grad():
                run_cell(store, f"{panel}|N{N}|alt", reps, lambda r: test(*alt(N, r)))
                run_cell(store, f"{panel}|N{N}|null", reps, lambda r: test(*null(N, r)))
