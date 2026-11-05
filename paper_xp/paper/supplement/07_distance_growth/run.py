"""Figures S5-S6 (fig:sweep, fig:singlerunnoise), MSW column: msw.distance and
its 95% jackknife interval as the perturbation grows.

Paired design, as for every baseline: repetition r draws the reference set and
the candidate set from one generator seeded 1000 r + 7 at EVERY rung, so within
a rep the reference and the clean part of the candidate are fixed and only eps
moves (eps = 0 is the paired null, two draws).  N = 500, 12 reps on fields slope,
CelebA noise and the confined boost (128-px fields, spectrum |k|^-3 with the band
1-3 cycles per image boosted by (1 + eps)^2 at fixed power); 15 reps on CIFAR
blur, CIFAR checker and CelebA shading; the CelebA-noise ladder again at
N = 1000, 200, 25 (12 reps) for S6.  Saved per rep: (distance, lo, hi).

    python run.py                          # all nine families, ~5 GPU-minutes
    python run.py --families boost         # one family
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, gen, setting
from ours import dist
from protocol import Store
from msw.spectral import freq_grid, powerlaw_spectrum, sample_field

NOISE = [0.0005, 0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04]
FAMILIES = {  # tag -> (draw setting, N, grid, reps)
    "grf_slope": ("fields_slope", 500, [0.008, 0.016, 0.032, 0.064, 0.128], 12),
    "celeba_noise": ("celeba_noise", 500, NOISE, 12),
    "boost": ("boost", 500, [0.04, 0.08, 0.16, 0.32], 12),
    "cifar_blur": ("cifar_blur", 500, [0.1, 0.15, 0.22, 0.33, 0.5], 15),
    "cifar_checker": ("cifar_checker", 500, [0.00125, 0.0025, 0.005, 0.01, 0.02], 15),
    "shading": ("celeba_shading", 500, [0.04, 0.08, 0.16, 0.32], 15),
    "noise_n1000": ("celeba_noise", 1000, NOISE, 12),
    "noise_n200": ("celeba_noise", 200, NOISE, 12),
    "noise_n25": ("celeba_noise", 25, NOISE, 12),
}


def key(x):
    return f"{x:.6g}"


def boost():
    """128-px fields; the band 1/128 <= |k| < 3/128 boosted, total power fixed."""
    S0 = powerlaw_spectrum(128, 3.0, total_power=1.0, device=DEV)
    band = (freq_grid(128, DEV) >= 1 / 128) & (freq_grid(128, DEV) < 3 / 128)

    def boosted(e):
        s = S0.clone()
        s[band] *= (1.0 + e) ** 2
        return s * (S0.sum() / s.sum())
    return (lambda n, g: sample_field(S0, n, generator=g),
            lambda e: lambda n, g: sample_field(boosted(e), n, generator=g))


@torch.no_grad()
def run(tag, reps=None, out=None):
    name, N, grid, R = FAMILIES[tag]
    draw, alt = boost() if name == "boost" else setting(name)
    store = Store(os.path.join(out, f"msw_{tag}.json"))
    for e in [0.0] + grid:
        arr = store.cell(f"{tag}|eps{key(e)}")
        f = draw if e == 0 else alt(e)
        while len(arr) < (reps or R):
            g = gen(1000 * len(arr) + 7)
            a = draw(N, g)
            arr.append(list(dist(a.float(), f(N, g).float())))
            store.save()
        print(f"{tag} eps={key(e)}: {len(arr)} reps", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--reps", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    for t in a.families:
        run(t, a.reps, a.out)
