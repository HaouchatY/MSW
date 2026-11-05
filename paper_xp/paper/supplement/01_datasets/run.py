"""Figure S1: the nine settings, one draw per column, clean and perturbed.

Display strengths are chosen to be visible by eye, far above the detection
thresholds of Table 1: fields slope +1.0, noise 0.32, Gaussian -> t3 marginal
(the field rescaled by the median t3 mixing scale), t3 field slope +1.0; CelebA
blur 0.75, noise 0.08 (32-px centre crops), shading 0.32 (64 px); CIFAR-10 blur
0.75 and checkerboard 0.02.  Each field column shows ONE draw (generator seeded
11-14) clean and perturbed with the same Fourier phases; CelebA shows images
0, 1, 2; CIFAR-10 a detailed image (rank 220 of 300 by gradient energy) for blur
and a smooth one (rank 3) for the checkerboard.  Tiles are 32 or 64 px,
upsampled 8x / 4x by pixel repetition so no viewer smooths them.

    python run.py          # seconds; writes data/c{1..9}_{clean,pert}.png
"""
import argparse
import math
import os
import sys

import numpy as np
import scipy.stats as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, S0, blur, celeba, cifar, freq_grid, gen, perturb


def field_pair(seed, s_pert, nu_clean=1.0, nu_pert=None):
    """One field under S0 and under s_pert, same phases."""
    z = torch.randn(32, 32, device=DEV, generator=gen(seed), dtype=torch.cfloat)
    f = lambda s: (torch.fft.ifft2(z * s.sqrt()).real * (32 * math.sqrt(2.0))).cpu().numpy()
    return f(S0) * nu_clean, f(s_pert) * (nu_clean if nu_pert is None else nu_pert)


def save(path, v):
    """v: (H, W) or (H, W, 3) in [0, 1]; pixel-repeated to 256 px."""
    import matplotlib.pyplot as plt
    r = 256 // v.shape[0]
    v = np.clip(v, 0, 1).repeat(r, 0).repeat(r, 1)
    plt.imsave(path, v, cmap="gray" if v.ndim == 2 else None, vmin=0.0, vmax=1.0)


def gray_pair(out, i, xc, xp):
    lo, hi = np.percentile(xc, 1), np.percentile(xc, 99)       # clean tile's range, padded 10%
    lo, hi = lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)
    for tag, x in (("clean", xc), ("pert", xp)):
        save(os.path.join(out, f"c{i}_{tag}.png"), (x - lo) / (hi - lo))


def rgb_pair(out, i, xc, xp):
    for tag, x in (("clean", xc), ("pert", xp)):
        save(os.path.join(out, f"c{i}_{tag}.png"), x.permute(1, 2, 0).cpu().numpy())


@torch.no_grad()
def main(out):
    os.makedirs(out, exist_ok=True)
    gray_pair(out, 1, *field_pair(11, perturb("slope", 1.0)))
    gray_pair(out, 2, *field_pair(12, perturb("noise", 0.32)))
    gray_pair(out, 3, *field_pair(13, S0, nu_pert=1.0 / math.sqrt(st.chi2.ppf(0.5, 3))))
    torch.manual_seed(14)                                       # the t3 field's one mixing scale
    s = math.sqrt(3.0 / float(torch.distributions.Gamma(1.5, 0.5).sample(()))) / math.sqrt(3.0)
    gray_pair(out, 4, *field_pair(14, perturb("slope", 1.0), nu_clean=s))

    cel = celeba()
    crop = lambda j: cel[j][:, 16:48, 16:48]
    rgb_pair(out, 5, crop(0), blur(crop(0), 0.75))
    rgb_pair(out, 6, crop(1), crop(1) + 0.08 * torch.randn(3, 32, 32, device=DEV, generator=gen(6)))
    k64 = freq_grid(64)
    z = torch.randn(1, 1, 64, 64, device=DEV, generator=gen(7))
    sh = torch.fft.ifft2(torch.fft.fft2(z) * ((k64 >= 1 / 64) & (k64 < 3 / 64))).real
    sh = sh / sh.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    rgb_pair(out, 7, cel[2], cel[2] + 0.32 * sh[0])

    X = cifar()[:300]
    d = X.mean(1)
    hf = ((d[:, :, 1:] - d[:, :, :-1]).abs().mean((1, 2)) + (d[:, 1:] - d[:, :-1]).abs().mean((1, 2))).cpu()
    order = np.argsort(hf.numpy(), kind="stable")
    i_blur, i_smooth = int(order[220]), int(order[3])
    rgb_pair(out, 8, X[i_blur], blur(X[i_blur], 0.75))
    r = torch.arange(32, device=DEV)
    P = ((-1.0) ** (r[:, None] + r[None, :])).view(1, 32, 32)
    rgb_pair(out, 9, X[i_smooth], (X[i_smooth] + 0.02 * P).clamp(0, 1))
    print(f"t3 scale {s:.3f}; CIFAR blur image {i_blur}, checker image {i_smooth}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    main(ap.parse_args().out)
