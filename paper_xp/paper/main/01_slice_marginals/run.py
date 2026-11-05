"""Figure 1: what one slice sees.  The pooled marginal of a single high-frequency
slice -- the 5x5 DCT atom (4,4) at full resolution, convolved with every image and
pooled over all positions -- under six perturbation families, three strengths
each.  500 images per curve, histograms normalised by the clean curve's std.
The CelebA-noise panel uses the same atom on the R-G chroma axis.

    python run.py          # seconds on a GPU; writes data/margrow_*.csv
"""
import io
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DATA, DEV, S0, blur, celeba, gen, perturb, sample_field

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N = 500


def dct1d(k):
    n = torch.arange(k, dtype=torch.float64)
    B = torch.stack([torch.cos(math.pi * (n + 0.5) * a / k) for a in range(k)])
    B[0] /= math.sqrt(2.0)
    return B * math.sqrt(2.0 / k)


B = dct1d(5)
ATOM = (B[4][:, None] * B[4][None, :]).float().to(DEV)
LUM = ATOM.view(1, 1, 5, 5)
RG = (torch.tensor([1.0, -1.0, 0.0], device=DEV).view(3, 1, 1) / math.sqrt(2.0) * ATOM).view(1, 3, 5, 5)
RG = RG / RG.norm()


def responses(x, w=LUM):
    return F.conv2d(x, w).flatten().cpu().numpy()


def panel(name, curves, lim, bins=201):
    s = curves[0].std()
    cols = []
    for v in curves:
        h, e = np.histogram(v / s, bins=bins, range=(-lim, lim), density=True)
        cols.append(h)
    x = 0.5 * (e[1:] + e[:-1])
    np.savetxt(os.path.join(OUT, f"margrow_{name}.csv"), np.column_stack([x] + cols), delimiter=",",
               header="x," + ",".join(f"c{i}" for i in range(len(curves))), comments="")


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    panel("grf_slope", [responses(sample_field(perturb("slope", e), N, generator=gen(20 + i)))
                        for i, e in enumerate([0.0, 0.256, 1.0])], lim=5)
    panel("t3_slope", [responses(sample_field(perturb("slope", e), N, marginal="student", df=3.0,
                                              generator=gen(30 + i)))
                       for i, e in enumerate([0.0, 0.256, 1.0])], lim=8)
    panel("grf_marginal", [responses(sample_field(S0, N, generator=gen(40))),
                           responses(sample_field(S0, N, marginal="student", df=8.0, generator=gen(41))),
                           responses(sample_field(S0, N, marginal="student", df=3.0, generator=gen(42)))],
          lim=8)
    cel = celeba()[:N, :, 16:48, 16:48]                      # central 32 px of the first 500 faces
    panel("cel_noise", [responses(cel, RG)] +
          [responses(cel + s * torch.randn(cel.shape, device=DEV, generator=gen(50 + i)), RG)
           for i, s in enumerate([0.0025, 0.005])], lim=12, bins=241)
    X = cel[:, 1:2]                                          # green channel
    panel("cel_blur", [responses(X)] + [responses(blur(X, s)) for s in [0.33, 0.75]], lim=5, bins=241)
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(os.path.join(DATA, "cifar10_train.parquet"))
    arr = np.stack([np.array(Image.open(io.BytesIO(r["bytes"]))) for r in t.column("img").to_pylist()[:N]])
    Y = torch.from_numpy(arr).float().permute(0, 3, 1, 2)[:, 1:2].to(DEV) / 255.0
    r = torch.arange(32, device=DEV)
    P = ((-1.0) ** (r[:, None] + r[None, :])).view(1, 1, 32, 32)
    panel("cif_checker", [responses(Y)] + [responses((Y + e * P).clamp(0, 1)) for e in [0.01, 0.02]],
          lim=10, bins=241)


if __name__ == "__main__":
    main()
