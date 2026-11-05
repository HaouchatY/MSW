"""Figures fig:ssp and fig:ssp1d: what the processes look like.

fig:ssp    s = (D_x D_y)^{-1} w under the near-integrator (a = 0.995 per axis):
           Bernoulli-Laplace innovations at p = 0.004 and 0.02 (64 px), at the
           benchmark's p = 0.05 (32 px), and Gaussian innovations (64 px); one
           generator seeded 21 per panel, each field standardized, grey scale
           [-2.5, 2.5].  -> data/ssp_tile_{0..3}.png
fig:ssp1d  one length-4096 realisation per innovation law (first 1500 samples)
           under the 1D near-integrator, drawn in turn from one generator
           seeded 3.  -> data/ssp1d_traces.csv

    python illustrate.py && pdflatex illustration.tex      # seconds
"""
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from draws import DEV
from ssp import Process, innov

TILES = [("bernlap", 64, 0.004), ("bernlap", 64, 0.02), ("bernlap", 32, 0.05), ("gaussian", 64, 0.0)]
LAWS = ["gaussian", "laplace", "student4", "bernlap_0.05", "bernlap_0.005"]


def field(kind, size, p, a=0.995, seed=21):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = innov("gaussian" if kind == "gaussian" else f"bernlap_{p}", (size, size), g, DEV)
    wx = 2 * math.pi * torch.fft.fftfreq(size, device=DEV)
    hx = 1.0 / (1.0 - a * torch.exp(-1j * wx))
    s = torch.fft.ifft2(torch.fft.fft2(w) * (hx.view(1, size) * hx.view(size, 1))).real
    return (s - s.mean()) / s.std()


if __name__ == "__main__":
    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)
    for i, (k, size, p) in enumerate(TILES):
        f = field(k, size, p).cpu().numpy()
        u8 = np.round(255 * np.clip((f + 2.5) / 5.0, 0, 1)).astype(np.uint8)
        Image.fromarray(u8).resize((256, 256), Image.NEAREST).save(os.path.join(out, f"ssp_tile_{i}.png"))
    proc, g = Process("near", "d1", DEV), torch.Generator(device=DEV).manual_seed(3)
    tr = np.stack([proc.sample(k, 1, g)[0, :1500].cpu().numpy() for k in LAWS], 1)
    np.savetxt(os.path.join(out, "ssp1d_traces.csv"), np.column_stack([np.arange(1500), tr]),
               delimiter=",", header="t," + ",".join(LAWS), comments="", fmt="%.5g")
