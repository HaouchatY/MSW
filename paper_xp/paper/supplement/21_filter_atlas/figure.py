"""Filter atlas: the 25 atoms of msw's 5x5 DCT bank (msw.dct_bank, DC included
here only to be drawn crossed out) and the three opponent channel axes of
msw.opponent_dct_bank (luminance, R-G, B-Y), read back from the bank itself.
Writes data/atlas_<row><col>.png (row = vertical, col = horizontal frequency;
mid-grey = 0, each atom scaled to its own max |w|) and data/axis_<name>.png
(left half +axis, right half -axis around mid-grey), plus data/atlas.csv.

    python figure.py && pdflatex figure.tex      # CPU, seconds
"""
import csv
import os

import numpy as np
import torch
from PIL import Image

import msw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
K, UP = 5, 40


def png(rgb, path, up=UP):
    """rgb in [0, 1], (h, w, 3) -> nearest-upsampled 8-bit PNG."""
    a = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(np.kron(a, np.ones((up, up, 1), dtype=np.uint8))).save(path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    msw.set_device("cpu")
    atoms = msw.dct_bank(K, 1, drop_dc=False)[:, 0].numpy()          # (25, 5, 5), index u*K + v
    rows = []
    for u in range(K):
        for v in range(K):
            w = atoms[u * K + v]
            g = 0.5 + 0.5 * w / np.abs(w).max()
            png(np.repeat(g[:, :, None], 3, 2), os.path.join(OUT, f"atlas_{u}{v}.png"))
            rows += [[u, v, i, j, f"{w[i, j]:.6f}"] for i in range(K) for j in range(K)]
    with open(os.path.join(OUT, "atlas.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["row", "col", "i", "j", "w"]] + rows)
    opp = msw.opponent_dct_bank(K, 3)[:3]                            # first AC atom on the 3 axes
    atom = msw.dct_bank(K, 1)[0, 0]                                  # the same atom, one channel
    for name, filt in zip(["lum", "rg", "by"], opp):
        v = (filt * atom).sum((1, 2)).numpy()                        # channel weights of the axis
        v = v / np.abs(v).max()
        tile = np.stack([0.5 + 0.4 * v, 0.5 - 0.4 * v])[None]        # (1, 2, 3): +axis | -axis
        png(tile, os.path.join(OUT, f"axis_{name}.png"))
        print(name, np.round(v, 3))
