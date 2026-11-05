"""Representation or inference: the Portilla-Simoncelli statistic vector run inside
MSW's blocks, pair group and exact test, against MSW's transport rows alone on the
same draws.

P-S vector s(x): plenoptic, 2 scales, 4 orientations, spatial_corr_width 7, on the
channel-mean image.  Both sets are cut into MSW's L = 16 contiguous blocks of
M = N // 16 images; with per-block mean vectors, standardised per coordinate by
the pooled (A u B) image std (orbit-invariant), D_i = s(a_i) - s(b_i).  The group
is MSW's own (msw.pairgroup.patterns_for: the 4096-element interleaved parity
subgroup of (Z2)^16), acting by D_i -> sigma_i D_i.  Two aggregations:
  plain  U = mean_{i != j} <D_i, D_j>, one statistic, rank p on the orbit
  std    per-coordinate U, each divided by its orbit RMS, max over coordinates
         (MSW's own per-component standardisation and max-z combiner)
MSW transport only = msw.test(a, b, portfolio='T_all').  N = 500, 15 repetitions
per rung (generator 1000 r + 7, house draws).  Saved per repetition:
[p_plain, p_std, p_T, T_T].

    python run.py                              # five rows, ~15 GPU-minutes
    python run.py --rows celeba_shading        # rows can run in parallel
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from baselines import ps_stats
from draws import alt_seed, gen, setting
from ours import test
from protocol import Store, rejection_rate
from msw.pairgroup import patterns_for

GRIDS = {"fields_slope": [0.008, 0.016, 0.032, 0.064],
         "fields_marginal": [1 / 192, 1 / 96, 1 / 48, 1 / 24, 1 / 12],
         "celeba_noise": [0.0005, 0.00125, 0.0025, 0.005],
         "celeba_shading": [0.02, 0.04, 0.08, 0.16],
         "cifar_checker": [0.000625, 0.00125, 0.0025, 0.005, 0.01]}
L = 16


def exact_p(D, signs):
    """D: (L, d) block differences -> (p_plain, p_std) on the orbit of `signs`."""
    G = D @ D.T
    G.fill_diagonal_(0.0)
    vals = torch.einsum("gi,ij,gj->g", signs, G, signs) / (L * (L - 1))
    p_plain = float((vals >= vals[0]).double().mean())
    out = D.T[:, :, None] * D.T[:, None, :]                               # (d, L, L)
    out = out * (1 - torch.eye(L, device=D.device, dtype=D.dtype))
    Uc = torch.einsum("gi,dij,gj->dg", signs, out, signs) / (L * (L - 1))   # (d, |G|)
    sd = Uc.pow(2).mean(1).sqrt()
    sd = sd.clamp_min(1e-3 * sd.median().clamp_min(1e-12) + 1e-12)
    Z = (Uc / sd[:, None]).max(0).values
    return p_plain, float((Z >= Z[0]).double().mean())


@torch.no_grad()
def rep(a, b, signs):
    va, vb = ps_stats(a).double(), ps_stats(b).double()
    m = va.shape[0] // L
    sd = torch.cat([va, vb]).std(0).clamp_min(1e-8)
    blk = lambda v: v[:L * m].view(L, m, -1).mean(1)
    pp, ps = exact_p((blk(va) - blk(vb)) / sd, signs)
    return [pp, ps] + list(test(a, b, portfolio="T_all"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=list(GRIDS))
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    for row in a.rows:
        draw, alt = setting(row)
        store = Store(os.path.join(a.out, f"{row}.json"))
        signs = None
        for e in GRIDS[row]:
            arr, t0 = store.cell(f"eps{e:.6g}"), time.time()
            n0 = len(arr)
            while len(arr) < a.reps:
                g = gen(alt_seed(len(arr)))
                x, y = draw(a.N, g), alt(e)(a.N, g)
                if signs is None:
                    signs = patterns_for(min(L, a.N), 4096, "subgroup", device=x.device)
                arr.append(rep(x, y, signs))
                store.save()
            if len(arr) > n0:
                print(f"{row:16s} eps={e:<10.6g} " + "  ".join(
                    f"{k}={rejection_rate([r[i] for r in arr]):.2f}"
                    for i, k in enumerate(["ps_plain", "ps_std", "msw_T"]))
                    + f"  ({(time.time() - t0) / (len(arr) - n0):.1f} s/rep)", flush=True)
