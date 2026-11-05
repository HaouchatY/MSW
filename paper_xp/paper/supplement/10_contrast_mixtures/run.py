"""Moment-matched contrast mixtures (fig:camouflage): MSW on Gaussian fields
whose second set is multiplied per image by an independent contrast nu,
nu^2 = (df - 2) / chi2_df, so E[nu^2] = 1 and the mean, covariance (the whole
spectrum) and all third-order pixel moments match the reference exactly.

Fields: 32 px, 1 channel, spectrum |k|^-3 at unit variance (draws.S0), N = 500
per set.  50 null repetitions (two fields draws, generator 70000 + r) and 50
alternative repetitions per df (fields a, b from generator 1000 r + 7, then nu
from torch's global generator reseeded to that seed, as in the original runs).
Published panel df = 4, 8, 16, 48; the frontier ladder df = 96 ... 3072.
Saved per repetition: msw.test (p, T) and msw.distance (value, interval).

    python run.py                    # panel + frontier, ~30 GPU-minutes
    python run.py --dfs 4 8 16 48    # the published panel only
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, S0, alt_seed, gen, null_seed, sample_field
from ours import dist, test
from protocol import Store

PANEL = [4.0, 8.0, 16.0, 48.0]
FRONTIER = [96.0, 192.0, 384.0, 768.0, 1536.0, 3072.0]
N = 500


def nu_draw(n, df, g):
    """Per-image contrast with E[nu^2] = 1 exactly: nu^2 = (df - 2) / chi2_df."""
    torch.manual_seed(int(g.initial_seed()) % (2 ** 31))
    chi2 = torch.distributions.Chi2(torch.tensor(float(df))).sample((n,)).to(DEV)
    return ((df - 2.0) / chi2.clamp_min(1e-8)).sqrt().view(n, 1, 1, 1)


@torch.no_grad()
def rep(seed, df):
    g = gen(seed)
    a = sample_field(S0, N, generator=g)
    b = sample_field(S0, N, generator=g)
    if df is not None:
        b = b * nu_draw(N, df, g)
    return list(test(a, b)) + list(dist(a, b))         # p, T, d, lo, hi


def fill(store, cell, reps, fn):
    arr = store.cell(cell)
    while len(arr) < reps:
        arr.append(fn(len(arr)))
        store.save()
    print(f"{cell:10s} reps={len(arr)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfs", type=float, nargs="*", default=PANEL + FRONTIER)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    store = Store(os.path.join(a.out, "msw_camo.json"))
    fill(store, "null", a.reps, lambda r: rep(null_seed(r), None))
    for df in a.dfs:
        fill(store, f"df{df:g}", a.reps, lambda r, df=df: rep(alt_seed(r), df))
