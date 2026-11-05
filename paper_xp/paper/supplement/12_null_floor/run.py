"""Zero floor (fig_null), MSW series: msw.distance on two draws of the SAME
distribution.  The floor-corrected U-statistic u = sign(d) d^2 behind the
distance is centred at zero under the null (Proposition 1), at every N.

300 null repetitions per family and N in {8, 16, 64, 500}; repetition r draws
both sets from one generator seeded 70000 + r.  Families:
  fields      Gaussian fields, 32 px, spectrum |k|^-3 (the house null draw)
  celeba32    32-px random crops, A from the first and B from the second fixed
              half of the 16,000 CelebA images
  celeba32rs  as celeba32, but the 16,000 images are split into two halves by a
              fresh random permutation in every repetition
Saved per repetition: msw.distance (signed value, 95% jackknife interval).

    python run.py                          # all three families, ~15 GPU-minutes
    python run.py --families celeba32rs    # families can run in parallel
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, S0, celeba, gen, null_seed, random_crops, sample_field
from ours import dist
from protocol import Store

FAMILIES = ["fields", "celeba32", "celeba32rs"]


def pair(fam, N, g):
    if fam == "fields":
        return sample_field(S0, N, generator=g), sample_field(S0, N, generator=g)
    cel = celeba()
    m = cel.shape[0] // 2
    if fam == "celeba32":
        return random_crops(cel[:m], N, gen=g), random_crops(cel[m:], N, gen=g)
    perm = torch.randperm(cel.shape[0], device=DEV, generator=g)
    return random_crops(cel[perm[:m]], N, gen=g), random_crops(cel[perm[m:]], N, gen=g)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--N", type=int, nargs="*", default=[8, 16, 64, 500])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    for fam in a.families:
        store = Store(os.path.join(a.out, f"msw_{fam}.json"))
        for N in a.N:
            arr, t0 = store.cell(f"{fam}|N{N}"), time.time()
            n0 = len(arr)
            while len(arr) < a.reps:
                arr.append(list(dist(*pair(fam, N, gen(null_seed(len(arr)))))))
                if len(arr) % 25 == 0 or len(arr) == a.reps:
                    store.save()
            if len(arr) > n0:
                print(f"{fam:11s} N={N:4d} reps={len(arr)} ({(time.time() - t0) / (len(arr) - n0):.2f} s/rep)",
                      flush=True)


if __name__ == "__main__":
    main()
