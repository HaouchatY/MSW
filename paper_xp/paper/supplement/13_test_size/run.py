"""Size of msw.test: rejection rate at alpha = 0.05 when both sets come from
the same distribution, on 1000 null repetitions (house seeds 70000 + r) per
draw family and image count.

    python run.py                 # five families x N in {16, 500}, ~20 GPU-minutes
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import run_cell
from draws import gen, null_seed, setting
from ours import null_rep
from protocol import Store

FAMILIES = {"fields": "fields_slope", "t3": "t3_slope", "celeba32": "celeba_blur",
            "celeba64": "celeba_shading", "cifar": "cifar_blur"}   # their null draw

ap = argparse.ArgumentParser()
ap.add_argument("--families", nargs="*", default=list(FAMILIES))
ap.add_argument("--N", type=int, nargs="*", default=[16, 500])
ap.add_argument("--reps", type=int, default=1000)
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
for fam in a.families:
    draw, _ = setting(FAMILIES[fam])
    store = Store(os.path.join(here, "results", f"msw_{fam}.json"))
    for N in a.N:
        run_cell(store, f"{fam}|N{N}|null", a.reps, lambda r: null_rep(draw, N, null_seed(r), gen))
