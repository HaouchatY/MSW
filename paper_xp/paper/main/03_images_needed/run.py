"""Table 2, MSW column: power of the exact test on the image-count ladder
8, 16, ..., 1024 (2048 for CelebA noise) at each row's fixed perturbation
(96 repetitions per rung, 48 null repetitions per rung).

    python run.py                        # all nine rows (about 2 GPU-hours)
    python run.py --rows cifar_checker   # one row; rows can run in parallel
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import CAPS, LADDER, LADDER_EPS, ROWS, measure_row
from protocol import Store

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="*", default=ROWS)
ap.add_argument("--reps", type=int, default=96)
ap.add_argument("--nulls", type=int, default=48)
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
for row in a.rows:
    store = Store(os.path.join(here, "results", f"msw_{row}.json"))
    for N in LADDER + ([CAPS[row]] if row in CAPS else []):
        measure_row(store, row, N, [LADDER_EPS[row]], a.reps, a.nulls)
