"""Table 1, MSW column: power of the exact test at N = 500 on every rung of the
nine perturbation grids (48 repetitions per rung, 48 null repetitions per row).

    python run.py                       # all nine rows (about 1 GPU-hour)
    python run.py --rows celeba_blur    # one row; rows can run in parallel
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import ROWS, grid, measure_row
from protocol import Store

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="*", default=ROWS)
ap.add_argument("--reps", type=int, default=48)
ap.add_argument("--nulls", type=int, default=48)
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
for row in a.rows:
    store = Store(os.path.join(here, "results", f"msw_{row}.json"))
    measure_row(store, row, 500, grid(row), a.reps, a.nulls)
