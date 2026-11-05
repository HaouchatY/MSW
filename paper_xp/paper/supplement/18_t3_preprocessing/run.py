"""The t3 row on RAW inputs: Table 1's t3 row standardizes every image (zero
mean, unit variance) before every metric; this measures the same row without
that step, so the effect of the preprocessing can be read off.

Fields as in Table 1 (32 px, spectrum |k|^-3, Student-t(3) scale mixture,
slope perturbation), N = 500, house seeds, same grid.

    python run.py              MSW: msw.test, 48 repetitions per rung and 48 null
                               repetitions (~6 GPU-minutes)
    python run.py baselines    the eleven baseline values (common/baselines.py),
                               100 null and 12 alternative repetitions, as for
                               Table 1 (~40 GPU-minutes; optional -- the archived
                               raw run is exp33, see results/baselines.json)

The standardized row is Table 1's: main/02_detection_thresholds/results.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import BASE, grid, key, run_cell
from draws import S0, alt_seed, gen, null_seed, perturb, sample_field
from ours import alt_rep, null_rep
from protocol import Store

HERE = os.path.dirname(os.path.abspath(__file__))
N = 500


def draw(n, g):
    return sample_field(S0, n, marginal="student", df=3.0, generator=g)


def alt(e):
    return lambda n, g: sample_field(perturb("slope", e), n, marginal="student", df=3.0, generator=g)


def ours(out, reps, nulls):
    store = Store(os.path.join(out, "msw_raw.json"))
    run_cell(store, f"t3raw|N{N}|null", nulls, lambda r: null_rep(draw, N, null_seed(r), gen))
    for e in grid("t3_slope"):
        run_cell(store, f"t3raw|N{N}|eps{key(e)}", reps,
                 lambda r, f=alt(e): alt_rep(draw, f, N, alt_seed(r), gen))


@torch.no_grad()
def baselines(out, reps, nulls):
    from baselines import make_measure
    measure = make_measure(draw, 1, 32)
    store = Store(os.path.join(out, "baselines_raw_rerun.json"))

    def fill(cell, n, seed, b_fn):
        arr = store.cell(cell)
        while len(arr) < n:
            g = gen(seed(len(arr)))
            a = draw(N, g)
            arr.append(measure(a, b_fn(N, g), len(arr)))
            store.save()
        print(f"{cell}: {len(arr)}", flush=True)
    fill(f"t3raw|N{N}|null", nulls, null_seed, draw)
    for e in BASE["t3_slope"]:
        fill(f"t3raw|N{N}|eps{key(e)}", reps, alt_seed, alt(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="ours", choices=["ours", "baselines"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--nulls", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    if a.what == "ours":
        ours(a.out, a.reps or 48, a.nulls or 48)
    else:
        baselines(a.out, a.reps or 12, a.nulls or 100)
