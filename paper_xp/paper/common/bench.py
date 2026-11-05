"""The nine-setting benchmark of Tables 1 and 2: grids, ladders, and the loop
that measures MSW's power on them."""
import sys
import time

from draws import alt_seed, gen, null_seed, setting
from ours import alt_rep, null_rep
from protocol import rejection_rate

ROWS = ["fields_slope", "fields_noise", "fields_marginal", "t3_slope", "celeba_blur",
        "celeba_noise", "celeba_shading", "cifar_blur", "cifar_checker"]

# Table 1 grids at N = 500.  BASE is the grid every method ran on; FINE adds
# rungs below it, run for MSW only.
BASE = {
    "fields_slope": [0.008, 0.016, 0.032, 0.064, 0.128, 0.256],
    "fields_noise": [0.01, 0.02, 0.04, 0.08, 0.16],
    "fields_marginal": [1 / 48, 1 / 24, 1 / 12, 1 / 6, 1 / 3],
    "t3_slope": [0.016, 0.032, 0.064, 0.128, 0.256],
    "celeba_blur": [0.1, 0.15, 0.22, 0.33, 0.5, 0.75],
    "celeba_noise": [0.0005, 0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04],
    "celeba_shading": [0.04, 0.08, 0.16, 0.32],
    "cifar_blur": [0.1, 0.15, 0.22, 0.33, 0.5],
    "cifar_checker": [0.00125, 0.0025, 0.005, 0.01, 0.02],
}
FINE = {
    "fields_slope": [0.001, 0.002, 0.004],
    "fields_noise": [0.0025, 0.005],
    "fields_marginal": [1 / 384, 1 / 192, 1 / 96],
    "t3_slope": [0.004, 0.008],
    "celeba_blur": [0.033, 0.05, 0.067],
    "celeba_noise": [0.0000625, 0.000125, 0.00025],
    "celeba_shading": [0.01, 0.02],
    "cifar_blur": [0.033, 0.05, 0.067],
    "cifar_checker": [0.0003, 0.000625],
}

# Table 2: the perturbation each row is held at, and the image-count ladder.
LADDER_EPS = {"fields_slope": 0.064, "fields_noise": 0.08, "fields_marginal": 1 / 12,
              "t3_slope": 0.128, "celeba_blur": 0.33, "celeba_noise": 0.0025,
              "celeba_shading": 0.08, "cifar_blur": 0.33, "cifar_checker": 0.01}
LADDER = [8, 16, 32, 64, 128, 256, 512, 1024]
CAPS = {"celeba_noise": 2048}


def grid(row):
    return sorted(BASE[row] + FINE[row])


def key(x):
    return f"{x:.6g}"


def run_cell(store, cell, reps, fn, log=""):
    """Fill store[cell] with (p, T) pairs from fn(r) until it holds `reps`."""
    arr = store.cell(cell)
    t0, n0 = time.time(), len(arr)
    while len(arr) < reps:
        arr.append(list(fn(len(arr))))
        if len(arr) % 8 == 0 or len(arr) == reps:
            store.save()
    if len(arr) > n0:
        dt = (time.time() - t0) / (len(arr) - n0)
        print(f"{log}{cell:40s} reps={len(arr):4d}  power={rejection_rate([p for p, _ in arr]):.3f}"
              f"  ({dt:.2f} s/rep)", flush=True)
    return arr


def measure_row(store, row, N, eps_list, reps, nulls):
    """MSW at one image count over a list of perturbations, plus its null size."""
    draw, alt = setting(row)
    run_cell(store, f"{row}|N{N}|null", nulls,
             lambda r: null_rep(draw, N, null_seed(r), gen))
    for e in eps_list:
        run_cell(store, f"{row}|N{N}|eps{key(e)}", reps,
                 lambda r, f=alt(e): alt_rep(draw, f, N, alt_seed(r), gen))
    sys.stdout.flush()
