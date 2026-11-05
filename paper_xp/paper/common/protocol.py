"""Decision rules shared by every experiment, the same for MSW and the baselines.

A baseline detects when its value exceeds the 95th percentile of its own null
repetitions (empirical quantile, 'higher' interpolation); MSW detects when
msw.test rejects at alpha = 0.05.  Power is the detection rate over alternative
repetitions.

  eps_08   the smallest rung of a perturbation grid with power >= 0.8
  n_star   the smallest rung of an image-count ladder with power >= 0.8 that
           keeps power >= 0.8 at every larger rung (one lucky crossing does not
           count)
"""
import json
import math
import os

import numpy as np

ALPHA = 0.05


def q95(null_values):
    return float(np.quantile(np.asarray(null_values, dtype=float), 0.95, method="higher"))


def power(values, threshold):
    return float(np.mean(np.asarray(values, dtype=float) > threshold))


def rejection_rate(pvalues, alpha=ALPHA):
    return float(np.mean(np.asarray(pvalues, dtype=float) <= alpha))


def eps_08(grid, power_at, level=0.8):
    """grid sorted ascending; power_at(eps) -> power or None (rung not run)."""
    for e in grid:
        p = power_at(e)
        if p is not None and p >= level:
            return e
    return None


def n_star(rungs, power_at, level=0.8):
    pw = {N: power_at(N) for N in rungs}
    pw = {N: p for N, p in pw.items() if p is not None}
    for i, N in enumerate(rungs):
        if N in pw and pw[N] >= level and all(pw[M] >= level for M in rungs[i:] if M in pw):
            return N
    return None


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    ph, den = k / n, 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


class Store:
    """A resumable JSON store: every finished repetition is saved at once, so
    a run can be stopped and restarted (or split across processes, one file
    each) without repeating work."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def cell(self, key):
        return self.d.setdefault(key, [])

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f)
        os.replace(tmp, self.path)
