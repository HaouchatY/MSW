"""Figure 3: MSW against a known answer.

key    Eight perturbations of one Gaussian field (32 px, spectrum |k|^-3), each
       calibrated to a known per-pixel W2 in closed form (msw.spectral.w2_exact):
       Nyquist-band boosts (0.025, 0.05), a fine band (0.075), a low band (0.10,
       0.14), a mid band (0.12) and two contrast changes (0.20, 0.28).  N = 500
       per set, 48 repetitions; repetition r draws both sets from one generator
       seeded 1000 r + 7.  Saved per repetition: msw.distance (value, interval)
       and msw.test (p, T).
calib  CelebA 32-px crops under eleven models (identity, contrast, colour cast,
       tone curve, blur), each a gradient of a convex function, so the exact W2
       between two models is the RMS displacement between transformed images.
       55 pairs, N = 800 per set, 16 repetitions (generator 5000 r + 17).

    python run.py key       # ~5 GPU-minutes
    python run.py calib     # ~10 GPU-minutes
"""
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, celeba, gen, random_crops
from ours import dist, test
from protocol import Store
from msw.spectral import freq_grid, powerlaw_spectrum, sample_field, w2_exact

HERE = os.path.dirname(os.path.abspath(__file__))


def key(reps=48, N=500):
    S0 = powerlaw_spectrum(32, 3.0, total_power=1.0).to(DEV)
    KK = freq_grid(32).to(DEV)

    def banded(lo, hi, a):
        s = S0.clone()
        s[(KK >= lo) & (KK < hi)] *= (1.0 + a) ** 2
        return s
    specs = {"nyq025": banded(12 / 32, 10.0, 0.174195),
             "nyq050": banded(12 / 32, 10.0, 0.05 * 32 / 21.0912 ** 0.5),
             "fine075": banded(8 / 32, 12 / 32, 0.426087),
             "low100": banded(1 / 32, 3 / 32, 0.10 * 32 / 803.0028 ** 0.5),
             "mid120": banded(3 / 32, 8 / 32, 0.296107),
             "low140": banded(1 / 32, 3 / 32, 0.158126),
             "con200": S0 * 1.20 ** 2,
             "con280": S0 * 1.28 ** 2}
    store = Store(os.path.join(HERE, "results", "key.json"))
    store.d["w2"] = {k: float(w2_exact(S0, s)[1]) for k, s in specs.items()}
    for k, s in specs.items():
        arr = store.cell(k)
        while len(arr) < reps:
            g = gen(1000 * len(arr) + 7)
            a, b = sample_field(S0, N, generator=g), sample_field(s, N, generator=g)
            arr.append(list(dist(a, b)) + list(test(a, b)))
            store.save()
        print(f"key {k}: {len(arr)} reps", flush=True)


MODELS = [("id", 0.0), ("contrast", 0.02), ("contrast", 0.05), ("contrast", 0.12),
          ("cast", 0.01), ("cast", 0.03), ("gamma", 0.04), ("gamma", 0.12),
          ("blur", 0.30), ("blur", 0.55), ("blur", 0.85)]


def transform(kind, p):
    if kind == "id":
        return lambda x: x
    if kind == "contrast":
        return lambda x: 0.5 + (1.0 + p) * (x - 0.5)
    if kind == "cast":
        return lambda x: x + p
    if kind == "gamma":
        return lambda x: x.clamp_min(1e-6) ** (1.0 + p)
    f = torch.fft.fftfreq(32, device=DEV)
    H = torch.exp(-2 * np.pi ** 2 * p ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    return lambda x: torch.fft.ifft2(torch.fft.fft2(x) * H).real


@torch.no_grad()
def calib(reps=16, N=800):
    cel = celeba()
    T = [transform(k, p) for k, p in MODELS]
    M = len(T)
    store = Store(os.path.join(HERE, "results", "calib.json"))
    store.d["models"] = [f"{k}{p:g}" for k, p in MODELS]
    if "w2" not in store.d:                         # exact W2 on a 4000-crop pool
        pool = random_crops(cel, 4000, gen=gen(0))
        im = [t(pool) for t in T]
        W = np.zeros((M, M))
        for i, j in itertools.combinations(range(M), 2):
            W[i, j] = W[j, i] = float(((im[i] - im[j]) ** 2).sum(dim=(1, 2, 3)).mean().sqrt()
                                      / np.sqrt(3 * 32 * 32))
        store.d["w2"] = W.tolist()
    for r in range(len(store.cell("reps")), reps):
        g = gen(5000 * r + 17)
        X1, X2 = random_crops(cel, N, gen=g), random_crops(cel, N, gen=g)
        A = [t(X1).float().clamp(0, 1) for t in T]
        B = [t(X2).float().clamp(0, 1) for t in T]
        store.cell("reps").append({f"{i},{j}": dist(A[i], B[j])[0]
                                   for i, j in itertools.combinations(range(M), 2)})
        store.save()
        print(f"calib rep {r + 1}/{reps}", flush=True)


if __name__ == "__main__":
    {"key": key, "calib": calib}[sys.argv[1]]()
