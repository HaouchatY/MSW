"""Leaderboard with an answer key (supplementary figure): ten stationary Gaussian
field 'models', two severities in each of five spectral families (slope, noise,
anisotropy, blur, contrast), tuned so their exact per-pixel W2 to the reference
(msw.spectral.w2_exact) is log-spaced over one decade with adjacent ranks in
different families.  Per repetition and N, a fresh reference set and one set per
model, all drawn in order from one generator seeded 10000 rep + N (exp35's draws);
15 repetitions, N in {50, 100, 250, 1000}.  Saved per (rep, N): msw.test (p, T)
and msw.distance (value, interval) for the ten models.

--classical  also reruns the classical stationary baselines on the same draws
             (spectrum distance, Karras pSWD, Portilla-Simoncelli; common/baselines.py),
             whose original per-repetition values were not archived.

    python run.py                    # ours, ~20 GPU-minutes
    python run.py --ns 1000          # one N; N values can run in parallel
    python run.py --classical        # + spec/pSWD/P-S, ~15 more GPU-minutes (plenoptic)
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, gen
from ours import dist, test
from protocol import Store
from msw.spectral import freq_grid, powerlaw_spectrum, sample_field, w2_exact

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 32
S0 = powerlaw_spectrum(SIZE, 3.0, total_power=1.0, device=DEV)
KK = freq_grid(SIZE, device=DEV)
_f = torch.fft.fftfreq(SIZE, device=DEV)
KX, KY = torch.meshgrid(_f, _f, indexing="ij")


def spec(family, p):
    if family == "slope":
        s = S0 * KK.clamp_min(1e-8).pow(-p)
        s[0, 0] = 0.0
        return s * (S0.sum() / s.sum())
    if family == "blur":
        return S0 * torch.exp(-2 * (math.pi * p * KK) ** 2) ** 2
    if family == "noise":
        return S0 + p ** 2
    if family == "contrast":
        return S0 * (1.0 + p) ** 2
    if family == "aniso":
        s = torch.sqrt((p * KX) ** 2 + (KY / p) ** 2).clamp_min(1e-8).pow(-3.0)
        s[0, 0] = 0.0
        return s * (S0.sum() / s.sum())
    raise ValueError(family)


def models():
    """exp35's ladder: bisect each severity onto its target W2bar."""
    ladder = ["slope", "noise", "aniso", "blur", "contrast"] * 2
    bounds = {"slope": (1e-4, 2.0), "blur": (1e-3, 2.0), "noise": (1e-3, 1.0),
              "contrast": (1e-4, 1.0), "aniso": (1.0 + 1e-4, 3.0)}
    t0 = w2_exact(S0, spec("slope", 0.032))[1]
    out = []
    for j, fam in enumerate(ladder):
        lo, hi = bounds[fam]
        for _ in range(50):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if w2_exact(S0, spec(fam, mid))[1] < t0 * 10 ** (j / 9) else (lo, mid)
        p = math.sqrt(lo * hi)
        out.append({"rank": j, "family": fam, "param": p, "w2bar_true": w2_exact(S0, spec(fam, p))[1]})
    return out


def classical_scorer():
    from baselines import PSDist, karras_swd, spec_dist
    draw = lambda n, g: sample_field(S0, n, generator=g)
    psd = PSDist(draw)
    pool = draw(500, gen(1))
    mu, sd = float(pool.mean()), float(pool.std())
    to01 = lambda t: ((t - (mu - 4 * sd)) / (8 * sd)).clamp(0, 1).expand(-1, 3, -1, -1)

    def score(a, b, rep):
        return {"spec": spec_dist(a, b), "pswd": karras_swd(to01(a), to01(b), rep), "ps": psd(a, b)}
    return score


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="*", default=[50, 100, 250, 1000])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    torch.manual_seed(0)
    ms = models()
    specs = [spec(m["family"], m["param"]) for m in ms]
    tag = "" if a.ns == [50, 100, 250, 1000] else "_N" + "_".join(map(str, a.ns))
    ours = Store(os.path.join(a.out, f"msw{tag}.json"))
    ours.d["models"] = ms
    cls = Store(os.path.join(a.out, f"classical{tag}.json")) if a.classical else None
    score = classical_scorer() if a.classical else None
    for rep in range(a.reps):
        for N in a.ns:
            key = f"N{N}|rep{rep}"
            if key in ours.d and "d_flat" in ours.d[key] and (cls is None or key in cls.d):
                continue
            g = gen(10_000 * rep + N)
            ref = sample_field(S0, N, generator=g)
            sets = [sample_field(s, N, generator=g) for s in specs]
            if key not in ours.d:
                pt = [test(ref, b) for b in sets]
                dd = [dist(ref, b) for b in sets]
                ours.d[key] = {"p": [x[0] for x in pt], "T": [x[1] for x in pt],
                               "d": [x[0] for x in dd], "lo": [x[1] for x in dd],
                               "hi": [x[2] for x in dd]}
                ours.save()
            if "d_flat" not in ours.d[key]:              # the 0.2.x distance, for comparison
                ours.d[key]["d_flat"] = [dist(ref, b, weighting="flat", signed=False)[0] for b in sets]
                ours.save()
            if cls is not None and key not in cls.d:
                v = [score(ref, b, rep) for b in sets]
                cls.d[key] = {m: [x[m] for x in v] for m in ("spec", "pswd", "ps")}
                cls.save()
            print(f"{key} done", flush=True)


if __name__ == "__main__":
    main()
