"""Figure fig_edm2 and Table tab:edm2paired: MSW on a public diffusion training
trajectory.  NVIDIA's raw edm2-img512-xs snapshots at 2/6/25/50/100% of training,
1024 images each (shared generation seeds and classes, deterministic sampler),
decoded and evaluated at 256 px against 4096 ImageNet-256 validation images.

unpaired  The draws of the original runs (CPU generators, so they reproduce on any
          device).  Real against snapshot at N = 512: null = two disjoint halves
          of the reals (seeds 70000 + r, 80000 + r), rung = reals (1000 r + 7)
          against the snapshot (2000 r + 11); 50 null / 30 rung repetitions,
          msw.test (p, T), and msw.distance on the first 20 of each (the
          distance ladder).  Adjacent snapshots at N = 384: self-null = the
          earlier snapshot's two halves (70000 + r, 80000 + r), pair = earlier
          first half (1000 r + 7) against later second half (2000 r + 11);
          50 / 30 repetitions, msw.test.
paired    The paired-seed differential test of the original exp60, unchanged:
          for each adjacent pair the 1024 matched images give per-image feature
          differences d_i; T = max over features of |paired t|, calibrated by
          500 sign flips (seed 12345); z = (T - mean_null) / sd_null and the
          sign-flip p.  Our per-image features are every per-image quantity the
          msw.test portfolio reads (msw.components.per_image_u of the DCT and
          chroma-opponent slices at all six levels, msw.components.pixel_u, and
          the product slices).  Control: 50% vs 100% with the pairing rolled by one.

Pools: DATA/sd_pools/edm_k{0033554,0134217,0536870,1073741,2147483}.pt and
DATA/sd_pools/imagenet256_ref.pt.

    python run.py unpaired                       # all cells, ~3 GPU-hours
    python run.py unpaired --cells null rung_2   # a subset; cells run in parallel
    python run.py paired                         # ~15 GPU-minutes
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import run_cell
from draws import DATA
from ours import dist, test
from protocol import Store

import msw
from msw import components as C
from msw.features import SliceFeatures
from msw.portfolio import Portfolio, _joint_u

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MSW_XP_OUT", os.path.join(HERE, "results"))
POOL = os.path.join(DATA, "sd_pools")
TAGS = {2: "0033554", 6: "0134217", 25: "0536870", 50: "1073741", 100: "2147483"}
PCT = list(TAGS)
PAIRS = list(zip(PCT[:-1], PCT[1:]))
N, NS = 512, 384
CELLS = (["null"] + [f"rung_{p}" for p in PCT] + [f"self_{p}" for p in PCT[:-1]]
         + [f"pair_{a}-{b}" for a, b in PAIRS])

_cache = {}


def pool(name):
    """'real' or a percentage; kept on the CPU (the draws use CPU generators)."""
    if name not in _cache:
        snaps = [k for k in _cache if k != "real"]
        if name != "real" and len(snaps) >= 2:      # at most two snapshots in memory
            del _cache[snaps[0]]
        f = "imagenet256_ref.pt" if name == "real" else f"edm_k{TAGS[name]}.pt"
        _cache[name] = torch.load(os.path.join(POOL, f), map_location="cpu")
    return _cache[name]


def dr(p, n, seed):
    return p[torch.randperm(p.shape[0], generator=torch.Generator().manual_seed(seed))[:n]]


def draw(cell, r):
    """The two image sets of repetition r of a cell, exactly as exp59 drew them."""
    kind, _, arg = cell.partition("_")
    if kind == "null":
        real = pool("real"); h = real.shape[0] // 2
        return dr(real[:h], N, 70000 + r), dr(real[h:], N, 80000 + r)
    if kind == "rung":
        return dr(pool("real"), N, 1000 * r + 7), dr(pool(int(arg)), N, 2000 * r + 11)
    if kind == "self":
        p = pool(int(arg))
        return dr(p[:512], NS, 70000 + r), dr(p[512:], NS, 80000 + r)
    a, b = (int(v) for v in arg.split("-"))
    return dr(pool(a)[:512], NS, 1000 * r + 7), dr(pool(b)[512:], NS, 2000 * r + 11)


def unpaired(cells, nulls, alts, dists):
    for cell in cells:
        store = Store(os.path.join(OUT, f"msw_{cell}.json"))
        reps = nulls if cell == "null" or cell.startswith("self") else alts
        run_cell(store, cell, reps, lambda r: test(*draw(cell, r)))
        if cell == "null" or cell.startswith("rung"):
            arr = store.cell(f"{cell}|dist")
            while len(arr) < dists:                 # [d, lo, hi] of msw.distance
                arr.append(list(dist(*draw(cell, len(arr)))))
                store.save()
            print(f"{cell}|dist reps={len(arr)}", flush=True)


# ---------------------------------------------------------------- paired seeds
@torch.no_grad()
def per_image_features(x, port, chunk):
    """(N, S) per-image features: every per-image quantity of the msw.test portfolio."""
    out = []
    fams = [(port.port.base, None), (port.port.opp, port.chroma)]
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk].to(port.device).float()
        cols = []
        for est, cmask in fams:
            f = est.features(xb)
            mats, acc = f.mats, {k: [] for k in C.IMG_SUMS}
            f.mats = []
            for li, q in enumerate(f.per_image):    # one level at a time, in float64
                u = C.per_image_u(SliceFeatures([mats[li].to(C.DT)], [q], xb.shape[0]))
                mats[li] = None
                for k in C.IMG_SUMS:
                    acc[k].append(u[k])
            for k in C.IMG_SUMS:                    # opponent bank: its chroma slices only
                v = torch.cat(acc[k], 0)
                cols.append(v if cmask is None else v[cmask])
            del mats, acc
        cols.append(C.pixel_u(xb))
        cols.append(_joint_u(port.jest, xb))
        out.append(torch.cat(cols, 0).t().cpu().numpy())
    return np.concatenate(out, 0)


def paired_test(fa, fb, flips=500, seed=12345):
    """exp60's test: max paired t over features, sign-flip null."""
    d = fa - fb
    n = d.shape[0]
    se = d.std(0, ddof=1) / np.sqrt(n)
    ok = np.isfinite(se) & (se > 1e-30) & np.isfinite(d.mean(0))
    t_obs = np.abs(d.mean(0)[ok] / se[ok]).max()
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(flips, n))
    tn = np.abs((signs @ d[:, ok]) / n / se[None, ok]).max(1)
    return {"z": float((t_obs - tn.mean()) / tn.std(ddof=1)), "t": float(t_obs),
            "p": float((1.0 + np.sum(tn >= t_obs)) / (1.0 + flips)),
            "rms_effect": float(np.sqrt(np.mean((d.mean(0)[ok] / d.std(0, ddof=1)[ok]) ** 2))),
            "dims": int(ok.sum()), "n_pairs": int(n)}


def paired(chunk):
    feat_dir = os.path.join(OUT, "feats")
    os.makedirs(feat_dir, exist_ok=True)
    port, F = None, {}
    for pc in PCT:
        path = os.path.join(feat_dir, f"edm_{pc}.npy")
        if not os.path.exists(path):
            x = pool(pc)
            if port is None:
                port = Portfolio(3, x.shape[-1], levels=6, k=5, n_filters=None, use_joint=True)
            t0 = time.time()
            np.save(path, per_image_features(x, port, chunk))
            print(f"features {pc}%: ({time.time() - t0:.0f} s)", flush=True)
        F[pc] = np.load(path)
    res = {f"{a}_{b}": paired_test(F[a], F[b]) for a, b in PAIRS}
    res["control_rolled"] = paired_test(F[50], np.roll(F[100], 1, axis=0))
    for k, v in res.items():
        print(f"paired {k:15s} z={v['z']:+.2f}  p={v['p']:.4g}  dims={v['dims']}", flush=True)
    json.dump(res, open(os.path.join(OUT, "msw_paired.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["unpaired", "paired"])
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--nulls", type=int, default=50)
    ap.add_argument("--alts", type=int, default=30)
    ap.add_argument("--dists", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    if a.mode == "unpaired":
        unpaired(a.cells, a.nulls, a.alts, a.dists)
    else:
        paired(a.chunk)
