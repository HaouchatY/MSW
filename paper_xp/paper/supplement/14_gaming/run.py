"""Gaming: can a metric be fooled without fixing the images?  Blurred CIFAR-10
(Gaussian blur, std 0.5 px) is scored against clean CIFAR-10 at N = 512 per set.

select  The selection attack of Kynkaanniemi et al. (2023): from 20,000 blurred
        candidates pick 1024 whose Inception k-means cluster histogram (k = 32,
        fitted on a disjoint 2048-image reference) matches the reference; the
        control is 1024 random candidates.  No pixel changes.
hb      A Heeger-Bergen attack aimed at MSW itself: alternately match the pooled
        pixel marginal and every pooled sub-band marginal of MSW's own 5x5 DCT
        bank on a 4-level Laplacian pyramid (60 iterations, 1024 images), from
        uniform noise or from 1024 blurred candidates.
score   msw.test of the clean evaluation pool against each pool: null (a disjoint
        clean pool), random and matched selections, and both HB attacks before
        and after; 200 repetitions per cell, house seeds (null 70000 + r, else
        1000 r + 7).  Saved per repetition: p, T and every CORE component.

Pools: one CUDA permutation (seed 0) of CIFAR-10 train gives ref_fit (2048),
ref_eval (2048), null_pool (4096) and 20,000 candidates, as in the original run.

    python run.py select        # Inception features + k-means, ~3 GPU-minutes
    python run.py hb noise      # ~20 GPU-minutes each; noise and blur can run in parallel
    python run.py hb blur
    python run.py score         # 7 cells x 200 reps, ~15 GPU-minutes
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DEV, blur, cifar, gen
from protocol import Store
import msw
from msw.banks import Pyramid, dct_bank

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
WORK = os.path.join(HERE, "work")                     # attacked images (not versioned)
N, NCAND, NSEL, K, BLUR = 512, 20000, 1024, 32, 0.5
NSYN, LEVELS, ITERS, DT = 1024, 4, 60, torch.float64


def pools():
    """The four pools of the original run, from one permutation on DEV."""
    x = cifar()
    p = torch.randperm(x.shape[0], device=DEV, generator=gen(0))
    return {"ref_fit": x[p[:2048]], "ref_eval": x[p[2048:4096]], "null_pool": x[p[4096:8192]],
            "cand": blur(x[p[8192:8192 + NCAND]], BLUR)}


# ------------------------------------------------------------- selection attack
@torch.no_grad()
def select():
    from baselines import inception
    P = pools()
    f_ref = torch.from_numpy(inception(P["ref_fit"])).float()
    f_cand = inception(P["cand"])
    gk = torch.Generator().manual_seed(0)                # k-means++ init, 50 Lloyd steps
    cidx = [int(torch.randint(f_ref.shape[0], (1,), generator=gk))]
    for _ in range(K - 1):
        dmin = torch.cdist(f_ref, f_ref[cidx]).min(1).values ** 2
        cidx.append(int(torch.multinomial(dmin / dmin.sum(), 1, generator=gk)))
    C = f_ref[cidx].clone()
    for _ in range(50):
        lab = torch.cdist(f_ref, C).argmin(1)
        for c in range(K):
            if (lab == c).any():
                C[c] = f_ref[lab == c].mean(0)
    lab_ref, cc = lab.numpy(), C.numpy()
    lab_cand, dist_cand = np.empty(len(f_cand), dtype=int), np.empty(len(f_cand))
    for i in range(0, len(f_cand), 2000):
        dd = ((f_cand[i:i + 2000, None, :] - cc[None]) ** 2).sum(-1)
        lab_cand[i:i + 2000], dist_cand[i:i + 2000] = dd.argmin(1), dd.min(1)
    target = np.bincount(lab_ref, minlength=K) / len(lab_ref)
    want = np.round(target * NSEL).astype(int)
    while want.sum() != NSEL:
        want[np.argmax(target)] += int(np.sign(NSEL - want.sum()))
    rng, idx, short = np.random.default_rng(0), [], 0
    for c in range(K):
        pc = np.where(lab_cand == c)[0]
        if len(pc) >= want[c]:
            idx += rng.choice(pc, want[c], replace=False).tolist()
        else:
            idx += pc.tolist()
            short += want[c] - len(pc)
    if short:                                            # fill from the closest leftovers
        left = np.setdiff1d(np.arange(len(f_cand)), np.array(idx))
        idx += left[np.argsort(dist_cand[left])[:short]].tolist()
    st = Store(os.path.join(OUT, "selection.json"))
    st.d.update(sel=sorted(int(i) for i in idx),
                ctrl=sorted(np.random.default_rng(1).choice(NCAND, NSEL, replace=False).tolist()),
                short=int(short), want=want.tolist())
    st.save()
    print(f"selection: {NSEL} matched ({short} filled by distance), {NSEL} random", flush=True)


# ------------------------------------------------------------- Heeger-Bergen attack
B = dct_bank(5, 1, drop_dc=False).to(DEV).to(DT)      # orthonormal: a tight frame, A^T A = 25 I
NF = B.shape[0]
G2 = Pyramid(levels=LEVELS).g2.to(DEV).to(DT)
NQ = 65536


def analyse(x):
    n, c, h, w = x.shape
    return F.conv2d(F.pad(x.reshape(n * c, 1, h, w), (2,) * 4, mode="circular"), B).reshape(n, c, NF, h, w)


def synth(r):
    n, c, f, h, w = r.shape
    Bf = torch.flip(B, dims=(-2, -1)).permute(1, 0, 2, 3)
    return F.conv2d(F.pad(r.reshape(n * c, f, h, w), (2,) * 4, mode="circular"), Bf).reshape(n, c, h, w) / NF


def _blur5(x, scale=1.0):
    return F.conv2d(F.pad(x, (2,) * 4, mode="circular"), (G2 * scale).expand(x.shape[1], 1, 5, 5),
                    groups=x.shape[1])


def expand(y, size):
    n, c, h, w = y.shape
    up = torch.zeros(n, c, 2 * h, 2 * w, device=y.device, dtype=y.dtype)
    up[:, :, ::2, ::2] = y
    return _blur5(up[:, :, :size, :size], 4.0)


def lap(x):
    G = [x]
    for _ in range(LEVELS - 1):
        G.append(_blur5(G[-1])[:, :, ::2, ::2])
    return [G[l] - expand(G[l + 1], G[l].shape[-1]) for l in range(LEVELS - 1)] + [G[-1]]


def collapse(L):
    cur = L[-1]
    for l in range(len(L) - 2, -1, -1):
        cur = L[l] + expand(cur, L[l].shape[-1])
    return cur


def _interp(s, idx):
    lo = idx.floor().long().clamp(0, s.numel() - 1)
    hi = (lo + 1).clamp(max=s.numel() - 1)
    fr = idx - lo.to(DT)
    return s[lo] * (1 - fr) + s[hi] * fr


def quantiles(v):
    s = v.flatten().sort().values
    return _interp(s, torch.linspace(0, 1, NQ, device=v.device, dtype=DT) * (s.numel() - 1))


def match_to(v, q):
    """Rank-preserving replacement of v by the target quantiles q."""
    flat = v.flatten()
    n = flat.numel()
    r = torch.empty(n, device=v.device, dtype=DT)
    r[flat.argsort()] = _interp(q, torch.arange(n, device=v.device, dtype=DT) / max(n - 1, 1) * (NQ - 1))
    return r.reshape(v.shape)


def hb_step(x, tgt):
    for c in range(3):
        x[:, c] = match_to(x[:, c], tgt["px"][c])
    L = lap(x)
    for l in range(len(L)):
        R = analyse(L[l])
        for c in range(3):
            for f in range(NF):
                R[:, c, f] = match_to(R[:, c, f], tgt[l][c][f])
        L[l] = synth(R)
    return collapse(L)


def marginal_gap(x, tgt):
    """Mean W1 between the attacked and reference pooled sub-band marginals."""
    gaps = [float((quantiles(analyse(Ll)[:, c, f]) - tgt[l][c][f]).abs().mean())
            for l, Ll in enumerate(lap(x)) for c in range(3) for f in range(NF)]
    return sum(gaps) / len(gaps)


def hb_start(start, P):
    g = gen(20250908)
    if start == "noise":
        return torch.rand(NSYN, 3, 32, 32, device=DEV, generator=g)
    return P["cand"][torch.randperm(P["cand"].shape[0], device=DEV, generator=g)[:NSYN]]


@torch.no_grad()
def hb(start, iters=ITERS):
    P = pools()
    ref = P["ref_fit"].to(DT)                           # fit on ref_fit, scored on ref_eval
    tgt = {"px": [quantiles(ref[:, c]) for c in range(3)]}
    for l, Ll in enumerate(lap(ref)):
        R = analyse(Ll)
        tgt[l] = [[quantiles(R[:, c, f]) for f in range(NF)] for c in range(3)]
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f"hb_{start}.pt")
    st = Store(os.path.join(OUT, f"hb_{start}_trace.json"))
    trace = st.cell("trace")
    x = (torch.load(path).to(DEV) if trace and os.path.exists(path) else hb_start(start, P)).to(DT)

    def probe(it):
        g = gen(999)
        a = P["ref_eval"][torch.randperm(P["ref_eval"].shape[0], device=DEV, generator=g)[:N]]
        b = x[torch.randperm(x.shape[0], device=DEV, generator=g)[:N]].float()
        r = msw.test(a, b)
        trace.append({"it": it, "T": float(r.T), "p": float(r.p), "gap": marginal_gap(x, tgt)})
        torch.save(x.float(), path)
        st.save()
        print(f"hb {start} iter {it:3d}: T={r.T:.4g} p={r.p:.5f} gap={trace[-1]['gap']:.3e}", flush=True)
    if not trace:
        probe(0)
    while len(trace) <= iters:
        x = hb_step(x, tgt)
        probe(len(trace))


# ------------------------------------------------------------- scoring
@torch.no_grad()
def score(reps=200, cells=None):
    import warnings
    P = pools()
    s = Store(os.path.join(OUT, "selection.json")).d
    idx = lambda k: torch.tensor(s[k], device=DEV)
    hbp = lambda k: torch.load(os.path.join(WORK, f"hb_{k}.pt")).to(DEV).float().clamp(0, 1)
    jobs = {"null": lambda: P["null_pool"],
            "select|random": lambda: P["cand"][idx("ctrl")],
            "select|matched": lambda: P["cand"][idx("sel")],
            "hbnoise|before": lambda: hb_start("noise", P),
            "hbnoise|after": lambda: hbp("noise"),
            "hbblur|before": lambda: hb_start("blur", P),
            "hbblur|after": lambda: hbp("blur")}
    st = Store(os.path.join(OUT, "scores.json"))
    for name in cells or jobs:
        arr = st.cell(name)
        if len(arr) >= reps:
            continue
        PB, t0, n0 = jobs[name](), time.time(), len(arr)
        while len(arr) < reps:
            r = len(arr)
            g = gen(70000 + r if name == "null" else 1000 * r + 7)
            a = P["ref_eval"][torch.randperm(P["ref_eval"].shape[0], device=DEV, generator=g)[:N]]
            b = PB[torch.randperm(PB.shape[0], device=DEV, generator=g)[:N]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = msw.test(a, b)
            arr.append({"p": float(res.p), "T": float(res.T),
                        "comp": {k: float(v) for k, v in res.components.items()}})
            if len(arr) % 20 == 0 or len(arr) == reps:
                st.save()
        rej = np.mean([v["p"] <= 0.05 for v in arr])
        print(f"{name:16s} reps={len(arr)} rejection={rej:.3f} "
              f"({(time.time() - t0) / (len(arr) - n0):.2f} s/rep)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "hb", "score"])
    ap.add_argument("start", nargs="?", choices=["noise", "blur"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--cells", nargs="*")
    a = ap.parse_args()
    {"select": select, "hb": lambda: hb(a.start), "score": lambda: score(a.reps, a.cells)}[a.step]()
