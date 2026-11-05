"""Ablations of the shipped test (Figures 'ablation' and 'depth', Table 'banks',
and the component-deletion numbers of the ablation appendix).  Every series is
ours; each one is a column subset, a pyramid truncation or a bank swap of the
same orbit table msw.test builds (msw.portfolio.Portfolio, msw.components).

levels    128-px fields, spectrum |k|^-1/2, amplitude boost (1 + delta) confined
          to [1, 3)/128 cyc/px at fixed total power; N = 200, delta in
          {0.04, 0.08, 0.16, 0.32}; 50 null reps (seed 70000 + r, both sets
          clean) and 50 alternative reps (seed 1000 r + 7).  Per rep: for every
          pyramid level the max-z of its transport + tail/shape columns and its
          exact p (Figure 'ablation', left); the per-level transport z of the DCT
          family (msw's per_level read-out, Figure 'depth', left); the orbit z
          and p of b_T_ml, b_T_iv, b_T_ivp and of CORE, at full depth and with
          the pyramid cut to one level (Figure 'depth', right).
families  N = 500, 50 null / 30 alternative reps, same seeds.  CelebA central
          32-px crops + Gaussian noise (sigma in {0.00025 .. 0.0025}); CIFAR-10
          + the +-eps Nyquist checkerboard (eps in {0.0025 .. 0.02}).  Per rep:
          (T, p) of each slice family inside CORE -- per-channel DCT, opponent
          chroma, raw pixel, product slices -- and of CORE itself.
banks     Table 'banks': the 5x5 spatial atoms of both slice families replaced
          by 24 random (gaussian_bank, seed 0), orthonormal (orthogonal_bank,
          seed 0) or patch-PCA atoms (top 24 eigen-patches of 100,000 5x5
          luminance patches of CIFAR-10 train images 25,000-49,999, seed 7),
          pyramid and protocol fixed; house settings at N = 500, 24 alternative
          reps per rung (exact test, no nulls).  p of CORE and CORE-J per rep.
loo       Component deletion: the nine house settings on the Table 1 grid at
          N = 500, 48 alternative reps per rung; p of CORE, of the transport
          rows alone (T_all) and of CORE minus each group (T, TL, PX, J).

    python run.py levels                    # ~20 GPU-minutes
    python run.py families                  # ~10 GPU-minutes
    python run.py banks [--settings ...]    # ~1 GPU-hour; settings can run in parallel
    python run.py loo [--rows ...]          # ~1 GPU-hour; rows can run in parallel
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from bench import ROWS, grid, key
from draws import DEV, celeba, cifar, gen, setting
from protocol import Store

import msw.components as C
from msw.api import _auto_levels
from msw.banks import _normalise, gaussian_bank, orthogonal_bank
from msw.pairgroup import block_plan, patterns_for
from msw.portfolio import Portfolio, _level_z, core_columns, groups, portfolios
from msw.spectral import freq_grid, powerlaw_spectrum, sample_field

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MSW_XP_OUT", os.path.join(HERE, "results"))
N = 500                     # images per set (families, banks, loo)
_PORT = {}


def port(ch, size, levels=None):
    """The portfolio msw.test uses (auto depth unless given)."""
    k = (ch, size, levels or _auto_levels(size))
    if k not in _PORT:
        _PORT[k] = Portfolio(ch, size, levels=k[2])
    return _PORT[k]


def maxz(A, idx):
    """(T, p) of the max-z combination over columns idx, as msw.test."""
    S = A[:, idx]
    return float(C.finite((S[0] - S.mean(0)) / C.std_floor(S)).max()), float(C.p_maxz(S))


def fill(store, cell, reps, fn):
    arr = store.cell(cell)
    t0, n0 = time.time(), len(arr)
    while len(arr) < reps:
        arr.append(fn(len(arr)))
        store.save()
    if len(arr) > n0:
        print(f"{cell:40s} reps={len(arr):4d}  ({(time.time() - t0) / (len(arr) - n0):.2f} s/rep)",
              flush=True)


# ------------------------------------------------------------------ levels
SIZE, N_LEV, DELTAS = 128, 200, [0.04, 0.08, 0.16, 0.32]


def boost(delta):
    s0 = powerlaw_spectrum(SIZE, 0.5, total_power=1.0, device=DEV)
    if delta == 0:
        return s0
    kk = freq_grid(SIZE, device=DEV)
    s = s0.clone()
    s[(kk >= 1 / SIZE) & (kk < 3 / SIZE)] *= (1 + delta) ** 2
    return s * (s0.sum() / s.sum())


@torch.no_grad()
def per_level(P, a, b):
    """Per pyramid level of the DCT family: max-z (T, p) of its transport and
    tail/shape columns, and the identity-row transport z (msw's read-out)."""
    est = P.port.base
    L, M = block_plan(a.shape[0], b.shape[0], P.L)
    pats = patterns_for(L, P.n_patterns, P.pattern_mode, seed=P.pattern_seed, device=P.device)
    fa = C.to_double(est.features(a.to(P.device).float()))
    fb = C.to_double(est.features(b.to(P.device).float()))
    psi, bs, bse = C.block_rows(fa, fb, L, M)
    T, p, j = [], [], 0
    for c in est.level_sizes:
        sl = slice(j, j + c)
        cols = C.transport_components(psi[:, :, sl].contiguous(), pats, [c])
        for k in ("bkurt", "bsh"):
            cols += C.zcomps(C.signed_z(bs[k][:, sl], bse[k][sl], pats))
        t, q = maxz(torch.stack(cols, 1), slice(None))
        T.append(t)
        p.append(q)
        j += c
    return T, p, _level_z(psi, est.level_sizes)


@torch.no_grad()
def levels_rep(a, b):
    rec = {}
    rec["lev_T"], rec["lev_p"], rec["lev_z"] = per_level(port(1, SIZE), a, b)
    for tag, lv in (("full", None), ("one", 1)):
        P = port(1, SIZE, lv)
        A = P.table(a.to(P.device).float(), b.to(P.device).float())
        Z = C.finite((A[0] - A.mean(0)) / C.std_floor(A))
        pc = C.col_p(A)
        for col in ("b_T_iv", "b_T_ml", "b_T_ivp"):
            i = P.names.index(col)
            rec[f"{tag}:{col}"], rec[f"{tag}:{col}:p"] = float(Z[i]), pc[i]
        rec[f"{tag}:CORE"], rec[f"{tag}:CORE:p"] = maxz(A, core_columns(P.names))
    return rec


def run_levels(a):
    store = Store(os.path.join(RES, "levels.json"))
    s0 = boost(0)
    for d in [0.0] + DELTAS:
        if d == 0:
            def fn(r):
                g = gen(70000 + r)
                return levels_rep(sample_field(s0, N_LEV, generator=g), sample_field(s0, N_LEV, generator=g))
        else:
            def fn(r, s1=boost(d)):
                g = gen(1000 * r + 7)
                return levels_rep(sample_field(s0, N_LEV, generator=g), sample_field(s1, N_LEV, generator=g))
        fill(store, f"delta{key(d)}", a.nulls if d == 0 else a.reps, fn)


# ---------------------------------------------------------------- families
FAM_GRID = {"celeba_noise": [0.00025, 0.0005, 0.00125, 0.0025],
            "cifar_checker": [0.0025, 0.005, 0.01, 0.02]}


def family_columns(names):
    core = set(n for g in ("T", "TL", "PX", "J") for n in groups(names)[g])
    out = {tag: [names.index(n) for n in names if n.startswith(pre) and n in core]
           for tag, pre in (("dct", "b_"), ("chroma", "c_"), ("pixel", "px_"), ("product", "j_"))}
    out["CORE"] = core_columns(names)
    return out


def run_families(a):
    for panel in a.panels:
        if panel == "celeba_noise":
            x = celeba()[:, :, 16:48, 16:48].contiguous()     # central crops, as the original
            alt = lambda b, e, g: b + e * torch.randn(b.shape, device=DEV, generator=g)
        else:
            x = cifar()
            r_ = torch.arange(32, device=DEV)
            chk = ((-1.0) ** (r_[:, None] + r_[None, :])).view(1, 1, 32, 32)
            alt = lambda b, e, g: (b + e * chk).clamp(0, 1)
        draw = lambda g: x[torch.randperm(x.shape[0], device=DEV, generator=g)[:N]]
        P = port(3, 32)
        cols = family_columns(P.names)
        store = Store(os.path.join(RES, f"families_{panel}.json"))

        @torch.no_grad()
        def fn(r, e):
            g = gen(70000 + r if e == 0 else 1000 * r + 7)
            A, B = draw(g), draw(g)
            if e:
                B = alt(B, e, g)
            T = P.table(A, B)
            return {f: maxz(T, i) for f, i in cols.items()}
        for e in [0.0] + FAM_GRID[panel]:
            fill(store, f"eps{key(e)}", a.nulls if e == 0 else a.reps, lambda r, e=e: fn(r, e))


# ------------------------------------------------------------------- banks
BANKS = ["dct", "gauss", "ortho", "pca"]
BANK_GRID = {"fields_slope": [0.002, 0.004, 0.008, 0.016, 0.032, 0.064],
             "celeba_noise": [0.000125, 0.00025, 0.0005, 0.00125, 0.0025, 0.005],
             "celeba_blur": [0.033, 0.05, 0.067, 0.1, 0.15, 0.22, 0.33],
             "celeba_shading": [0.005, 0.01, 0.02, 0.04, 0.08],
             "cifar_checker": [0.00015625, 0.0003125, 0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02]}
_CH = {"fields_slope": (1, 32), "celeba_shading": (3, 64)}


def pca_atoms():
    """Top-24 eigen-patches of held-out CIFAR luminance (cached in results/:
    the patch positions come from a device generator)."""
    path = os.path.join(RES, "pca_atoms.pt")
    if os.path.exists(path):
        return torch.load(path).to(DEV)
    lum = cifar()[25000:].mean(1)
    g = gen(7)
    idx = torch.randint(0, lum.shape[0], (100_000,), device=DEV, generator=g)
    py = torch.randint(0, 28, (100_000,), device=DEV, generator=g)
    px = torch.randint(0, 28, (100_000,), device=DEV, generator=g)
    r = torch.arange(5, device=DEV)
    p = lum[idx[:, None, None], (py[:, None] + r)[:, :, None], (px[:, None] + r)[:, None, :]].reshape(-1, 25)
    p = p - p.mean(1, keepdim=True)
    _, v = torch.linalg.eigh((p.t() @ p / p.shape[0]).double())
    at = v[:, -24:].flip(-1).t().float().reshape(24, 5, 5)
    at = at / at.flatten(1).norm(dim=1).view(-1, 1, 1)
    os.makedirs(RES, exist_ok=True)
    torch.save(at.cpu(), path)
    return at


def atoms(bank):
    if bank == "gauss":
        return gaussian_bank(24, 5, 1, generator=gen(0)).squeeze(1)
    if bank == "ortho":
        return orthogonal_bank(24, 5, 1, generator=gen(0)).squeeze(1)
    return pca_atoms()


def bank_port(bank, ch, size):
    """A fresh portfolio whose two slice families use `bank`'s spatial atoms in
    the same placements (per channel; opponent axes) as the DCT."""
    P = Portfolio(ch, size, levels=_auto_levels(size))
    if bank == "dct":
        return P
    at = atoms(bank).to(P.device)
    if ch == 1:
        pc = _normalise(at.unsqueeze(1).clone())
    else:
        pc = torch.zeros(72, 3, 5, 5, device=at.device)
        for c in range(3):
            pc[c::3, c] = at
        pc = _normalise(pc)
    fams = [(P.port.base, pc)]
    if P.port.opp is not None:
        vs = torch.tensor([[1., 1., 1.], [1., -1., 0.], [1., 1., -2.]], device=at.device)
        vs = vs / vs.norm(dim=1, keepdim=True)
        fams.append((P.port.opp, _normalise(torch.stack([v.view(3, 1, 1) * x for x in at for v in vs]))))
    for est, w in fams:
        for la in est.layouts:
            assert la.filters.shape == w.shape, (la.filters.shape, w.shape)
            la.filters = w.to(la.filters.dtype)
            la.scale = 1.0 / est.pyr.slice_norms(la.filters, la.level)
    return P


def run_banks(a):
    for s in a.settings:
        draw, alt = setting(s)
        ch, size = _CH.get(s, (3, 32))
        store = Store(os.path.join(RES, f"banks_{s}.json"))
        for bank in BANKS:
            P = bank_port(bank, ch, size)
            pf = portfolios(P.names)
            idx = {"CORE": pf["CORE"], "CORE-J": pf["CORE-J"]}

            @torch.no_grad()
            def fn(r, f):
                g = gen(1000 * r + 7)
                A = P.table(draw(N, g), f(N, g))
                return {k: float(C.p_maxz(A[:, v])) for k, v in idx.items()}
            for e in BANK_GRID[s]:
                fill(store, f"{bank}|eps{key(e)}", a.reps, lambda r, f=alt(e): fn(r, f))
            del P


# --------------------------------------------------------------------- loo
LOO = ["CORE", "T_all", "CORE-T", "CORE-TL", "CORE-PX", "CORE-J"]


def run_loo(a):
    for row in a.rows:
        draw, alt = setting(row)
        x = draw(2, gen(0))
        P = port(x.shape[1], x.shape[-1])
        pf = portfolios(P.names)
        store = Store(os.path.join(RES, f"loo_{row}.json"))

        @torch.no_grad()
        def fn(r, f):
            g = gen(1000 * r + 7)
            A = P.table(draw(N, g), f(N, g))
            return {k: float(C.p_maxz(A[:, pf[k]])) for k in LOO}
        for e in grid(row):
            fill(store, f"eps{key(e)}", a.reps, lambda r, f=alt(e): fn(r, f))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["levels", "families", "banks", "loo"])
    ap.add_argument("--reps", type=int)
    ap.add_argument("--nulls", type=int, default=50)
    ap.add_argument("--panels", nargs="*", default=list(FAM_GRID))
    ap.add_argument("--settings", nargs="*", default=list(BANK_GRID))
    ap.add_argument("--rows", nargs="*", default=ROWS)
    a = ap.parse_args()
    a.reps = a.reps or {"levels": 50, "families": 30, "banks": 24, "loo": 48}[a.part]
    {"levels": run_levels, "families": run_families, "banks": run_banks, "loo": run_loo}[a.part](a)
