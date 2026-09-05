"""The public surface: ``msw.test``, ``msw.distance``, ``msw.stationarity_index``.

``test(A, B)`` runs the calibrated two-sample test between two image sets and
returns a :class:`TestResult` with the portfolio statistic, an EXACT
finite-sample p-value, per-component and per-level diagnostics, and a
stationarity self-check.

The exact p-value comes from a permutation-subgroup argument: within each
disjoint image group, relabeling the two half-blocks of one dataset flips the
sign of that group's four-term statistic exactly, and under H0 the joint law
is invariant, so the 2^G sign patterns of the per-group rows are equally
likely.  Enumerating them (256 patterns at G=8) gives an exact test with no
Monte-Carlo permutations and no external null runs, at ~milliseconds cost.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .banks import make_bank
from .core import get_device
from .estimators import MultiScaleSW, PortfolioSW
from .features import balanced_blocked_values, balanced_group_sizes, blocked_values
from .testing import rel_floor

__all__ = ["test", "distance", "stationarity_index", "TestResult"]

# two-sided 97.5% Student-t quantiles for df = 1..30 (df > 30 -> 1.96)
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def _t975(df: int) -> float:
    return _T975[df - 1] if 1 <= df <= 30 else 1.96


def _auto_levels(size: int) -> int:
    return 4 if size <= 32 else (5 if size <= 64 else 6)


@dataclass
class TestResult:
    """Everything ``msw.test`` measured, plus a plain-language diagnosis."""

    T: float                    # portfolio statistic (max null-standardized component)
    p: float                    # exact sign-flip p-value of T
    p_cct: float                # Cauchy combination of per-component sign-flip p-values
    components: dict            # component name -> observed value
    component_p: dict           # component name -> exact sign-flip p-value
    per_level: dict             # bank name -> list of per-level t statistics
    stationarity_index: float   # see `stationarity_index`; > 0.1 prints a warning
    n_used: int                 # images actually consumed per dataset
    G: int                      # number of disjoint image groups
    _diag: str = field(default="", repr=False)

    def diagnose(self) -> str:
        """Name the pyramid level and slice family carrying the difference."""
        return self._diag


def _pattern_components(tg: torch.Tensor, level_sizes: list[int], pats: torch.Tensor,
                        iv_cols: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(inverse-variance, max-level) component values under every sign pattern.

    tg    (G, S) blocked per-slice rows;  pats (P, G) of +-1.
    Returns iv (P,), ml (P,).  The identity pattern must be row 0.
    """
    g = tg.shape[0]
    t_iv = tg if iv_cols is None else tg[:, iv_cols]
    f = torch.einsum("pg,gs->pgs", pats, t_iv)                       # (P, G, S')
    tbar = f.mean(1)
    se = f.std(1) / g ** 0.5
    med = se.median(dim=1, keepdim=True).values
    se = torch.maximum(se, torch.maximum(torch.full_like(se, 1e-12), 1e-3 * med))
    w = 1.0 / se ** 2
    iv = (w * tbar).sum(1) / w.sum(1)
    fz = torch.einsum("pg,gs->pgs", pats, tg)
    zs, i = [], 0
    ses = []
    means = []
    for c in level_sizes:
        tl = fz[:, :, i:i + c].mean(2)                               # (P, G)
        means.append(tl.mean(1))
        ses.append(tl.std(1) / g ** 0.5)
        i += c
    se_l = torch.stack(ses, 1)                                       # (P, L)
    med = se_l.median(dim=1, keepdim=True).values
    se_l = torch.maximum(se_l, torch.maximum(torch.full_like(se_l, 1e-12), 1e-3 * med))
    ml = (torch.stack(means, 1) / se_l).max(1).values
    return iv, ml


def _patterns(G: int, device, max_patterns: int = 4096) -> torch.Tensor:
    if 2 ** G <= max_patterns:
        idx = torch.arange(2 ** G, device=device)
    else:  # identity + random subsample of the subgroup (still a valid test)
        idx = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                         torch.randint(1, 2 ** G, (max_patterns - 1,), device=device,
                                       generator=torch.Generator(device=device).manual_seed(0))])
    return ((idx.view(-1, 1) >> torch.arange(G, device=device)) & 1) * -2.0 + 1.0


def _level_ts(tg: torch.Tensor, level_sizes: list[int]) -> list[float]:
    g = tg.shape[0]
    out, i = [], 0
    for c in level_sizes:
        tl = tg[:, i:i + c].mean(1)
        se = float(rel_floor((tl.std() / g ** 0.5).reshape(1))[0])
        out.append(float(tl.mean()) / se)
        i += c
    return out


def test(a: torch.Tensor, b: torch.Tensor, groups: int | None = None,
         splitting: str = "balanced", seed: int = 0) -> TestResult:
    """Calibrated two-sample test between image sets ``a`` and ``b``.

    a, b       (N, C, n, n) tensors on any device, C in {1, 3}, N >= 4.
    groups     number of disjoint image groups G (default: min(8, N//2)).
    splitting  'balanced' uses all floor(N/2) image pairs (groups may differ in
               size by one); 'paper' reproduces the fixed-size splitting of the
               paper, consuming 2G * floor(N / 2G) images.
    """
    if a.dim() != 4 or b.dim() != 4 or a.shape[1:] != b.shape[1:]:
        raise ValueError("expected (N, C, n, n) tensors with matching shapes")
    C, size = a.shape[1], a.shape[-1]
    port = PortfolioSW(C, size, levels=_auto_levels(size), k=5, n_filters=None,
                       seed=seed)
    dev = port.base.layouts[0].filters.device
    a, b = a.to(dev).float(), b.to(dev).float()

    si = stationarity_index(a)
    if si > 0.1:
        warnings.warn(
            f"stationarity index {si:.2f} > 0.1: the pooled statistic's "
            "translation-invariance assumptions are strained on this data "
            "(aligned/registered images?); interpret results with caution.",
            stacklevel=2)

    ests = [("dct", port.base, None)]
    if port.opp is not None:
        ests.append(("opponent", port.opp, torch.tensor(port._chroma_idx, device=dev)))

    tgs = []
    if splitting == "balanced":
        sizes = balanced_group_sizes(a.shape[0], b.shape[0], groups)
        G, n_used = len(sizes), 2 * sum(balanced_group_sizes(a.shape[0], b.shape[0], groups))
        for _, est, _c in ests:
            tgs.append(balanced_blocked_values(est.features(a), est.features(b), groups))
    elif splitting == "paper":
        G = groups if groups is not None else 8
        n_used = 2 * G * (min(a.shape[0], b.shape[0]) // (2 * G))
        for _, est, _c in ests:
            tgs.append(blocked_values(est.features(a), est.features(b), groups=G))
    else:
        raise ValueError("splitting must be 'balanced' or 'paper'")

    pats = _patterns(G, dev)
    comps, names = [], []
    for (name, est, cols), tg in zip(ests, tgs):
        iv, ml = _pattern_components(tg, est.level_sizes, pats, iv_cols=cols)
        comps += [iv, ml]
        names += ([f"{name}_iv", f"{name}_ml"] if name == "dct"
                  else ["chroma_iv", "opponent_ml"])
    A = torch.stack(comps, 1)                                        # (P, ncomp)
    mu, sd = A.mean(0), rel_floor(A.std(0))
    T = ((A - mu) / sd).max(1).values                                # (P,)
    P = A.shape[0]
    p = float((1 + (T[1:] >= T[0]).sum()) / P)
    comp_p = {nm: float((1 + (A[1:, i] >= A[0, i]).sum()) / P) for i, nm in enumerate(names)}
    # Cauchy combination of the per-component exact p-values: level-valid under
    # arbitrary dependence between components (Liu & Xie, 2020).
    cct = sum(math.tan((0.5 - pc) * math.pi) for pc in comp_p.values()) / len(comp_p)
    p_cct = 0.5 - math.atan(cct) / math.pi

    per_level = {name: _level_ts(tg, est.level_sizes)
                 for (name, est, _c), tg in zip(ests, tgs)}
    diag = _diagnose(ests, tgs)
    return TestResult(T=float(T[0]), p=p, p_cct=p_cct,
                      components={nm: float(A[0, i]) for i, nm in enumerate(names)},
                      component_p=comp_p, per_level=per_level,
                      stationarity_index=si, n_used=n_used, G=G, _diag=diag)


def _diagnose(ests, tgs) -> str:
    best = (0.0, "no clear concentration")
    for (name, est, _c), tg in zip(ests, tgs):
        if name == "dct":
            fams = [("per-channel DCT", list(range(tg.shape[1])), est.level_sizes)]
        else:
            lum, chr_, i = [], [], 0
            for c in est.level_sizes:
                lum += [i + j for j in range(c) if j % 3 == 0]
                chr_ += [i + j for j in range(c) if j % 3 != 0]
                i += c
            per_lvl = [c // 3 for c in est.level_sizes]
            fams = [("opponent luminance", lum, per_lvl),
                    ("opponent chroma", chr_, [2 * v for v in per_lvl])]
        for fam, cols, lsizes in fams:
            ts = _level_ts(tg[:, cols], lsizes)
            for lvl, t in enumerate(ts):
                if abs(t) > best[0]:
                    best = (abs(t), f"difference concentrated at level {lvl}, {fam} "
                                    f"(t = {t:+.1f})")
    return best[1]


def distance(a: torch.Tensor, b: torch.Tensor, groups: int = 10,
             seed: int = 0) -> tuple[float, float, float]:
    """The plain (pseudo-)metric value with a 95% jackknife-style CI.

    Returns (d, lo, hi): d is the fixed-weight multi-scale sliced distance and
    [lo, hi] a Student-t interval over the G disjoint image groups.  The CI is
    on the METHOD'S value D -- it is NOT a certified bound on the true W2:
    D's frequency tilt (documented in the paper) means D can exceed W2-bar on
    low-frequency differences.
    """
    C, size = a.shape[1], a.shape[-1]
    est = MultiScaleSW(C, size, levels=_auto_levels(size), k=5, n_filters=None,
                       bank="dct", seed=seed)
    dev = est.layouts[0].filters.device
    a, b = a.to(dev).float(), b.to(dev).float()
    d = est.distance(a, b)
    rows = balanced_blocked_values(est.features(a), est.features(b), groups).mean(1)
    G = rows.shape[0]
    m, se = float(rows.mean()), float(rows.std() / G ** 0.5)
    tq = _t975(G - 1)
    lo = math.sqrt(max(0.0, m - tq * se))
    hi = math.sqrt(max(0.0, m + tq * se))
    return d, lo, hi


@torch.no_grad()
def stationarity_index(x: torch.Tensor, k: int = 5, grid: int = 4,
                       batch: int = 256) -> float:
    """How non-stationary a dataset looks to the level-0 filter bank.

    Pools the mean |response| of the k x k DCT bank over a `grid` x `grid`
    partition of output positions and returns the relative spread (std/mean)
    of the cell energies.  Calibration points: Gaussian random fields ~0.005,
    random crops of photographs ~0.07, aligned face datasets ~0.22.  Above
    ~0.1 the pooled statistic's translation-invariance assumptions are
    strained: prefer image-split nulls and treat cross-pipeline comparisons
    with caution.
    """
    C = x.shape[1]
    w = make_bank("dct", None, k, C)
    dev = w.device
    acc = None
    for i in range(0, x.shape[0], batch):
        y = F.conv2d(x[i:i + batch].to(dev).float(), w).abs().sum(dim=(0, 1))
        acc = y if acc is None else acc + y
    h = acc.shape[-1]
    g = min(grid, h)
    cells = F.adaptive_avg_pool2d(acc.view(1, 1, h, h), g).flatten()
    return float(cells.std(unbiased=False) / cells.mean().clamp_min(1e-12))
