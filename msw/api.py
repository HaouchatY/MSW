"""The public surface: ``msw.test``, ``msw.distance``, ``msw.stationarity_index``.

``test(A, B)`` runs the calibrated two-sample test between two image sets and
returns a :class:`TestResult` with the portfolio statistic, an EXACT
finite-sample p-value, per-component and per-level diagnostics, and a
stationarity self-check.

Where the exact p-value comes from
----------------------------------
The two datasets are cut into ``L`` matched blocks of equal size.  Under H0 the
2L blocks are i.i.d., so swapping the two halves of any matched pair is measure
preserving: the randomization group is the pair group ``(Z2)^L`` of
:mod:`msw.pairgroup`.  Every component of the statistic transforms under that
group by pure sign algebra -- the transport Gram matrix by
``Psi_ij -> sigma_i sigma_j Psi_ij``, the first-order rows by ``m_i -> -m_i`` --
so the whole orbit is enumerated without recomputing a single feature, sort or
convolution.  The p-value is the rank of the observed value in its own orbit:
exact in finite samples, with no Monte-Carlo permutations, no asymptotics and no
external null runs.  The orbit is the full ``2^L`` group when that is affordable
(L <= 12, e.g. every N < 16) and a parity SUBGROUP of 4096 elements otherwise.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import components as C
from .weights import slice_weights
from .banks import make_bank
from .core import get_device
from .estimators import MultiScaleSW
from .pairgroup import block_plan
from .portfolio import Portfolio, core_columns, portfolios

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

    T: float                    # portfolio statistic (max orbit-standardized component)
    p: float                    # exact p-value of T on the orbit
    p_cct: float                # Cauchy combination of the per-component p-values
    p_wy: float                 # Westfall-Young min-p over the same components
    components: dict            # component name -> observed value
    component_p: dict           # component name -> exact p-value
    per_level: dict             # slice family -> per-level transport z
    stationarity_index: float   # see `stationarity_index`; > 0.1 prints a warning
    n_used: int                 # images actually consumed per dataset (= L * M)
    L: int                      # matched image blocks
    M: int                      # images per block
    group_size: int             # group elements enumerated (p granularity = 1/group_size)
    portfolio: str              # which named portfolio was combined
    _diag: str = field(default="", repr=False)

    @property
    def G(self) -> int:
        """Deprecated alias of ``L`` (v0.1 called the blocks 'groups')."""
        return self.L

    def diagnose(self) -> str:
        """Name the pyramid level and slice family carrying the difference."""
        return self._diag


_PORTFOLIOS: dict = {}


def _portfolio(channels, size, levels, k, n_filters, L, n_patterns, pattern_mode,
               seed, use_joint) -> Portfolio:
    """Cached portfolio: banks and pyramids are deterministic, so reuse is safe."""
    key = (channels, size, levels, k, n_filters, L, n_patterns, pattern_mode, seed,
           use_joint, str(get_device()))
    if key not in _PORTFOLIOS:
        _PORTFOLIOS[key] = Portfolio(channels, size, levels=levels, k=k,
                                     n_filters=n_filters, use_joint=use_joint, L=L,
                                     n_patterns=n_patterns, pattern_mode=pattern_mode,
                                     seed=seed)
    return _PORTFOLIOS[key]


def test(a: torch.Tensor, b: torch.Tensor, L: int = 16, n_patterns: int = 4096,
         portfolio: str = "CORE", combiner: str = "maxz", levels: int | None = None,
         k: int = 5, n_filters: int | None = None, use_joint: bool = True,
         seed: int = 0, groups: int | None = None, splitting: str | None = None
         ) -> TestResult:
    """Calibrated two-sample test between image sets ``a`` and ``b``.

    a, b        (N, C, n, n) tensors on any device, C in {1, 3}, N >= 2.
    L           matched image blocks (default 16, dropping to N when N < 16).
                Blocks must have EQUAL size, so ``N mod L`` images per dataset go
                unused; at an awkward N (25, say) a smaller L uses more of them.
    n_patterns  group elements to enumerate.  The full group is used whenever
                ``2^L <= n_patterns``; otherwise a parity subgroup of this order,
                which keeps the test exact (a random subset would not).
    portfolio   which named portfolio to combine; 'CORE' (transport + tail/shape
                + raw pixel + product slices) is the shipped choice.  See
                ``msw.portfolio.portfolios`` for the menu.
    combiner    'maxz' (default), 'cct' or 'wy'.  All three are reported; maxz
                measures at least as powerful as the other two everywhere, and
                is the only one that keeps its size near nominal at N = 8, where
                the orbit has 256 elements and rank-based combiners run out of
                resolution.
    n_filters   filters per level of the DCT family; None (default) keeps the
                whole bank, which is deterministic and device-independent.
    groups      deprecated alias of ``L``.
    splitting   accepted and ignored: blocks are always equal-sized and every
                pair of blocks contributes (v0.1 offered 'balanced' / 'paper').
    """
    if a.dim() != 4 or b.dim() != 4 or a.shape[1:] != b.shape[1:]:
        raise ValueError("expected (N, C, n, n) tensors with matching shapes")
    if groups is not None:
        L = groups
    if splitting is not None and splitting not in ("balanced", "paper"):
        raise ValueError("splitting must be 'balanced' or 'paper' (both ignored)")
    channels, size = a.shape[1], a.shape[-1]
    port = _portfolio(channels, size, levels or _auto_levels(size), k, n_filters,
                      L, n_patterns, "subgroup", seed, use_joint)
    a, b = a.to(port.device).float(), b.to(port.device).float()

    si = stationarity_index(a)
    if si > 0.1:
        warnings.warn(
            f"stationarity index {si:.2f} > 0.1: the pooled statistic's "
            "translation-invariance assumptions are strained on this data "
            "(aligned/registered images?); interpret results with caution.",
            stacklevel=2)

    if combiner not in ("maxz", "cct", "wy"):
        raise ValueError("combiner must be 'maxz', 'cct' or 'wy'")
    diag_z: dict = {}
    A = port.table(a, b, diagnostics=diag_z)
    names = port.names
    if portfolio == "CORE":
        idx = core_columns(names)
    else:
        known = portfolios(names)
        if portfolio not in known:
            raise ValueError(f"unknown portfolio {portfolio!r}; "
                             f"choose one of {sorted(known)}")
        idx = known[portfolio]
    S = A[:, idx]
    pc = C.col_p(A)
    p_maxz, p_cct = C.p_maxz(S), C.p_cct([pc[i] for i in idx])
    p_wy = C.p_minp_wy(S)
    p = {"maxz": p_maxz, "cct": p_cct, "wy": p_wy}[combiner]
    Z = (S[0] - S.mean(0)) / C.std_floor(S)
    Lb, M = port.plan(a, b)
    diag = (_diagnose(diag_z, names, idx, pc) if p <= 0.1 else
            f"no significant difference detected (p = {p:.3f})")
    return TestResult(
        T=float(Z.max()), p=p, p_cct=p_cct, p_wy=p_wy,
        components={names[i]: float(A[0, i]) for i in idx},
        component_p={names[i]: pc[i] for i in idx},
        per_level=diag_z, stationarity_index=si, n_used=Lb * M, L=Lb, M=M,
        group_size=A.shape[0], portfolio=portfolio, _diag=diag)


def _diagnose(diag_z: dict, names: list[str], idx: list[int], pc: list[float]) -> str:
    """Name the level and family with the largest transport z, and the top component."""
    best = (0.0, None)
    for fam, zs in diag_z.items():
        for lvl, t in enumerate(zs):
            if abs(t) > best[0]:
                best = (abs(t), f"difference concentrated at level {lvl}, {fam} "
                                f"(t = {t:+.1f})")
    top = min(idx, key=lambda i: (pc[i], names[i]))
    where = (best[1] if best[0] >= 2.0 else
             "no scale concentration in the transport rows")
    return f"{where}; strongest component {names[top]} (p = {pc[top]:.4g})"


def distance(a: torch.Tensor, b: torch.Tensor, L: int = 16, seed: int = 0,
             levels: int | None = None, k: int = 5,
             groups: int | None = None, weighting: str = "P1s",
             signed: bool = True) -> tuple[float, float, float]:
    """The plain (pseudo-)metric value with a 95% jackknife CI.

    Estimates ``sum_s lambda_s W2^2(mu_s^A, mu_s^B)`` over pooled slice
    marginals and returns its signed square root with a delete-one-block
    jackknife interval.

    ``weighting`` selects the FIXED convex weights.  The default ``"P1s"`` is
    ``4**-l / sigmabar_s``: the frequency-flat factor divided by a canonical
    per-slice response scale, frozen in ``msw.weights`` and measured once on
    four reference corpora, so it never depends on the two datasets being
    compared and the result stays a metric.  It certifies all nine benchmark
    perturbations where the previous uniform weighting certified four, is
    strictly monotone in perturbation size on every one of them, and tightens
    the ratio to the true Wasserstein distance from [0.37, 2.55] to
    [0.50, 0.95].  ``"uniform"`` reproduces the v0.2 value exactly;
    ``"P2s"`` (``4**-l / sigmabar_s**2``) buys a little more resolution at the
    cost of fidelity; ``"flat"`` is ``4**-l`` alone.

    ``signed=True`` (default) returns ``sign(u) * sqrt(|u|)``.  The estimator is
    centred at zero under the null, so half of all null draws give a negative
    value; clamping them to zero -- what ``signed=False`` does -- hides that the
    interval covers zero and makes an unresolvable comparison look like a
    measured zero.  Identical inputs still give exactly 0.

    Use this to ORDER differences.  To decide whether a difference exists at
    all, use ``msw.test``: it may weight adaptively, which buys far more
    sensitivity but forfeits the metric property.

    ``groups`` is a deprecated alias of ``L``.
    """
    if groups is not None:
        L = groups
    channels, size = a.shape[1], a.shape[-1]
    est = MultiScaleSW(channels, size, levels=levels or _auto_levels(size), k=k,
                       n_filters=None, bank="dct", seed=seed)
    dev = est.layouts[0].filters.device
    a, b = a.to(dev).float(), b.to(dev).float()
    n = min(a.shape[0], b.shape[0])
    Lb, M = block_plan(n, n, L)
    if Lb < 3:
        raise ValueError("need at least 3 image blocks for the jackknife interval")
    psi, _, _ = C.block_rows(C.to_double(est.features(a[:n])),
                             C.to_double(est.features(b[:n])), Lb, M,
                             want_block_sums=False)
    w = torch.tensor(slice_weights(channels, est.level_sizes, scheme=weighting),
                     dtype=psi.dtype, device=psi.device)
    g = (psi * w).sum(2)                              # (L, L), canonical weights
    g.fill_diagonal_(0.0)
    tot = float(g.sum()) / 2.0                        # sum over unordered pairs
    npair = Lb * (Lb - 1) / 2.0
    u = tot / npair
    rowsum = g.sum(1)                                 # (L,)
    npair_i = (Lb - 1) * (Lb - 2) / 2.0
    loo = (tot - rowsum.double()) / npair_i           # delete-one U-statistics
    se = float(((Lb - 1) / Lb * ((loo - loo.mean()) ** 2).sum()).clamp_min(0).sqrt())
    tq = _t975(Lb - 1)
    root = (lambda v: math.copysign(math.sqrt(abs(v)), v)) if signed else \
           (lambda v: math.sqrt(max(0.0, v)))
    return (root(u), root(u - tq * se), root(u + tq * se))


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
    C_ = x.shape[1]
    w = make_bank("dct", None, k, C_)
    dev = w.device
    acc = None
    for i in range(0, x.shape[0], batch):
        y = F.conv2d(x[i:i + batch].to(dev).float(), w).abs().sum(dim=(0, 1))
        acc = y if acc is None else acc + y
    h = acc.shape[-1]
    g = min(grid, h)
    cells = F.adaptive_avg_pool2d(acc.view(1, 1, h, h), g).flatten()
    return float(cells.std(unbiased=False) / cells.mean().clamp_min(1e-12))
