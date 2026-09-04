"""Aggregating per-slice values into a distance or into a test statistic.

Two different jobs, deliberately kept apart.

*Distance*   ``combine(..., mode='uniform'|'flat')`` uses **fixed** weights, so the
             result is a genuine (pseudo-)metric that does not depend on the
             sample: the object one would quote as "the distance between these two
             datasets".

*Test*       ``combine(..., mode='invvar'|'zsum'|'invvar_split')`` weights each
             slice by its estimated precision.  This is what one wants for
             deciding whether two datasets differ, and it is essential for
             multi-scale slices: coarse pyramid levels carry large discrepancies
             *and* large noise (their effective sample size is ~N, not N n^2), so
             a uniform average over levels is dominated by their noise and can be
             worse than using the finest level alone.  Weights are data-dependent,
             so the null must be calibrated by permutation or by independent H0
             draws -- which is what a two-sample test does anyway.
"""

from __future__ import annotations

import torch

__all__ = ["combine", "level_weights", "maxlevel_z", "power_at", "auc", "rel_floor"]


def level_weights(sizes: list[int], mode: str = "flat") -> torch.Tensor:
    """Fixed per-slice weights from the number of slices at each pyramid level.

    'uniform'  every slice equal.
    'flat'     lambda_l ~ 4^-l, which makes the *average frequency envelope* of the
               family flat (each level covers ~4^-l of the frequency plane), so the
               family is frequency-unbiased and the resulting distance obeys
               D <= W_2 / sqrt(d) exactly as classic SW does.
    """
    w = []
    for l, s in enumerate(sizes):
        v = 1.0 if mode == "uniform" else 4.0 ** (-l)
        w += [v / s] * s
    t = torch.tensor(w)
    return t / t.sum()


def combine(tg: torch.Tensor, mode: str = "invvar", weights: torch.Tensor | None = None,
            eps: float = 1e-12) -> float:
    """Aggregate a (G, S) blocked per-slice statistic into one scalar.

    'uniform' / 'fixed'  weighted mean with `weights` (uniform if None).
    'invvar'             weights 1/se_s^2, se_s the standard error over the G blocks.
    'zsum'               mean of the per-slice z-scores t_s / se_s.
    'invvar_split'       weights estimated on the first half of the blocks and
                         applied to the second half -- removes the weight/estimate
                         correlation, at the price of half the data for each part.
    'max'                largest per-slice z-score.
    """
    g = tg.shape[0]
    tbar = tg.mean(0)
    if mode in ("uniform", "fixed"):
        w = torch.ones_like(tbar) if weights is None else weights.to(tbar.device)
        return float((w * tbar).sum() / w.sum())
    if g < 2:
        raise ValueError("variance-aware modes need groups >= 2")
    se = rel_floor(tg.std(0) / g**0.5, eps)
    if mode == "invvar":
        w = 1.0 / se**2
        return float((w * tbar).sum() / w.sum())
    if mode == "zsum":
        return float((tbar / se).mean())
    if mode == "max":
        return float((tbar / se).max())
    if mode == "invvar_split":
        h = g // 2
        se1 = rel_floor(tg[:h].std(0) / h**0.5, eps)
        t2 = tg[h:].mean(0)
        w = 1.0 / se1**2
        return float((w * t2).sum() / w.sum())
    raise ValueError(mode)


def power_at(t0: list[float], t1: list[float], alpha: float = 0.05) -> float:
    """Power of the one-sided test whose threshold is the (1-alpha) H0 quantile."""
    s = sorted(t0)
    thr = s[min(len(s) - 1, int(round((1 - alpha) * (len(s) - 1))))]
    return sum(1 for v in t1 if v > thr) / len(t1)


def auc(t0: list[float], t1: list[float]) -> float:
    a, b = torch.tensor(t0), torch.tensor(t1)
    d = b.unsqueeze(1) - a.unsqueeze(0)
    return float((d > 0).float().mean() + 0.5 * (d == 0).float().mean())


def rel_floor(se: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Floor a vector of standard errors RELATIVELY to its own scale.

    An absolute floor (1e-12) on context-scaled slice values (typically
    1e-7 .. 1e-5) lets a degenerate slice -- one whose blocked spread happens
    to vanish -- take essentially infinite weight and blow the statistic up.
    Flooring at 1e-3 of the median standard error keeps degenerate slices
    from dominating while leaving healthy ones untouched.
    """
    if se.numel() <= 1:
        return se.clamp_min(eps)
    med = float(se.median())
    return se.clamp_min(max(eps, 1e-3 * med))


def maxlevel_z(tg: torch.Tensor, level_sizes: list[int]) -> float:
    """Largest per-level z of a (G, S) blocked matrix.

    Uniform mean within each slice family (pyramid level), then the largest
    per-family z over the blocks; the level-aware repair when a difference is
    confined to one scale (see `combine` and the paper).
    """
    g = tg.shape[0]
    zs, i = [], 0
    ses = []
    for c in level_sizes:
        tl = tg[:, i:i + c].mean(1)
        ses.append(tl.std() / g ** 0.5)
        i += c
    se_t = rel_floor(torch.stack(ses))
    i = 0
    for c, se in zip(level_sizes, se_t):
        tl = tg[:, i:i + c].mean(1)
        zs.append(float(tl.mean() / se))
        i += c
    return max(zs)
