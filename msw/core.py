"""Core numerics: device handling and 1-D Wasserstein distances.

Every sliced estimator in this package reduces to the same primitive: given two
1-D samples (one per "slice"), return W_p^p.  Two paths are provided.

``w1d_sort``  exact (up to the empirical-quantile convention), O(M log M), needs
              the full sample in memory.
``w1d_quantile``  shared quantile grid, for unequal sample sizes.

Pooled-pixel estimators produce enormous 1-D samples (a single 3x3 filter on 20k
32x32 images already gives 1.8e7 values).  Rather than approximating the sort with
bins -- equal-mass bins cannot resolve the tails, and W_2 is tail-sensitive -- the
estimators cap the pooled sample by random position subsampling, which is exact
and unbiased because positions are exchangeable under stationarity.  See
`MultiScaleSW.slice_values(max_samples=...)` for details.
"""

from __future__ import annotations

import torch

_DEVICE = None

__all__ = ["get_device", "set_device", "w1d_sort", "w1d_quantile", "running_estimate"]


def get_device() -> torch.device:
    """Default device, resolved lazily on first use (never at import)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def set_device(device) -> torch.device:
    """Override the default device for all subsequent estimator construction."""
    global _DEVICE
    _DEVICE = torch.device(device)
    return _DEVICE


# ---------------------------------------------------------------------------
# exact / sort-based
# ---------------------------------------------------------------------------
def w1d_sort(a: torch.Tensor, b: torch.Tensor, p: int = 2) -> torch.Tensor:
    """W_p^p between 1-D samples, column-wise, by sorting.

    a, b : (M, K) or (M,).  Equal M -> exact order-statistic matching.
    Unequal M -> falls back to the shared-quantile-grid estimator.
    Returns (K,) or scalar.
    """
    squeeze = a.dim() == 1
    if squeeze:
        a, b = a.unsqueeze(1), b.unsqueeze(1)
    if a.shape[0] != b.shape[0]:
        out = w1d_quantile(a, b, p=p)
    else:
        out = (a.sort(dim=0).values - b.sort(dim=0).values).abs_().pow_(p).mean(dim=0)
    return out.squeeze(0) if squeeze else out


def _quantiles_of_sorted(sorted_x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Empirical quantile function of each column of `sorted_x` at levels `q`.

    Order statistic i is placed at cumulative probability (i + 0.5) / M and the
    inverse CDF is interpolated linearly in between (the usual "type 5" rule).
    """
    m = sorted_x.shape[0]
    pos = (torch.arange(m, device=sorted_x.device, dtype=sorted_x.dtype) + 0.5) / m
    idx = torch.searchsorted(pos, q.to(pos.dtype)).clamp_(1, m - 1)
    lo, hi = idx - 1, idx
    wgt = ((q - pos[lo]) / (pos[hi] - pos[lo]).clamp_min(1e-12)).unsqueeze(1)
    return sorted_x[lo] + wgt * (sorted_x[hi] - sorted_x[lo])


def w1d_quantile(a: torch.Tensor, b: torch.Tensor, p: int = 2, n_q: int | None = None) -> torch.Tensor:
    """W_p^p on a shared grid of `n_q` quantile levels (handles unequal sizes).

    `n_q = None` uses min(Ma, Mb) capped at 200k, which keeps the discretisation
    error well below the sampling error.
    """
    squeeze = a.dim() == 1
    if squeeze:
        a, b = a.unsqueeze(1), b.unsqueeze(1)
    if n_q is None:
        n_q = min(min(a.shape[0], b.shape[0]), 200_000)
    q = (torch.arange(n_q, device=a.device, dtype=a.dtype) + 0.5) / n_q
    qa = _quantiles_of_sorted(a.sort(dim=0).values, q)
    qb = _quantiles_of_sorted(b.sort(dim=0).values, q)
    out = (qa - qb).abs_().pow_(p).mean(dim=0)
    return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
def running_estimate(wpp: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Running estimate after 1, 2, ... slices: (cumsum / k) ** (1/p)."""
    k = torch.arange(1, wpp.numel() + 1, device=wpp.device, dtype=wpp.dtype)
    return (wpp.cumsum(0) / k).clamp_min(0).pow(1.0 / p)
