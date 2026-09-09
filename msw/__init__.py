"""msw -- Multi-Scale Sliced Wasserstein distances and calibrated two-sample
tests between image datasets, built for stationary (texture-like) image
distributions.

Quick start::

    import msw
    r = msw.test(A, B)        # A, B: (N, C, n, n) tensors, any device
    r.T, r.p                  # portfolio statistic, EXACT randomization p-value
    r.component_p             # which component saw it
    r.diagnose()              # "difference concentrated at level 0, opponent chroma ..."
    d, lo, hi = msw.distance(A, B)

The test cuts both datasets into L matched blocks and enumerates the pair group
(Z2)^L, under which every component transforms by sign algebra; the p-value is
the rank of the observed statistic in that orbit, exact in finite samples.  It
is a real test at N = 8 images per side (256 group elements, granularity 1/256).
"""

__version__ = "0.3.0"

from .core import get_device, set_device, w1d_quantile, w1d_sort, running_estimate
from .banks import (Pyramid, dct_bank, gabor_bank, gaussian_bank, make_bank,
                    opponent_dct_bank, orthogonal_bank)
from .features import (SliceFeatures, balanced_blocked_values, blocked_values,
                       concat_features, paired_values)
from .estimators import (CSWSlicer, MultiScaleSW, PortfolioSW, SWSlicer,
                         aggregate, conv_sw, csw, sliced_wasserstein)
from .testing import auc, combine, level_weights, maxlevel_z, power_at
from .weights import slice_weights, SIGMA_SHAPE
from .api import TestResult, distance, stationarity_index, test
from .pairgroup import block_plan, patterns_for
from .portfolio import Portfolio, core_columns, evaluate, portfolios
from .joint import JointSW
from .sequential import SequentialCertifier
from . import components, spectral

__all__ = [
    "test", "distance", "stationarity_index", "TestResult",
    "MultiScaleSW", "PortfolioSW", "SWSlicer", "CSWSlicer",
    "SequentialCertifier",
    "get_device", "set_device",
    "Pyramid", "make_bank", "dct_bank", "opponent_dct_bank", "gaussian_bank",
    "orthogonal_bank", "gabor_bank",
    "SliceFeatures", "blocked_values", "balanced_blocked_values",
    "paired_values", "concat_features",
    "aggregate", "combine", "level_weights", "maxlevel_z", "power_at", "auc",
    "slice_weights", "SIGMA_SHAPE",
    "sliced_wasserstein", "conv_sw", "csw",
    "w1d_sort", "w1d_quantile", "running_estimate",
    "Portfolio", "JointSW", "portfolios", "core_columns", "evaluate",
    "block_plan", "patterns_for",
    "components", "spectral",
]
