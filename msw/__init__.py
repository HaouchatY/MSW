"""msw -- Multi-Scale Sliced Wasserstein distances and calibrated two-sample
tests between image datasets, built for stationary (texture-like) image
distributions.

Quick start::

    import msw
    r = msw.test(A, B)        # A, B: (N, C, n, n) tensors, any device
    r.T, r.p                  # portfolio statistic, EXACT sign-flip p-value
    r.diagnose()              # "difference concentrated at level 0, opponent chroma ..."
    d, lo, hi = msw.distance(A, B)
"""

__version__ = "0.1.0"

from .core import get_device, set_device, w1d_quantile, w1d_sort, running_estimate
from .banks import (Pyramid, dct_bank, gabor_bank, gaussian_bank, make_bank,
                    opponent_dct_bank, orthogonal_bank)
from .features import (SliceFeatures, balanced_blocked_values, blocked_values,
                       concat_features, paired_values)
from .estimators import (CSWSlicer, MultiScaleSW, PortfolioSW, SWSlicer,
                         aggregate, conv_sw, csw, sliced_wasserstein)
from .testing import auc, combine, level_weights, maxlevel_z, power_at
from .api import TestResult, distance, stationarity_index, test
from .sequential import SequentialCertifier
from . import spectral

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
    "sliced_wasserstein", "conv_sw", "csw",
    "w1d_sort", "w1d_quantile", "running_estimate",
    "spectral",
]
