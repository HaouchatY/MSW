"""SIZE GATE: the rejection rate under H0 must sit at the nominal alpha.

The p-value is exact by construction -- it is the rank of the observed statistic
in its own orbit -- but "by construction" is worth nothing if the construction
has a bug, so it is measured: hundreds of null draws (same distribution on both
sides), alpha = 0.05, at a large N and at the smallest N the test supports.

Data are synthetic Gaussian random fields drawn by ``msw.spectral``, so the gate
needs no downloads and no external files.  It is slow by nature (a few hundred
full test runs), so it only runs when asked:

    MSW_RUN_SIZE_GATE=1 pytest tests/test_size.py -s

``MSW_SIZE_REPS`` overrides the number of draws (default 400 per cell).

Measured on Gaussian random fields, 32 px, 1200 draws per cell:

    N = 500   max-z 0.048   CCT 0.057   WY 0.050     (1 se = 0.0063)
    N = 8     max-z 0.037   CCT 0.013   WY 0.002

At N = 8 the orbit has 2^8 = 256 elements, so p-values are multiples of 1/256
and the rejection region ``p <= 0.05`` is really ``p <= 12/256 = 0.0469``: the
attainable size is bounded by 0.0469, not 0.05, and the observed 0.037 is that
bound minus the usual discreteness slack.  The two rank-based combiners are far
more conservative there -- they run out of resolution -- which is why max-z is
the one the library ships.
"""
import math
import os

import pytest
import torch

import msw
from msw.spectral import powerlaw_spectrum, sample_field

pytestmark = pytest.mark.skipif(os.environ.get("MSW_RUN_SIZE_GATE") != "1",
                                reason="set MSW_RUN_SIZE_GATE=1 to run the size gate")

REPS = int(os.environ.get("MSW_SIZE_REPS", 400))
ALPHA = 0.05


def _rate(n, reps, size=32, seed0=70_000, **kw):
    spec = powerlaw_spectrum(size, 3.0, total_power=1.0)
    dev = msw.get_device()
    rej = {"maxz": 0, "cct": 0, "wy": 0}
    for r in range(reps):
        g = torch.Generator(device=dev).manual_seed(seed0 + r)
        a, b = sample_field(spec, n, generator=g), sample_field(spec, n, generator=g)
        res = msw.test(a, b, **kw)
        rej["maxz"] += res.p <= ALPHA
        rej["cct"] += res.p_cct <= ALPHA
        rej["wy"] += res.p_wy <= ALPHA
    return {k: v / reps for k, v in rej.items()}


@pytest.mark.parametrize("n", [500, 8])
def test_size_at_nominal_alpha(n):
    reps = REPS
    out = _rate(n, reps)
    se = math.sqrt(ALPHA * (1 - ALPHA) / reps)
    print(f"\nN={n} reps={reps}  " +
          "  ".join(f"{k}={v:.3f}" for k, v in out.items()) +
          f"   (nominal {ALPHA}, 1 se = {se:.4f})")
    # The test is exact, so the size can only be at or below nominal; allow three
    # binomial standard errors above (a real failure) and, below, the room the
    # discreteness of a 2^L-element orbit needs.
    assert out["maxz"] <= ALPHA + 3 * se, out
    assert out["maxz"] >= ALPHA - 3 * se - 1.0 / min(2 ** n, 4096), out
