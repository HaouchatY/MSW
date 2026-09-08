"""AGREEMENT GATE: the same numbers as the implementation the paper measured.

Two halves.

``test_matches_reference_implementation`` runs the reference research
implementation and this library on IDENTICAL draws and compares p-values.  It
needs the reference tree -- the research scripts this release was ported from --
so it is skipped unless ``MSW_REFERENCE_PATH`` points at it (colon-separated
directories to put on ``sys.path``, from which the reference's own ``msw2`` and
``port2`` modules must be importable).  Measured on the paper's nine
settings at their certified perturbation strength, N = 500, 24 draws each: with
the reference configured to use the whole DCT bank -- the library's default --
the p-values are IDENTICAL to the last bit on all nine rows.  Against the
reference's own default (a random 64-of-72 subsample of the 3-channel bank, the
one non-determinism this release removes) the 1-channel rows are still identical
and the 3-channel rows differ by at most 0.0015, i.e. a few ranks out of 4096,
with every detect/no-detect verdict agreeing.

``test_golden_p_values`` needs nothing external: it pins the p-values of fixed
synthetic draws so that a refactor cannot quietly move them.
"""
import os
import sys

import pytest
import torch

import msw
from msw.spectral import powerlaw_spectrum, sample_field


def _fields(n, p=3.0, size=32, seed=0, marginal="gauss", df=3.0):
    """Gaussian random fields, drawn on the CPU so the DATA never depend on the
    device -- only the arithmetic of the statistic does."""
    cpu = torch.device("cpu")
    spec = powerlaw_spectrum(size, p, total_power=1.0, device=cpu)
    g = torch.Generator().manual_seed(seed)
    return sample_field(spec, n, marginal=marginal, df=df, generator=g)


@pytest.mark.skipif(not os.environ.get("MSW_REFERENCE_PATH"),
                    reason="set MSW_REFERENCE_PATH to the reference implementation")
def test_matches_reference_implementation():
    for d in os.environ["MSW_REFERENCE_PATH"].split(":"):
        sys.path.insert(0, d)
    import msw2 as ref_algebra          # noqa: E402
    import port2 as ref_portfolio       # noqa: E402

    ref = ref_portfolio.Port2(1, 32)
    for seed, alt in ((11, 3.0), (12, 3.10)):
        dev = msw.get_device()
        a = _fields(200, 3.0, seed=seed).to(dev)
        b = _fields(200, alt, seed=seed + 500).to(dev)
        A = ref.A(a, b)
        p_ref = ref_algebra.p_maxz(A[:, ref_portfolio.core_cols(ref.names)])
        assert msw.test(a, b).p == p_ref


@pytest.mark.parametrize("case,expect", [
    ("null", 0.17041015625),
    ("slope", 0.00048828125),
    ("marginal", 0.00048828125),
])
def test_golden_p_values(case, expect):
    """Fixed draws, pinned p-values: a refactor must not move them.

    Tolerance is a few ranks of the 4096-element orbit, which is what a change
    of floating-point summation order (a different device, say) can move.
    """
    a = _fields(200, 3.0, seed=1)
    b = {"null": lambda: _fields(200, 3.0, seed=2),
         "slope": lambda: _fields(200, 3.06, seed=2),
         "marginal": lambda: _fields(200, 3.0, seed=2, marginal="student", df=6.0),
         }[case]()
    p = msw.test(a, b).p
    assert abs(p - expect) <= 4 / 4096, f"{case}: p = {p!r}"
