"""VALIDITY GATE: the sign algebra against physical relabelling.

Row ``e`` of the orbit table claims to be the statistic that would be observed
if the image blocks had been relabelled by group element ``e``.  Here we
actually perform the relabelling -- physically swapping the matched image blocks
between the two datasets -- and recompute the ENTIRE statistic from scratch: the
features, the sorts, the Gram matrix, every component.  Row 0 of the recomputed
table must equal row ``e`` of the original one.

This is the only check that cannot be fooled by an algebra error shared between
the statistic and its randomization: an error in the sign rules would move both
sides of an internal consistency check the same way, but it cannot survive a
recomputation from the relabelled pixels.
"""
import pytest
import torch

import msw
from msw.components import std_floor
from msw.pairgroup import block_plan, patterns_for
from msw.portfolio import Portfolio, portfolios


def synth(n, c=3, size=16, seed=0):
    """Spatially smooth random images (low-pass filtered white noise)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, c, size, size, generator=g)
    k = torch.ones(c, 1, 5, 5) / 25.0
    x = torch.nn.functional.conv2d(
        torch.nn.functional.pad(x, (2, 2, 2, 2), mode="circular"), k, groups=c)
    return 0.5 + 0.2 * x


def relabel(a, b, s, L, M):
    """Physically swap the i-th matched block whenever ``s[i] < 0``."""
    a2, b2 = a.clone(), b.clone()
    for i in range(L):
        if s[i] < 0:
            sl = slice(i * M, (i + 1) * M)
            a2[sl], b2[sl] = b[sl], a[sl]
    return a2, b2


def _audit(port, a, b, elements):
    L, M = block_plan(a.shape[0], b.shape[0], port.L)
    pats = patterns_for(L, port.n_patterns, port.pattern_mode, device=port.device)
    A0 = port.table(a, b)
    worst, worst_name = 0.0, None
    for e in elements:
        e = min(e, pats.shape[0] - 1)
        A1 = port.table(*relabel(a, b, pats[e].cpu(), L, M))
        d = (A1[0] - A0[e]).abs() / A0[e].abs().clamp_min(1e-30)
        k = int(d.argmax())
        if float(d[k]) > worst:
            worst, worst_name = float(d[k]), port.names[k]
    return worst, worst_name, A0, pats


@pytest.mark.parametrize("channels,n,n_patterns", [(1, 32, 256), (3, 32, 256), (3, 8, 4096)])
def test_relabelling_matches_the_sign_algebra(channels, n, n_patterns):
    port = Portfolio(channels, 16, levels=2, n_patterns=n_patterns)
    a, b = synth(n, channels, seed=1), synth(n, channels, seed=2)
    worst, name, _, _ = _audit(port, a, b, (1, 7, 63, 200))
    assert worst < 1e-8, f"relabelling deviates by {worst:.3e} at {name}"


def test_p_value_is_a_function_of_the_orbit():
    """A relabelled dataset must get the p-value of its own row in the orbit.

    The pattern set is a subgroup, so ``e . S = S``: relabelling permutes the
    orbit without changing it, and the rank of the observed value -- the
    p-value -- can only be the rank of row e in the original table.
    """
    port = Portfolio(3, 16, levels=2, n_patterns=256)
    a, b = synth(32, 3, seed=3), synth(32, 3, seed=4)
    L, M = block_plan(32, 32, port.L)
    pats = patterns_for(L, port.n_patterns, port.pattern_mode, device=port.device)
    A0 = port.table(a, b)
    pf = portfolios(port.names)

    def p_of_row(A, row, cols):
        S = A[:, cols]
        Z = (S - S.mean(0)) / std_floor(S)
        T = Z.max(1).values
        return (1.0 + float((torch.cat([T[:row], T[row + 1:]]) >= T[row]).sum())) / S.shape[0]

    tol = 1.0 / A0.shape[0] + 1e-12       # one rank: a tie can break either way
    for e in (1, 5, 42):
        A1 = port.table(*relabel(a, b, pats[e].cpu(), L, M))
        dev = max(abs(p_of_row(A1, 0, c) - p_of_row(A0, e, c)) for c in pf.values())
        assert dev <= tol, f"p-value not a function of the orbit (dev {dev})"


def test_interleaving_lets_pair_aligned_components_reject():
    """The parity subgroup must be interleaved, or half the components go blind.

    With the contiguous layout the sign vector of a pair-aligned row set is
    pinned by the block parities: such a component then takes the SAME value on
    every group element and can never reject.  Interleaving is what breaks that
    degeneracy, so it is a correctness property, not a cosmetic one.
    """
    L, n_pat = 16, 4096
    for mode, want in (("contig", True), ("subgroup", False)):
        P = patterns_for(L, n_pat, mode, device=torch.device("cpu"))
        pair = P[:, 0::2] * P[:, 1::2]              # what a pair-aligned row sees
        # products of pair-products within one parity block: pinned to +1 iff the
        # block holds whole pairs, which is exactly the contiguous layout
        pinned = bool((pair[:, 0::2] * pair[:, 1::2] == 1).all())
        assert pinned is want
