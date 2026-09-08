"""CPU test suite for msw: exactness properties, calibration, determinism.

The heavier gates live next door: ``test_relabel.py`` (the sign algebra against
physical relabelling), ``test_size.py`` (rejection rate under H0 on hundreds of
draws) and ``test_agreement.py`` (agreement with the reference implementation).
"""
import math

import pytest
import torch

import msw
from msw.features import balanced_blocked_values, balanced_group_sizes, blocked_values


def smooth(n, c=3, size=16, seed=0):
    """Random spatially-smooth images: low-pass filtered white noise."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, c, size, size, generator=g)
    k = torch.ones(c, 1, 5, 5) / 25.0
    x = torch.nn.functional.conv2d(
        torch.nn.functional.pad(x, (2, 2, 2, 2), mode="circular"), k, groups=c)
    return 0.5 + 0.2 * x


def test_floor_zero_mean():
    """E[t] = 0 under H0: mean blocked value over same-distribution pairs."""
    est = msw.MultiScaleSW(3, 16, levels=2, k=5, n_filters=None, bank="dct")
    means = []
    for r in range(20):
        a, b = smooth(32, seed=100 + r), smooth(32, seed=900 + r)
        tg = est.blocked(a, b, groups=4)
        means.append(float(tg.mean()))
    t = torch.tensor(means)
    se = float(t.std() / math.sqrt(len(means)))
    assert abs(float(t.mean())) < 3 * se + 1e-12


def test_signflip_null_and_alternative():
    ps = []
    for r in range(20):
        res = msw.test(smooth(32, seed=r), smooth(32, seed=500 + r))
        ps.append(res.p)
    m = sum(ps) / len(ps)
    assert 0.2 < m < 0.8, f"null p-values not centered: mean={m}"
    g = torch.Generator().manual_seed(7)
    a = smooth(32, seed=1)
    b = smooth(32, seed=2) + 0.3 * torch.randn(32, 3, 16, 16, generator=g)
    res = msw.test(a, b)
    assert res.p <= 0.05


def test_presort_equivalence():
    est = msw.MultiScaleSW(3, 16, levels=2, k=5, n_filters=None, bank="dct")
    fa, fb = est.features(smooth(32, seed=3)), est.features(smooth(32, seed=4))
    fast = blocked_values(fa, fb, groups=4, presort=True)
    slow = blocked_values(fa, fb, groups=4, presort=False)
    assert torch.allclose(fast, slow, atol=1e-7)


def test_balanced_splitting_uses_all_pairs():
    assert sum(balanced_group_sizes(25, 25)) == 12          # 24 of 25 images
    assert sum(balanced_group_sizes(8, 8)) == 4             # all 8 images


def test_block_plan_and_group_size():
    """L equal blocks; the group is enumerated in full whenever it is small."""
    r = msw.test(smooth(25, seed=5), smooth(25, seed=6))
    # 16 blocks of 1 image: an awkward N leaves N mod L images unused
    assert (r.L, r.M, r.n_used) == (16, 1, 16)
    assert r.group_size == 4096                             # parity subgroup
    r8 = msw.test(smooth(8, seed=5), smooth(8, seed=6))
    assert (r8.L, r8.M, r8.n_used, r8.G) == (8, 1, 8, 8)
    assert r8.group_size == 256                             # the FULL group 2^8
    r12 = msw.test(smooth(24, seed=5), smooth(24, seed=6), L=12)
    assert (r12.L, r12.M, r12.n_used, r12.group_size) == (12, 2, 24, 4096)


def test_balanced_matches_paper_at_round_n():
    est = msw.MultiScaleSW(3, 16, levels=2, k=5, n_filters=None, bank="dct")
    fa, fb = est.features(smooth(32, seed=8)), est.features(smooth(32, seed=9))
    assert torch.allclose(balanced_blocked_values(fa, fb, groups=4),
                          blocked_values(fa, fb, groups=4), atol=1e-7)


def test_determinism():
    a, b = smooth(24, seed=11), smooth(24, seed=12)
    r1, r2 = msw.test(a, b), msw.test(a, b)
    assert r1.T == r2.T and r1.p == r2.p and r1.components == r2.components


def test_bank_is_deterministic_and_ignores_the_global_rng():
    """The DCT bank must not depend on what the process drew before it.

    It used to: the subsample was taken with a generator-less ``torch.randperm``,
    i.e. from the global RNG, so the bank -- and every number downstream of it --
    silently depended on unrelated draws elsewhere in the program.
    """
    b1 = msw.make_bank("dct", 40, 5, 3)
    torch.randn(1000)                                        # disturb the global RNG
    torch.manual_seed(12345)
    b2 = msw.make_bank("dct", 40, 5, 3)
    assert torch.equal(b1, b2)
    assert torch.equal(msw.make_bank("dct", None, 5, 3), msw.make_bank("dct", 999, 5, 3))
    e1 = msw.MultiScaleSW(3, 16, levels=2, k=5, n_filters=40, bank="dct", seed=0)
    torch.manual_seed(999)
    e2 = msw.MultiScaleSW(3, 16, levels=2, k=5, n_filters=40, bank="dct", seed=0)
    assert all(torch.equal(x.filters, y.filters) for x, y in zip(e1.layouts, e2.layouts))


def test_small_n_can_reject_at_five_percent():
    """N = 8 per side is a real test: 2^8 = 256 group elements, granularity 1/256.

    Before 0.2.0 the group had one element per disjoint image GROUP (four of them
    at N = 8, 16 patterns), so the smallest attainable p-value was 1/16 = 0.0625
    and rejection at alpha = 0.05 was arithmetically impossible.
    """
    g = torch.Generator().manual_seed(3)
    a = smooth(8, seed=41)
    b = smooth(8, seed=42) + 0.5 * torch.randn(8, 3, 16, 16, generator=g)
    r = msw.test(a, b)
    assert r.group_size == 256 and r.p <= 0.05


def test_equal_block_layout_is_enforced():
    from msw.components import block_rows, to_double
    est = msw.MultiScaleSW(3, 16, levels=2, k=5, n_filters=None, bank="dct")
    fa = to_double(est.features(smooth(16, seed=51)))
    fb = to_double(est.features(smooth(8, seed=52)))         # half as many images
    with pytest.raises(ValueError):
        block_rows(fa, fb, 16, 1)


def test_distance_ci_orders():
    a, b = smooth(32, seed=13), smooth(32, seed=14)
    d, lo, hi = msw.distance(a, b, groups=8)                 # deprecated alias of L
    assert 0 <= lo <= hi and d >= 0
    assert msw.distance(a, b, L=8) == (d, lo, hi)


def test_distance_tracks_a_real_difference():
    a = smooth(64, seed=17)
    b = smooth(64, seed=18) * 1.5                            # a much wider marginal
    d0 = msw.distance(a, smooth(64, seed=19))[0]
    d1, lo1, _ = msw.distance(a, b)[0], *msw.distance(a, b)[1:]
    assert d1 > lo1 >= 0 and d1 > 5 * d0


def test_stationarity_index_orders():
    grf = smooth(64, seed=15)                                # stationary-ish
    g = torch.Generator().manual_seed(16)
    nonstat = grf.clone()                                    # texture energy varies
    nonstat[..., 8:] += 0.3 * torch.randn(64, 3, 16, 8, generator=g)  # with position
    assert msw.stationarity_index(nonstat) > 3 * msw.stationarity_index(grf)


def test_sequential_smoke():
    seq = msw.SequentialCertifier(3, 16, alpha=0.05)
    g = torch.Generator().manual_seed(21)
    for r in range(6):
        a = smooth(16, seed=300 + r)
        b = smooth(16, seed=700 + r) + 0.5 * torch.randn(16, 3, 16, 16, generator=g)
        seq.update(a, b)
    assert seq.e_value > 0 and seq.n_batches == 6
