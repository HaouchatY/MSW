"""MSW through the public library only (msw, pinned in requirements.txt).

test        the exact test (msw.test, CORE portfolio, max-z combiner): (p, T)
dist        msw.distance: signed value and its 95% jackknife interval
"""
import warnings

import msw
import torch


def test(a, b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # stationarity warning on aligned data
        r = msw.test(a, b, **kw)
    return float(r.p), float(r.T)


def dist(a, b, **kw):
    d, lo, hi = msw.distance(a, b, **kw)
    return float(d), float(lo), float(hi)


@torch.no_grad()
def null_rep(draw, N, seed, gen, **kw):
    """(p, T) of one null repetition: two independent draws."""
    g = gen(seed)
    return test(draw(N, g), draw(N, g), **kw)


@torch.no_grad()
def alt_rep(draw, alt_fn, N, seed, gen, **kw):
    """(p, T) of one alternative repetition: a draw against a perturbed draw."""
    g = gen(seed)
    return test(draw(N, g), alt_fn(N, g), **kw)
