"""Sliced-Wasserstein estimators between two image datasets.

Every estimator exposes the same two things:

``features(images) -> SliceFeatures``  the 1-D samples of each slice, laid out
                                      image-major (see msw.features);
``slice_values(a, b)``                 per-slice W_p^p, floor-corrected.

Sharing the machinery means comparisons between estimators differ only in the
slices themselves.

Estimators
----------
``SWSlicer``       classic SW: a dense random unit direction gives one scalar per
                   image.
``MultiScaleSW``   this work: k x k filters on every level of a blur/decimate
                   pyramid, each rescaled so that the equivalent level-0 filter is
                   a unit vector, with output positions pooled into the slice's
                   1-D sample.  ``levels=1`` recovers single-scale pooled conv SW.
``CSWSlicer``      Nguyen & Ho (2022): a stack of unit-norm convolutions collapsing
                   each image to L scalars.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .banks import Pyramid, make_bank
from .core import DEVICE
from .features import SliceFeatures, blocked_values, paired_values

torch.backends.cudnn.benchmark = True

__all__ = ["aggregate", "SWSlicer", "MultiScaleSW", "CSWSlicer",
           "sliced_wasserstein", "conv_sw", "csw"]


def aggregate(wpp: torch.Tensor, mode: str = "mean", p: int = 2) -> float:
    """Combine per-slice W_p^p into a distance.

    'mean'    plain Monte-Carlo average (the definition of SW).
    'ebsw-e'  importance-sampling energy-based average with energy exp (Nguyen &
              Ho, NeurIPS 2023): weights = softmax(W_p^p); upper-bounds 'mean'.
    'ebsw-1'  same with the identity energy.
    'max'     max-slice (an upper bound; noisy with few slices).
    """
    v = wpp.double()
    if mode == "mean":
        val = v.mean()
    elif mode == "ebsw-e":
        val = (v * torch.softmax(v, 0)).sum()
    elif mode == "ebsw-1":
        val = (v * (v / v.sum().clamp_min(1e-300))).sum()
    elif mode == "max":
        val = v.max()
    else:
        raise ValueError(mode)
    return float(val.clamp_min(0) ** (1.0 / p))


class _Base:
    """Shared plumbing: features -> per-slice values / blocked estimates."""

    level_sizes: list[int]

    def features(self, x: torch.Tensor) -> SliceFeatures:  # pragma: no cover
        raise NotImplementedError

    def slice_values(self, a, b, p: int = 2, debias: bool = False) -> torch.Tensor:
        """Per-slice W_p^p.  `debias=True` subtracts the finite-sample floor."""
        return paired_values(self.features(a), self.features(b), p=p, debias=debias)

    def distance(self, a, b, p: int = 2, mode: str = "mean") -> float:
        """The distance itself: (weighted mean of per-slice W_p^p) ** (1/p)."""
        return aggregate(self.slice_values(a, b, p=p, debias=False), mode=mode, p=p)

    def blocked(self, a, b, groups: int = 4, p: int = 2) -> torch.Tensor:
        """(G, S) independent floor-corrected per-slice estimates -- see msw.features."""
        return blocked_values(self.features(a), self.features(b), groups=groups, p=p)

    def statistic(self, a, b, groups: int = 4, p: int = 2, mode: str = "invvar") -> float:
        """Two-sample test statistic; 0 in expectation under H0 (see msw.testing).

        mode "maxlevel": uniform mean within each slice family (pyramid level),
        then the largest per-family z over the blocks.  Inverse-variance across
        *all* slices is optimal only when every slice carries the same signal;
        when the difference is confined to one scale it buries the informative
        family under the high-weight uninformative ones (a confined 74-sd signal
        can aggregate down to 1.6).  The max over families is the level-
        aware repair; its null is calibrated empirically like everything else.
        """
        from .testing import combine

        tg = self.blocked(a, b, groups=groups, p=p)
        if mode == "maxlevel":
            zs, i = [], 0
            for c in self.level_sizes:
                tl = tg[:, i:i + c].mean(1)                     # (G,)
                se = (tl.std() / tg.shape[0] ** 0.5).clamp_min(1e-12)
                zs.append(float(tl.mean() / se))
                i += c
            return max(zs)
        return combine(tg, mode=mode)

    __call__ = distance


# ---------------------------------------------------------------------------
# classic SW
# ---------------------------------------------------------------------------
class SWSlicer(_Base):
    """Classic sliced Wasserstein: `n_proj` dense random unit directions."""

    def __init__(self, channels: int, size: int, n_proj: int = 512, zero_mean: bool = True,
                 seed: int | None = None, chunk: int = 2048):
        g = torch.Generator(device=DEVICE)
        if seed is not None:
            g.manual_seed(seed)
        d = channels * size * size
        th = torch.randn(n_proj, d, device=DEVICE, generator=g)
        if zero_mean:
            th -= th.mean(1, keepdim=True)
        self.dirs = th / th.norm(dim=1, keepdim=True)
        self.level_sizes = [n_proj]
        self.chunk = chunk

    def features(self, x: torch.Tensor) -> SliceFeatures:
        flat = x.reshape(x.shape[0], -1)
        out = torch.cat([flat @ self.dirs[i : i + self.chunk].t()
                         for i in range(0, self.dirs.shape[0], self.chunk)], dim=1)
        return SliceFeatures([out.t().contiguous()], [1], x.shape[0])


# ---------------------------------------------------------------------------
# multi-scale pooled-pixel SW  (this work)
# ---------------------------------------------------------------------------
@dataclass
class SliceLayout:
    """Bookkeeping for the slices produced at one pyramid level."""

    level: int
    filters: torch.Tensor    # (F, C, k, k)
    scale: torch.Tensor      # (F,) = 1 / ||A_l^T w||, makes every slice a unit direction
    crop: int                # positions dropped on each side, in level-l pixels
    blocks: int              # spatial pooling granularity at this level (<= grid size)
    pos: torch.Tensor | None # flat indices of the kept output positions, or None


class MultiScaleSW(_Base):
    """Multi-scale pooled-pixel sliced Wasserstein.

    Parameters
    ----------
    channels, size    image shape (C, size, size).
    levels            requested pyramid depth (capped by `size`).
    k                 filter side length at every level.
    n_filters         filters per level.
    bank              'gauss' | 'ortho' | 'dct' | 'gabor'.
    blocks            spatial pooling granularity.  1 pools every position into one
                      histogram, which is exact for translation-invariant data
                      (see the paper); b > 1 keeps b x b position groups as
                      separate slices, which spatially aligned datasets (faces,
                      centred digits) benefit from.
    max_pos           hard cap on output positions kept per image per level;
                      None (default) means "all of them".  Positions are
                      exchangeable under stationarity, so a fixed random subset is
                      an exact subsample -- but it is *not* free: it leaves the
                      value unbiased while inflating the variance; empirically a
                      position is worth almost as much as an image
                      (exponent -0.48 vs -0.53).  Keep as many as fit.
    sample_budget     cap on pooled samples per slice, `N * positions`.  This is
                      what actually bounds memory and sort time; positions per
                      image shrink as the dataset grows.
    normalise_slices  rescale each filter so the equivalent level-0 filter has unit
                      norm.  Keep True: it is what makes levels comparable and puts
                      every slice on the unit sphere of the image space.
    crop_margin       drop positions whose receptive field wraps around the torus.
                      Unnecessary for torus-stationary data, useful for photographs.
    """

    def __init__(self, channels: int, size: int, levels: int = 4, k: int = 3,
                 n_filters: int = 64, bank: str = "gauss", blur: str = "binom5",
                 blocks: int = 1, max_pos: int | None = None, sample_budget: int = 4_000_000,
                 crop_margin: bool = False, normalise_slices: bool = True,
                 seed: int | None = None):
        self.pyr = Pyramid(levels=levels, blur=blur)
        self.blocks, self.max_pos, self.sample_budget = blocks, max_pos, sample_budget
        self.size, self.channels = size, channels
        gen = torch.Generator(device=DEVICE)
        if seed is not None:
            gen.manual_seed(seed)
        with torch.no_grad():
            sizes = [t.shape[-1] for t in self.pyr.analyse(
                torch.zeros(1, channels, size, size, device=DEVICE))]
        self.layouts: list[SliceLayout] = []
        for l, s_l in enumerate(sizes):
            if s_l < k + 2:
                break
            w = make_bank(bank, n_filters, k, channels, generator=gen) \
                if bank not in ("dct", "dctopp") else make_bank(bank, n_filters, k, channels)
            nrm = self.pyr.slice_norms(w, l)
            scale = (1.0 / nrm) if normalise_slices else torch.ones_like(nrm)
            crop = 0
            if crop_margin and l > 0:
                support = self.pyr.composite(w, l).shape[-1]
                crop = max(0, min(int(-(-support // (2 ** (l + 1)))), (s_l - k) // 2))
            h = s_l - k + 1 - 2 * crop
            b_l = max(1, min(blocks, h))  # coarse levels may be smaller than `blocks`
            pos = self._choose_positions(h, b_l, gen)
            self.layouts.append(SliceLayout(l, w, scale, crop, b_l, pos))
        self.level_sizes = [la.filters.shape[0] * la.blocks ** 2 for la in self.layouts]
        self.n_slices = sum(self.level_sizes)

    # -- positions ---------------------------------------------------------
    def _choose_positions(self, h: int, b: int, gen) -> torch.Tensor | None:
        """A fixed random ordering of the h x h output grid, one per block.

        A prefix of it is taken at feature time (`_keep`), so the number of
        positions adapts to the dataset size instead of being fixed.
        """
        hb = h // b
        if self.max_pos is not None and self.max_pos >= hb * hb and b == 1:
            return None
        return torch.stack([torch.randperm(hb * hb, device=DEVICE, generator=gen)
                            for _ in range(b * b)])       # (b*b, hb*hb)

    def _keep(self, avail: int, n_images: int) -> int:
        """How many positions per image to use.

        Positions are *not* redundant for detection: sweeping them moves the
        smallest detectable difference with exponent -0.48, almost exactly the
        -0.53 of adding images.  So keep as many as the budget allows
        rather than a fixed small number -- an earlier default of 64 cost a
        factor 2.6 in detectable difference.  (Subsampling leaves the *value* of
        the statistic unbiased either way; what it inflates is the variance.)

        `n_images` is the size of the whole dataset, not of the current image
        batch -- the budget bounds `N * positions`, which is what the sort sees.
        """
        cap = avail if self.max_pos is None else min(self.max_pos, avail)
        return max(1, min(cap, self.sample_budget // max(n_images, 1)))

    # -- features ----------------------------------------------------------
    def _level_features(self, lvl: torch.Tensor, la: SliceLayout,
                        n_total: int) -> tuple[torch.Tensor, int]:
        y = F.conv2d(lvl, la.filters) * la.scale.view(1, -1, 1, 1)  # (N, F, h, h)
        c = la.crop
        if c > 0:
            y = y[:, :, c:-c, c:-c]
        n, f, h, _ = y.shape
        b = la.blocks
        if la.pos is None:
            if h * h * n_total <= self.sample_budget:
                return y.permute(1, 0, 2, 3).reshape(f, n * h * h), h * h
            keep = self._keep(h * h, n_total)
            sel = torch.arange(keep, device=y.device) * (h * h // keep)
            y = y.reshape(n, f, h * h)[:, :, sel]
            return y.permute(1, 0, 2).reshape(f, n * keep), keep
        hb = h // b
        y = y[:, :, : hb * b, : hb * b].reshape(n, f, b, hb, b, hb)
        y = y.permute(1, 2, 4, 0, 3, 5).reshape(f, b * b, n, hb * hb)
        keep = self._keep(la.pos.shape[1], n_total)
        idx = la.pos[:, :keep]
        y = torch.gather(y, 3, idx.view(1, b * b, 1, keep).expand(f, b * b, n, keep))
        return y.reshape(f * b * b, n * keep), keep

    @torch.no_grad()
    def features(self, x: torch.Tensor, image_batch: int | None = None) -> SliceFeatures:
        if image_batch is None:
            image_batch = max(1, 2 ** 23 // (self.channels * self.size * self.size))
        chunks = [self.pyr.analyse(x[i : i + image_batch])
                  for i in range(0, x.shape[0], image_batch)]
        mats, per = [], []
        for la in self.layouts:
            outs, q = zip(*[self._level_features(c[la.level], la, x.shape[0])
                            for c in chunks])
            mats.append(torch.cat(outs, dim=1))
            per.append(q[0])
        return SliceFeatures(mats, list(per), x.shape[0])

    # -- analysis helpers --------------------------------------------------
    def composite_filters(self) -> list[torch.Tensor]:
        """Level-0 equivalent unit-norm filters, one tensor (F, C, kh, kw) per level."""
        out = []
        for la in self.layouts:
            h = self.pyr.composite(la.filters, la.level)
            out.append(h / h.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1))
        return out

    def envelopes(self, size: int | None = None) -> torch.Tensor:
        """Frequency envelopes |h_hat|^2 / n^2 of every slice (see msw.spectral)."""
        from .spectral import envelopes_from_filters

        size = size or self.size
        return torch.cat([envelopes_from_filters(h, size) for h in self.composite_filters()])


# ---------------------------------------------------------------------------
# CSW  (Nguyen & Ho, NeurIPS 2022) -- reference baseline
# ---------------------------------------------------------------------------
class _ConvStack(nn.Module):
    """Stride-2 conv stack collapsing (N,C,S,S) -> (N,L); every filter unit-norm.

    Reproduces the `Conv_MNIST_Slicer` / `ConvSlicer` design of the CSW repo: the
    first conv mixes channels into L slice-channels, later convs are depthwise so
    the L slices stay independent, and a final kernel collapses to 1x1.
    """

    def __init__(self, channels: int, size: int, L: int = 32, bottom_width: int = 8):
        super().__init__()
        convs, in_ch, s, first = [], channels, size, True
        while s > bottom_width:
            convs.append(nn.Conv2d(in_ch, L, 4, 2, 1, bias=False, groups=1 if first else L))
            in_ch, first, s = L, False, (s + 2 - 4) // 2 + 1
        convs.append(nn.Conv2d(in_ch, L, s, 1, 0, bias=False, groups=1 if first else L))
        self.convs, self.L = nn.ModuleList(convs), L
        self.to(DEVICE)
        self.reset()

    @torch.no_grad()
    def reset(self):
        for c in self.convs:
            w = torch.randn_like(c.weight)
            c.weight.data = w / w.flatten(1).norm(dim=1).view(-1, 1, 1, 1)

    @torch.no_grad()
    def forward(self, x):
        for c in self.convs:
            x = c(x)
        return x.reshape(x.shape[0], -1)


class CSWSlicer(_Base):
    """`n_slicers` independent CSW slicers, giving n_slicers * L slices."""

    def __init__(self, channels: int, size: int, n_slicers: int = 16, L: int = 32,
                 bottom_width: int = 8, seed: int | None = None, image_batch: int = 4096):
        if seed is not None:
            torch.manual_seed(seed)
        self.stacks = [_ConvStack(channels, size, L, bottom_width) for _ in range(n_slicers)]
        self.level_sizes = [n_slicers * L]
        self.image_batch = image_batch

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> SliceFeatures:
        cols = []
        for st in self.stacks:
            cols.append(torch.cat([st(x[i : i + self.image_batch])
                                   for i in range(0, x.shape[0], self.image_batch)]))
        return SliceFeatures([torch.cat(cols, dim=1).t().contiguous()], [1], x.shape[0])


# ---------------------------------------------------------------------------
# thin functional wrappers (kept for the notebooks)
# ---------------------------------------------------------------------------
def sliced_wasserstein(a, b, n_proj: int = 512, p: int = 2, debias: bool = False, **kw):
    return SWSlicer(a.shape[1], a.shape[-1], n_proj, **kw).slice_values(a, b, p=p, debias=debias)


def conv_sw(a, b, k: int = 3, n_filters: int = 64, p: int = 2, debias: bool = False, **kw):
    return MultiScaleSW(a.shape[1], a.shape[-1], levels=1, k=k, n_filters=n_filters,
                        **kw).slice_values(a, b, p=p, debias=debias)


def csw(a, b, n_slicers: int = 16, L: int = 32, p: int = 2, debias: bool = False, **kw):
    return CSWSlicer(a.shape[1], a.shape[-1], n_slicers, L, **kw).slice_values(
        a, b, p=p, debias=debias)


class PortfolioSW:
    """Two slice families, one calibrated bet each.

    Family 1: the per-channel DCT bank (the paper's default estimator).
    Family 2: the pure-chroma slices of the opponent bank (R-G, B-Y atoms) --
    the quiet bands of natural-image color.  `values(a, b)` returns the two
    floor-corrected inverse-variance values; the detection statistic is
    T = max_f (v_f - mu0_f) / sd0_f with per-family null moments estimated by
    the usual matched-split protocol.  Rationale: a single inverse-variance
    weighting across heterogeneous families bets everything on the quietest
    slices (weight share ~99.9%), which is a 10x win when the signal is
    chromatic and total blindness when it is pure luminance (CIFAR checker:
    per-slice z=+157 at 0.1% weight).  Holding one calibrated bet per family
    keeps both.  On 1-channel data the chroma family is empty and the
    portfolio degenerates to the plain estimator.
    """

    def __init__(self, channels: int, size: int, levels: int = 4, k: int = 5,
                 n_filters: int = 64, seed: int | None = 0, blocks: int = 1):
        self.base = MultiScaleSW(channels, size, levels=levels, k=k,
                                 n_filters=n_filters, bank="dct", seed=seed,
                                 blocks=blocks)
        self.opp = None
        self._chroma_idx = None
        if channels == 3:
            self.opp = MultiScaleSW(channels, size, levels=levels, k=k,
                                    n_filters=None, bank="dctopp", seed=seed,
                                    blocks=blocks)
            idx, i = [], 0
            for c in self.opp.level_sizes:
                idx += [i + j for j in range(c) if j % 3 != 0]
                i += c
            self._chroma_idx = idx

    def values(self, a, b, groups: int = 8) -> tuple[float, float | None]:
        from .testing import combine
        v1 = combine(self.base.blocked(a, b, groups=groups), mode="invvar")
        if self.opp is None:
            return v1, None
        tg = self.opp.blocked(a, b, groups=groups)
        v2 = combine(tg[:, self._chroma_idx], mode="invvar")
        return v1, v2

    def components(self, a, b, groups: int = 8) -> list[float]:
        """The 2x2 component vector [dct-iv, dct-ml, chr-iv, opp-ml].

        {invvar, maxlevel} x {per-channel, opponent/chroma}: one bet per
        (aggregation x slice-family) prior.  The detection statistic is
        T = max_c (v_c - mu0_c)/sd0_c with per-component null moments from
        matched splits; each component owns a territory (invvar: distributed
        signals; maxlevel: scale-confined; per-channel: luminance-borne;
        chroma: the quiet color bands).  On 1-channel data only the first two
        exist.  Empirically calibrated like every statistic here.
        """
        from .testing import combine, maxlevel_z

        tg1 = self.base.blocked(a, b, groups=groups)
        out = [float(combine(tg1, mode="invvar")), maxlevel_z(tg1, self.base.level_sizes)]
        if self.opp is not None:
            tg2 = self.opp.blocked(a, b, groups=groups)
            out += [float(combine(tg2[:, self._chroma_idx], mode="invvar")),
                    maxlevel_z(tg2, self.opp.level_sizes)]
        return out
