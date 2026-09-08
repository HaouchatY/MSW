"""Product ("joint") slice families: pointwise products of two band-pass responses.

Every slice family in :mod:`msw.estimators` is *marginal*: it pools the values of
a single linear response.  Such a slice can only ever measure the non-negative
frequency envelope ``int S(k) |w_hat(k)|^2``, and since ``|w_hat|`` of a real
filter is symmetric under ``k -> -k``, no marginal slice -- of any bank, at any
scale -- can tell a +45 degree structure from a -45 degree one.  That blind spot
is provable, and it is what the families here close.

Each family is a pointwise feature map of the image: a product of two band-pass
responses, at the same or at neighbouring positions, at the same or at
neighbouring pyramid levels, rectified or not.  Nothing else about the machinery
changes -- the pooled values of such a map feed the same blocked rows and the
same per-image means, and the group argument is untouched, because a pair swap
relabels whole image blocks and says nothing about the feature map.

Families
--------
var        ``u = r_i(x)^2``                    always present; also the
           denominator of every normalised slice.
autoprod   ``u = r_i(x) r_i(x + d)``, d != 0.  Pooled mean =
           ``int S(k) |w_hat_i|^2 cos(2 pi k.d)`` -- a SIGNED frequency envelope.
           The lag pair (1,1)/(1,-1) is what separates the two diagonals.
crossprod  ``u = r_i(x) r_j(x)``               pooled mean = Cov(r_i, r_j).
magprod    ``u = |r_i^(l)(x)| |r_j^(l')(x)|``, ``l'`` in ``{l, l+1}`` (coarse
           resampled to the fine grid) -- the cross-orientation and cross-scale
           magnitude correlations that carry the Portilla-Simoncelli model.
magauto    ``u = |r_i(x)| |r_i(x + d)|``       magnitude autocorrelation.

Filter pairs ``(i, j)`` are chosen deliberately -- DCT index transposes
(orientation partners) and nearest-frequency neighbours -- never all pairs.
Slices are reported in Portilla-Simoncelli correlation form (see ``normalise``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .banks import Pyramid, make_bank
from .core import get_device
from .estimators import _Base
from .features import SliceFeatures

__all__ = ["JointSW", "LAGSETS"]

L1 = ((1, 0), (0, 1), (1, 1), (1, -1))
L2 = L1 + ((2, 0), (0, 2), (2, 1), (1, 2), (2, -1), (1, -2), (2, 2), (2, -2))
LAGSETS = {"1": ((1, 0),), "4": L1, "12": L2, "6": L1 + ((2, 0), (0, 2))}
ALL = ("var", "autoprod", "crossprod", "magprod", "magauto")


def _dct_index_pairs(k: int, channels: int):
    def bi(u, v):
        return u * k + v - 1
    pairs = []
    for u in range(k):
        for v in range(k):
            if u * k + v == 0:
                continue
            if u < v:
                pairs.append((bi(u, v), bi(v, u)))          # orientation partner
            if u + 1 < k:
                pairs.append((bi(u, v), bi(u + 1, v)))      # frequency neighbour
    pairs = sorted(set(pairs))
    out = []
    for c in range(channels):
        out += [(i * channels + c, j * channels + c) for i, j in pairs]
    return out


def _shift(t, dy, dx):
    h, w = t.shape[-2], t.shape[-1]
    y0, y1 = max(0, -dy), h - max(0, dy)
    x0, x1 = max(0, -dx), w - max(0, dx)
    return t[:, :, y0:y1, x0:x1], t[:, :, y0 + dy:y1 + dy, x0 + dx:x1 + dx]


class JointSW(_Base):
    """A bank of product feature maps over a blur/decimate pyramid.

    The DCT bank is used in full at every level (no random subsampling), so the
    family is deterministic given ``(channels, size, levels, k)``.
    """

    def __init__(self, channels: int, size: int, levels: int = 4, k: int = 5,
                 families=ALL, lags=L1, mag_lags=None, blur: str = "binom5",
                 sample_budget: int = 200_000, gray: bool = True,
                 seed: int | None = 0):
        dev = get_device()
        self.pyr = Pyramid(levels=levels, blur=blur)
        self.gray = bool(gray) and channels > 1
        channels = 1 if self.gray else channels
        self.size, self.channels, self.k = size, channels, k
        self.sample_budget = sample_budget
        self.families = tuple(f for f in ALL if f in families or f == "var")
        self.lags = tuple(d for d in lags if d != (0, 0))
        self.mag_lags = tuple(d for d in (mag_lags if mag_lags is not None else lags)
                              if d != (0, 0))
        with torch.no_grad():
            sizes = [t.shape[-1] for t in self.pyr.analyse(
                torch.zeros(1, channels, size, size, device=dev))]
        self.banks, self.scales = [], []
        for l, s_l in enumerate(sizes):
            if s_l < k + 2:
                break
            w = make_bank("dct", None, k, channels)
            self.banks.append(w)
            self.scales.append(1.0 / self.pyr.slice_norms(w, l))
        self.nL, self.nF = len(self.banks), self.banks[0].shape[0]
        self.pairs = _dct_index_pairs(k, channels)
        self._layout(dev)
        self.norm = None

    # ------------------------------------------------------------------ layout
    def _layout(self, dev):
        keys, sizes, den1, den2 = [], [], [], []
        var_base = {}
        i = 0
        rng = lambda a, n: list(range(a, a + n))
        for l in range(self.nL):
            keys.append(("var", l, None)); sizes.append(self.nF)
            var_base[l] = i
            den1 += rng(i, self.nF); den2 += rng(i, self.nF)
            i += self.nF
        V = lambda l, idx: [var_base[l] + int(j) for j in idx]
        allf = list(range(self.nF))
        ii = [p[0] for p in self.pairs]
        jj = [p[1] for p in self.pairs]
        if "autoprod" in self.families:
            for l in range(self.nL):
                for d in self.lags:
                    keys.append(("autoprod", l, d)); sizes.append(self.nF)
                    den1 += V(l, allf); den2 += V(l, allf); i += self.nF
        if "crossprod" in self.families:
            for l in range(self.nL):
                keys.append(("crossprod", l, None)); sizes.append(len(self.pairs))
                den1 += V(l, ii); den2 += V(l, jj); i += len(self.pairs)
        if "magprod" in self.families:
            for l in range(self.nL):
                keys.append(("magprod", l, "orient")); sizes.append(len(self.pairs))
                den1 += V(l, ii); den2 += V(l, jj); i += len(self.pairs)
            for l in range(self.nL - 1):
                keys.append(("magprod", l, "scale")); sizes.append(self.nF)
                den1 += V(l, allf); den2 += V(l + 1, allf); i += self.nF
        if "magauto" in self.families:
            for l in range(self.nL):
                for d in self.mag_lags:
                    keys.append(("magauto", l, d)); sizes.append(self.nF)
                    den1 += V(l, allf); den2 += V(l, allf); i += self.nF
        self.keys, self.level_sizes = keys, sizes
        self.n_slices = sum(sizes)
        self.den1 = torch.tensor(den1, device=dev)
        self.den2 = torch.tensor(den2, device=dev)
        n_var = self.nL * self.nF
        self.norm_rows = torch.arange(n_var, self.n_slices, device=dev)
        self.norm_level_sizes = sizes[self.nL:]
        self._pair_idx = (torch.tensor(ii, device=dev), torch.tensor(jj, device=dev))

    # ------------------------------------------------------------- feature maps
    def _map_iter(self, x_batch):
        if self.gray:
            x_batch = x_batch.mean(1, keepdim=True)
        lv = self.pyr.analyse(x_batch)
        r = [F.conv2d(lv[l], self.banks[l]) * self.scales[l].view(1, -1, 1, 1)
             for l in range(self.nL)]
        ii, jj = self._pair_idx
        for l in range(self.nL):
            yield r[l] ** 2
        if "autoprod" in self.families:
            for l in range(self.nL):
                for d in self.lags:
                    a, b = _shift(r[l], *d)
                    yield a * b
        if "crossprod" in self.families:
            for l in range(self.nL):
                yield r[l][:, ii] * r[l][:, jj]
        if "magprod" in self.families:
            am = [t.abs() for t in r]
            for l in range(self.nL):
                yield am[l][:, ii] * am[l][:, jj]
            for l in range(self.nL - 1):
                yield am[l] * F.interpolate(am[l + 1], size=am[l].shape[-2:],
                                            mode="nearest")
        if "magauto" in self.families:
            am = [t.abs() for t in r]
            for l in range(self.nL):
                for d in self.mag_lags:
                    a, b = _shift(am[l], *d)
                    yield a * b

    def _maps(self, x):
        return list(self._map_iter(x))

    # ------------------------------------------------------------------- scales
    @torch.no_grad()
    def fit(self, x_ref: torch.Tensor) -> "JointSW":
        """Fixed per-slice scale from a reference pool (data-independent)."""
        self.norm = [m.std(dim=(0, 2, 3)).clamp_min(1e-12).view(1, -1, 1, 1)
                     for m in self._maps(x_ref[: min(128, x_ref.shape[0])])]
        return self

    # ------------------------------------------------- pooled samples (W2 path)
    @torch.no_grad()
    def features(self, x: torch.Tensor, image_batch: int | None = None) -> SliceFeatures:
        if image_batch is None:
            image_batch = max(1, 2 ** 22 // (self.channels * self.size * self.size))
        n = x.shape[0]
        acc, per, sel = None, None, None
        for i in range(0, n, image_batch):
            ms = self._maps(x[i:i + image_batch])
            if self.norm is not None:
                ms = [m / q for m, q in zip(ms, self.norm)]
            if per is None:
                per, sel = [], []
                for m in ms:
                    hh = m.shape[-1] * m.shape[-2]
                    keep = max(1, min(hh, self.sample_budget // max(n, 1)))
                    per.append(keep)
                    sel.append(torch.arange(keep, device=m.device) * (hh // keep))
            cols = [m.reshape(m.shape[0], m.shape[1], -1)[:, :, s]
                    .permute(1, 0, 2).reshape(m.shape[1], -1)
                    for m, s in zip(ms, sel)]
            acc = cols if acc is None else [torch.cat([u, v], 1) for u, v in zip(acc, cols)]
        return SliceFeatures(acc, per, n)

    # ------------------------------------------------------------ group means
    @torch.no_grad()
    def group_means(self, x: torch.Tensor, groups: int = 8,
                    image_batch: int | None = None) -> torch.Tensor:
        """(groups, S) per-slice mean over ALL positions of all images of each
        disjoint image block -- one pyramid pass, O(one feature map) memory."""
        if image_batch is None:
            image_batch = max(1, 2 ** 24 // (self.channels * self.size * self.size))
        n = x.shape[0]
        per = n // groups
        gid = (torch.arange(n, device=x.device) // per).clamp_max(groups - 1)
        tot = torch.zeros(groups, self.n_slices, device=x.device, dtype=torch.float64)
        cnt = torch.zeros(groups, self.n_slices, device=x.device, dtype=torch.float64)
        for i in range(0, n, image_batch):
            gi = gid[i:i + image_batch]
            oh = torch.zeros(groups, gi.shape[0], device=x.device)
            oh.scatter_(0, gi.view(1, -1), 1.0)
            j = 0
            for m in self._map_iter(x[i:i + image_batch]):
                s = m.shape[1]
                tot[:, j:j + s] += (oh @ m.sum(dim=(2, 3))).double()
                cnt[:, j:j + s] += (oh.sum(1, keepdim=True) *
                                    (m.shape[2] * m.shape[3])).double()
                j += s
        return (tot / cnt.clamp_min(1)).float()

    def normalise(self, m: torch.Tensor) -> torch.Tensor:
        """Correlation-form slices ``mean(u_s) / sqrt(mean(v_i) mean(v_j))``.

        A ratio of two pooled means of the SAME block, i.e. still a fixed
        function of that block's data, so the relabelling argument is untouched.
        This is the Portilla-Simoncelli normalisation convention, and it is what
        makes these statistics precisely measured: the multiplicative
        fluctuation of the local contrast -- exactly where the per-slice noise of
        an unnormalised second-moment slice comes from -- cancels between
        numerator and denominator.
        """
        d = (m[:, self.den1] * m[:, self.den2]).clamp_min(1e-30).sqrt()
        return (m / d)[:, self.norm_rows]

    # ------------------------------------------------------------ image means
    @torch.no_grad()
    def image_means(self, x: torch.Tensor, image_batch: int | None = None) -> torch.Tensor:
        """(N, S) per-image mean of every slice feature over all its positions.

        One pyramid pass, O(one feature map) memory, no sort.  Block means and
        the pooled per-slice variance both derive from this, so the coherent
        statistic and its weights cost a single pass over the data.
        """
        if image_batch is None:
            image_batch = max(1, 2 ** 24 // (self.channels * self.size * self.size))
        n = x.shape[0]
        out = torch.empty(n, self.n_slices, device=x.device, dtype=torch.float32)
        for i in range(0, n, image_batch):
            j = 0
            for m in self._map_iter(x[i:i + image_batch]):
                s = m.shape[1]
                out[i:i + m.shape[0], j:j + s] = m.mean(dim=(2, 3))
                j += s
        return out
