"""`msw.distance`, computed for datasets too large to hold on the GPU.

`msw.distance(a, b)` moves both datasets to the device and builds the pyramid of
every image at once, which at N = 30,000 images of 256 or 512 px does not fit in
32 GB.  Nothing about the estimator requires that: each image's pyramid and its
kept filter responses depend on that image alone, plus the dataset size N (which
fixes how many positions per image the sample budget allows).  So we feed images
chunk by chunk, keep only the selected responses, and then run the library's own
transport rows and U-statistic, one pyramid level at a time.

The result must be msw.distance itself, not a variant: `check()` compares the two
on data small enough for the library to take whole.
"""
from __future__ import annotations

import math

import torch

import msw
from msw import components as C
from msw.api import _auto_levels, _t975
from msw.estimators import MultiScaleSW
from msw.features import SliceFeatures
from msw.pairgroup import block_plan
from msw.weights import slice_weights


class MSWStream:
    """Feature extractor + distance with exactly the configuration of msw.distance."""

    def __init__(self, channels: int, size: int, n_total: int, L: int = 16,
                 seed: int = 0, levels: int | None = None, k: int = 5,
                 store: str = "cuda"):
        self.est = MultiScaleSW(channels, size, levels=levels or _auto_levels(size),
                                k=k, n_filters=None, bank="dct", seed=seed)
        self.channels, self.size, self.n_total, self.L = channels, size, n_total, L
        self.store = store
        self.reset()

    # -- streaming features ------------------------------------------------
    def reset(self):
        self._mats = [[] for _ in self.est.layouts]
        self._per = [None] * len(self.est.layouts)
        self._seen = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor):
        """Append a chunk of images (B, C, H, W), float in [0, 1], on the device."""
        pyr = self.est.pyr.analyse(x)
        for j, la in enumerate(self.est.layouts):
            y, q = self.est._level_features(pyr[la.level], la, self.n_total)
            self._mats[j].append(y.to(self.store))
            self._per[j] = q
        self._seen += x.shape[0]

    def features(self) -> SliceFeatures:
        if self._seen != self.n_total:
            raise ValueError(f"fed {self._seen} images, declared n_total={self.n_total}")
        f = SliceFeatures([torch.cat(m, dim=1) for m in self._mats], list(self._per),
                          self.n_total)
        self.reset()
        return f

    # -- the distance ------------------------------------------------------
    @torch.no_grad()
    def distance(self, fa: SliceFeatures, fb: SliceFeatures,
                 weighting: str = "P1s", signed: bool = True) -> dict:
        """msw.distance's arithmetic on precomputed features, one level at a time."""
        n = min(fa.n_images, fb.n_images)
        Lb, M = block_plan(n, n, self.L)
        if Lb < 3:
            raise ValueError("need at least 3 image blocks for the jackknife interval")
        dev = msw.core.get_device()
        psis = []
        for ma, mb, q in zip(fa.mats, fb.mats, fa.per_image):
            sa = C.to_double(SliceFeatures([ma.to(dev)], [q], fa.n_images))
            sb = C.to_double(SliceFeatures([mb.to(dev)], [q], fb.n_images))
            psi, _, _ = C.block_rows(sa, sb, Lb, M, want_block_sums=False)
            psis.append(psi)
            del sa, sb
        psi = torch.cat(psis, dim=2)
        w = torch.tensor(slice_weights(self.channels, self.est.level_sizes, scheme=weighting),
                         dtype=psi.dtype, device=psi.device)
        g = (psi * w).sum(2)
        g.fill_diagonal_(0.0)
        tot = float(g.sum()) / 2.0
        npair = Lb * (Lb - 1) / 2.0
        u = tot / npair
        rowsum = g.sum(1)
        npair_i = (Lb - 1) * (Lb - 2) / 2.0
        loo = (tot - rowsum.double()) / npair_i
        se = float(((Lb - 1) / Lb * ((loo - loo.mean()) ** 2).sum()).clamp_min(0).sqrt())
        tq = _t975(Lb - 1)
        root = (lambda v: math.copysign(math.sqrt(abs(v)), v)) if signed else \
               (lambda v: math.sqrt(max(0.0, v)))
        # leave-one-block-out U values: two sets measured on the same captions share
        # their block structure, so the difference of two distances can be jackknifed
        # pairwise (analyze.py) -- far tighter than combining two independent intervals
        return {"d": root(u), "lo": root(u - tq * se), "hi": root(u + tq * se),
                "u": u, "se_u": se, "L": Lb, "M": M, "loo_u": [float(v) for v in loo]}


def check(n: int = 512, size: int = 64, chunk: int = 37, seed: int = 0) -> float:
    """Streaming == msw.distance on a dataset small enough for the library."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, 3, size, size, generator=g)
    b = (torch.rand(n, 3, size, size, generator=g) ** 1.15)      # a real difference
    ref = msw.distance(a, b)
    s = MSWStream(3, size, n)
    dev = msw.core.get_device()
    for i in range(0, n, chunk):
        s.add(a[i:i + chunk].to(dev))
    fa = s.features()
    for i in range(0, n, chunk):
        s.add(b[i:i + chunk].to(dev))
    fb = s.features()
    out = s.distance(fa, fb)
    rel = max(abs(out[k] - r) / max(abs(r), 1e-30) for k, r in zip(("d", "lo", "hi"), ref))
    print(f"msw.distance {ref[0]:.10g} [{ref[1]:.10g}, {ref[2]:.10g}]")
    print(f"streaming    {out['d']:.10g} [{out['lo']:.10g}, {out['hi']:.10g}]")
    print(f"max relative difference {rel:.2e}")
    return rel


if __name__ == "__main__":
    import sys
    sys.exit(0 if check() < 1e-6 else 1)
