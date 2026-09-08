"""Filter banks and the multi-scale (pyramid) analysis operator.

Design rule enforced throughout: **every slice is a unit-norm direction of the
original image space.**  A k x k filter applied at pyramid level l corresponds,
after unrolling the blur/decimate chain, to a single linear functional
``x -> <h, x>`` on the input image; we compute that composite filter ``h``
exactly (adjoint of the analysis operator) and rescale so that ||h||_2 = 1.

That is what lets the estimator be compared to W_2 (see the accompanying paper): a family of
unit-norm slices always gives a *lower* bound on the per-pixel-normalised W_2.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .core import get_device

__all__ = [
    "BLUR_KERNELS",
    "gaussian_bank",
    "orthogonal_bank",
    "dct_bank",
    "gabor_bank",
    "make_bank",
    "Pyramid",
]

# 1-D blur prototypes (unnormalised; normalised to unit L1 at use time so the
# pyramid is an averaging operator and no DC gain is introduced).
BLUR_KERNELS = {
    "binom5": torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]),
    "binom3": torch.tensor([1.0, 2.0, 1.0]),
    "box2": torch.tensor([1.0, 1.0]),
}


# ---------------------------------------------------------------------------
# banks:  (F, C, k, k) tensors with unit Frobenius norm per filter
# ---------------------------------------------------------------------------
def _normalise(w: torch.Tensor) -> torch.Tensor:
    return w / w.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)


def gaussian_bank(n: int, k: int, channels: int = 1, zero_mean: bool = True,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """i.i.d. Gaussian filters, unit norm.  Uniform on the sphere of R^{C k^2}.

    `zero_mean` projects out the DC component so the filter responds to
    structure rather than to mean brightness; the resulting directions are
    uniform on the sphere of the DC-orthogonal subspace.
    """
    w = torch.randn(n, channels, k, k, device=get_device(), generator=generator)
    if zero_mean and k * k * channels > 1:
        w = w - w.mean(dim=(1, 2, 3), keepdim=True)
    return _normalise(w)


def orthogonal_bank(n: int, k: int, channels: int = 1, zero_mean: bool = True,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Stratified bank: concatenated random orthonormal frames of R^{C k^2}.

    Filters come in groups of D = C k^2 mutually orthogonal unit vectors, each
    group a Haar-random rotation of the canonical basis.  Same marginal law as
    `gaussian_bank` but the slice average has lower variance (the groups tile
    the sphere instead of clustering).
    """
    d = channels * k * k
    groups = math.ceil(n / d)
    mats = []
    for _ in range(groups):
        g = torch.randn(d, d, device=get_device(), generator=generator)
        q, _ = torch.linalg.qr(g)
        mats.append(q.t())  # rows are orthonormal
    w = torch.cat(mats, 0)[:n].reshape(n, channels, k, k)
    if zero_mean and d > 1:
        w = w - w.mean(dim=(1, 2, 3), keepdim=True)
    return _normalise(w)


def dct_bank(k: int, channels: int = 1, drop_dc: bool = True) -> torch.Tensor:
    """Deterministic separable 2-D DCT-II basis of the k x k window.

    A fixed, interpretable, training-free band-pass family: basis (u, v) has a
    frequency envelope concentrated around (u, v) / (2k).  Returned bank has
    (k^2 - drop_dc) * channels filters (one per (basis, channel) pair).
    """
    i = torch.arange(k, device=get_device(), dtype=torch.float32)
    u = torch.arange(k, device=get_device(), dtype=torch.float32)
    b = torch.cos(math.pi * (i.unsqueeze(0) + 0.5) * u.unsqueeze(1) / k)  # (k, k)
    b = b / b.norm(dim=1, keepdim=True)
    basis = (b.unsqueeze(1).unsqueeze(3) * b.unsqueeze(0).unsqueeze(2)).reshape(k * k, k, k)
    if drop_dc:
        basis = basis[1:]
    out = torch.zeros(basis.shape[0] * channels, channels, k, k, device=get_device())
    for c in range(channels):
        out[c :: channels, c] = basis
    return _normalise(out)


def gabor_bank(n: int, k: int, channels: int = 1, generator: torch.Generator | None = None,
               n_orient: int = 4) -> torch.Tensor:
    """Random band-pass (Gabor) filters: a cosine carrier under a Gaussian window.

    Frequency and orientation are drawn at random inside the band a k x k window
    can resolve.  The envelope |w_hat|^2 is much better localised in frequency
    than for an i.i.d. Gaussian filter, which is what a slice needs in order to
    resolve the power spectrum (band-pass envelopes resolve the power spectrum).
    """
    g = generator
    c = (k - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(k, device=get_device(), dtype=torch.float32) - c,
        torch.arange(k, device=get_device(), dtype=torch.float32) - c,
        indexing="ij",
    )
    sigma = k / 4.0
    win = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    total = n * channels if channels > 1 else n
    freq = torch.rand(total, device=get_device(), generator=g) * 0.5  # cycles / pixel, up to Nyquist
    ang = (torch.randint(n_orient, (total,), device=get_device(), generator=g).float() * math.pi / n_orient)
    ang = ang + torch.rand(total, device=get_device(), generator=g) * (math.pi / n_orient)
    phase = torch.rand(total, device=get_device(), generator=g) * 2 * math.pi
    proj = (xx.unsqueeze(0) * torch.cos(ang).view(-1, 1, 1)
            + yy.unsqueeze(0) * torch.sin(ang).view(-1, 1, 1))
    w = win.unsqueeze(0) * torch.cos(2 * math.pi * freq.view(-1, 1, 1) * proj + phase.view(-1, 1, 1))
    w = w - w.mean(dim=(1, 2), keepdim=True)
    if channels == 1:
        return _normalise(w.unsqueeze(1))
    out = w.reshape(n, channels, k, k)
    return _normalise(out)


def opponent_dct_bank(k: int, channels: int = 3, drop_dc: bool = True) -> torch.Tensor:
    """DCT atoms crossed with opponent color axes -- luminance (1,1,1)/sqrt3,
    R-G (1,-1,0)/sqrt2, B-Y (1,1,-2)/sqrt6 (the OpponentSIFT axes: van de Sande
    et al., TPAMI 2010; Ohta 1980; Buchsbaum & Gottschalk 1983).

    The decorrelating basis of natural-image color: chroma slices carry almost
    no image power ("quiet bands"), which concentrates class-composition signal
    and makes small additive noise stand out.  Spatially constant color casts are invisible (AC atoms).
    Falls back to the per-channel DCT bank when channels != 3.
    """
    if channels != 3:
        return dct_bank(k, channels, drop_dc=drop_dc)
    i = torch.arange(k, device=get_device(), dtype=torch.float32)
    u = torch.arange(k, device=get_device(), dtype=torch.float32)
    b = torch.cos(math.pi * (i.unsqueeze(0) + 0.5) * u.unsqueeze(1) / k)
    b = b / b.norm(dim=1, keepdim=True)
    basis = (b.unsqueeze(1).unsqueeze(3) * b.unsqueeze(0).unsqueeze(2)).reshape(k * k, k, k)
    if drop_dc:
        basis = basis[1:]
    vs = torch.tensor([[1., 1., 1.], [1., -1., 0.], [1., 1., -2.]], device=get_device())
    vs = vs / vs.norm(dim=1, keepdim=True)
    out = torch.stack([v.view(3, 1, 1) * a for a in basis for v in vs])
    return _normalise(out)


def _subsample(b: torch.Tensor, n: int | None, generator: torch.Generator | None,
               seed: int) -> torch.Tensor:
    """Keep `n` filters of a deterministic bank, chosen reproducibly.

    `n = None` (or `n` at least the bank size) keeps the whole bank, which is
    the default everywhere and involves no randomness at all.  When a strict
    subset is asked for, the permutation is drawn from an EXPLICIT generator --
    never from the global RNG, whose state the caller does not control, and
    which made the bank silently depend on everything else the process had
    drawn.  With no generator supplied the draw is made on the CPU from `seed`,
    so the same subset is selected whether the bank lives on CPU or GPU.
    """
    if n is None or n >= b.shape[0]:
        return b
    if generator is None:
        generator = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(b.shape[0], device=generator.device, generator=generator)[:n]
    return b[idx.to(b.device)]


def make_bank(kind: str, n: int | None, k: int, channels: int = 1,
              generator: torch.Generator | None = None, seed: int = 0,
              **kw) -> torch.Tensor:
    """Dispatch on a bank name: 'gauss' | 'ortho' | 'dct' | 'dctopp' | 'gabor'.

    `generator` seeds the random banks and, for the deterministic DCT banks, the
    choice of subset when `n` is smaller than the bank; `seed` is used to build
    a CPU generator when none is given, so every bank is reproducible without
    touching the global RNG.
    """
    if kind == "gauss":
        return gaussian_bank(n, k, channels, generator=generator, **kw)
    if kind == "ortho":
        return orthogonal_bank(n, k, channels, generator=generator, **kw)
    if kind == "gabor":
        return gabor_bank(n, k, channels, generator=generator, **kw)
    if kind == "dct":
        return _subsample(dct_bank(k, channels, **kw), n, generator, seed)
    if kind == "dctopp":
        return _subsample(opponent_dct_bank(k, channels, **kw), n, generator, seed)
    raise ValueError(f"unknown bank {kind!r}")


# ---------------------------------------------------------------------------
# pyramid
# ---------------------------------------------------------------------------
class Pyramid:
    """Blur-and-decimate analysis operator, plus the adjoint needed for norms.

    ``analyse(x)`` returns the list ``[x^(0), ..., x^(L-1)]`` with
    ``x^(l+1) = decimate_2( g * x^(l) )``, ``g`` a separable unit-L1 blur.

    ``composite(w, l)`` returns the equivalent level-0 filters of a level-l bank
    ``w``, i.e. ``A_l^T w`` where ``A_l`` is the analysis map: the value read at
    output pixel 0 equals ``<A_l^T w, x>``.  This is the adjoint chain
    (zero-upsample, correlate with g) applied ``l`` times; it is exact, cheap and
    vectorised over filters, and gives the per-slice normalisation.

    A Laplacian variant is deliberately *not* offered: its adjoint is not the
    blur/decimate chain, so `composite` -- and hence the unit-norm slice
    normalisation everything else relies on -- would no longer be exact.  It is
    also unnecessary: zero-mean level-l filters already act as band-passes on
    what the decimation left.
    """

    def __init__(self, levels: int = 4, blur: str = "binom5", padding_mode: str = "circular"):
        self.levels = levels
        self.padding_mode = padding_mode
        g1 = BLUR_KERNELS[blur].to(get_device())
        self.g1 = g1 / g1.sum()
        self.g2 = (self.g1.unsqueeze(1) * self.g1.unsqueeze(0))  # separable 2-D, unit L1
        self.pad = (self.g1.numel() - 1) // 2

    # -- forward -----------------------------------------------------------
    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        k = self.g2.expand(c, 1, *self.g2.shape)
        p = self.pad
        x = F.pad(x, (p, p, p, p), mode=self.padding_mode)
        return F.conv2d(x, k, groups=c)

    def analyse(self, x: torch.Tensor) -> list[torch.Tensor]:
        out, cur = [], x
        for l in range(self.levels):
            out.append(cur)
            if l < self.levels - 1:
                if cur.shape[-1] < 2 * self.g1.numel():
                    break
                cur = self._blur(cur)[:, :, ::2, ::2]
        return out

    # -- adjoint (composite filters) --------------------------------------
    def composite(self, w: torch.Tensor, level: int) -> torch.Tensor:
        """Level-0 equivalent of a level-`level` bank `w` of shape (F, C, k, k)."""
        h = w
        for _ in range(level):
            n_f, c, kh, kw = h.shape
            up = torch.zeros(n_f, c, 2 * kh - 1, 2 * kw - 1, device=h.device, dtype=h.dtype)
            up[:, :, ::2, ::2] = h  # zero-upsample = adjoint of decimation
            g = self.g2.expand(c, 1, *self.g2.shape)
            q = self.g1.numel() - 1  # *full* convolution: the support grows by 2p each level
            h = F.conv2d(F.pad(up, (q, q, q, q)), g, groups=c)  # g symmetric -> corr == conv
        return h

    def slice_norms(self, w: torch.Tensor, level: int) -> torch.Tensor:
        """||A_l^T w||_2 per filter — the factor that makes each slice a unit direction."""
        return self.composite(w, level).flatten(1).norm(dim=1)
