"""Sparse stochastic processes s = L^{-1} w, as in the original runs (exp73-76).

w are unit-variance white innovations; L is one of three whitening operators,
each in 1D (length-4096 signals) and 2D (32 x 32 torus).  Within an operator
every innovation law gives the same expected power spectrum.

  near    near-integrator 1/(1 - a e^{-iw}) per axis, a = 0.995 (the benchmark,
          exp76); control a = 0.9.  Innovations laplace, student4,
          bernlap_0.05, bernlap_0.005 (Bernoulli-Laplace at density p).
  smooth  1D: AR(1) response with a = 0.95, control 0.96 (exp73, unnormalized);
          2D: fractional |k|^{-3/2}, DC removed, control |k|^{-1.6} (exp74).
          Innovations laplace, student4, berngauss (Bernoulli-Gauss, p = 0.05).
  axis    integrator with the zero frequency (1D) / both axes (2D) zeroed,
          control the near-integrator a = 0.9 (exp75).  Innovations laplace,
          student4, bernlap_0.05.

Each operator's output is divided by a fixed constant: the pixel std of a
seed-0 Gaussian pool (128 signals / 256 fields), one constant per filter
(exp74: one constant shared by the filter and its control; exp73: none).
draw(op, dom, contrast, n, g) returns the pair (reference, other) from one
generator g, the reference always Gaussian; 1D signals come back as the
row-major 64 x 64 raster, the (N, 1, 64, 64) image msw.test accepts.
"""
import math

import torch

L1D, SZ = 4096, 32
LOG_TINY = math.log(2.0 ** -24)    # torch.rand can return 0 exactly: u = -0.5, log1p(-1) = -inf

CONTRASTS = {
    "near": ["laplace", "student4", "bernlap_0.05", "bernlap_0.005", "opshift"],
    "smooth": ["laplace", "student4", "berngauss", "opshift"],
    "axis": ["laplace", "student4", "bernlap_0.05", "opshift"],
}
NS = {"d1": [8, 32, 128], "d2": [8, 32, 128, 500]}


def innov(kind, shape, g, dev):
    if kind == "gaussian":
        return torch.randn(*shape, device=dev, generator=g)
    if kind == "laplace":
        u = torch.rand(*shape, device=dev, generator=g) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs()).clamp_min(LOG_TINY) / math.sqrt(2)
    if kind == "student4":
        z = torch.randn(*shape, device=dev, generator=g)
        chi = torch.randn(*shape, 4, device=dev, generator=g).pow(2).sum(-1)
        return z / (chi / 4).sqrt() / math.sqrt(2.0)
    if kind == "berngauss":
        z = torch.randn(*shape, device=dev, generator=g) / math.sqrt(0.05)
        return z * (torch.rand(*shape, device=dev, generator=g) < 0.05).float()
    if kind.startswith("bernlap"):
        p = float(kind.split("_")[1])
        u = torch.rand(*shape, device=dev, generator=g) - 0.5
        lap = -torch.sign(u) * torch.log1p(-2 * u.abs()).clamp_min(LOG_TINY) / math.sqrt(2)
        return lap * (torch.rand(*shape, device=dev, generator=g) < p).float() / math.sqrt(p)
    raise ValueError(kind)


def _ar(w, a):
    return 1.0 / (1.0 - a * torch.exp(-1j * w))


def _int(w):
    h = torch.zeros(w.shape, dtype=torch.cfloat)
    h[1:] = 1.0 / (1.0 - torch.exp(-1j * w[1:]))
    return h


def _sep(hx):
    return hx.view(1, SZ) * hx.view(SZ, 1)


def _frac(gamma):
    f = torch.fft.fftfreq(SZ)
    k = (f[:, None] ** 2 + f[None, :] ** 2).sqrt()
    k[0, 0] = 1.0
    h = k.pow(-gamma)
    h[0, 0] = 0.0
    return h


class Process:
    """One operator in one dimension: base filter H, control filter Hc."""

    def __init__(self, op, dom, dev):
        self.dom, self.dev = dom, dev
        w1 = 2 * math.pi * torch.fft.rfftfreq(L1D)
        w2 = 2 * math.pi * torch.fft.fftfreq(SZ)
        H, Hc = {("near", "d1"): (_ar(w1, 0.995), _ar(w1, 0.9)),
                 ("near", "d2"): (_sep(_ar(w2, 0.995)), _sep(_ar(w2, 0.9))),
                 ("smooth", "d1"): (_ar(w1, 0.95), _ar(w1, 0.96)),
                 ("smooth", "d2"): (_frac(1.5), _frac(1.6)),
                 ("axis", "d1"): (_int(w1), _ar(w1, 0.9)),
                 ("axis", "d2"): (_sep(_int(w2)), _sep(_ar(w2, 0.9)))}[(op, dom)]
        self.H, self.Hc = H.to(dev), Hc.to(dev)
        n0 = 128 if dom == "d1" else 256
        g0 = lambda: torch.Generator(device=dev).manual_seed(0)
        if op == "smooth" and dom == "d1":
            self.c = self.cc = 1.0
        else:
            self.c = float(self._filter(innov("gaussian", self._shape(n0), g0(), dev), self.H).std())
            self.cc = self.c if op == "smooth" else \
                float(self._filter(innov("gaussian", self._shape(n0), g0(), dev), self.Hc).std())

    def _shape(self, n):
        return (n, L1D) if self.dom == "d1" else (n, SZ, SZ)

    def _filter(self, w, H):
        if self.dom == "d1":
            return torch.fft.irfft(torch.fft.rfft(w) * H, n=L1D)
        return torch.fft.ifft2(torch.fft.fft2(w) * H).real

    def sample(self, kind, n, g, control=False):
        """n realisations: (n, 4096) in 1D, (n, 1, 32, 32) in 2D."""
        H, c = (self.Hc, self.cc) if control else (self.H, self.c)
        x = self._filter(innov(kind, self._shape(n), g, self.dev), H) / c
        return x if self.dom == "d1" else x.unsqueeze(1)

    def pair(self, contrast, n, g):
        """(Gaussian reference, other) as images for msw; contrast 'null' is Gaussian."""
        a = self.sample("gaussian", n, g)
        if contrast == "null":
            b = self.sample("gaussian", n, g)
        elif contrast == "opshift":
            b = self.sample("gaussian", n, g, control=True)
        else:
            b = self.sample(contrast, n, g)
        if self.dom == "d1":
            a, b = a.reshape(n, 1, 64, 64), b.reshape(n, 1, 64, 64)
        return a, b
