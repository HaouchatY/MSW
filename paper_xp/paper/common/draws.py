"""The image distributions every paper experiment draws from.

Each setting is a pair (draw, alt): draw(n, g) samples n reference images,
alt(eps)(n, g) n images perturbed with strength eps, both from the torch
generator g.  A repetition r uses one generator seeded 70000 + r under the null
(two calls of draw) and 1000 r + 7 under the alternative (draw, then alt), on
CUDA; the same seeds on another device give different, equally valid draws.

  fields     stationary Gaussian fields, 32 px, spectrum |k|^-3 (unit variance);
             slope:    spectrum steepened by |k|^-eps at fixed power
             noise:    white noise of std eps added to the spectrum
             marginal: Student-t scale mixture with df = 1/eps, same spectrum
  t3         the same fields as a Student-t(3) scale mixture, every image
             standardized; slope as above.  The t mixing variable comes from
             torch's global generator, as in the original runs.
  celeba32   random 32-px crops of CelebA 64 px (16,000 images); blur / noise
  celeba64   whole CelebA 64-px images; shading = a random smooth field in the
             band 1-3 cycles per image, scaled to unit std, times eps
  cifar      CIFAR-10 train (50,000 images); blur / checkerboard (+-eps at
             Nyquist, clamped to [0, 1])
"""
import io
import math
import os

import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = os.environ.get("MSW_XP_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


# ------------------------------------------------------------- Gaussian fields
def freq_grid(size, device=DEV):
    """Radial frequency |k| in cycles per pixel on the DFT grid."""
    f = torch.fft.fftfreq(size, device=device)
    return torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)


def powerlaw_spectrum(size, p, device=DEV):
    """S_k ~ |k|^-p, DC removed, scaled to unit pixel variance."""
    k = freq_grid(size, device)
    s = torch.zeros_like(k)
    s[k > 0] = k[k > 0].pow(-p)
    return s * size ** 2 / s.sum()


def sample_field(spectrum, n, marginal="gauss", df=3.0, generator=None):
    """n fields with covariance eigenvalues `spectrum`: (n, 1, size, size).
    marginal='student' multiplies each field by an independent chi-square scale
    (unit variance for df > 2), keeping the spectrum's shape."""
    size = spectrum.shape[-1]
    z = torch.randn(n, size, size, device=spectrum.device, generator=generator,
                    dtype=torch.cfloat)
    x = torch.fft.ifft2(z * spectrum.sqrt()).real * (size * math.sqrt(2.0))
    if marginal == "gauss":
        return x.unsqueeze(1)
    chi2 = torch.distributions.Gamma(df / 2.0, 0.5).sample((n,)).to(x.device)
    s = torch.rsqrt(chi2 / df)
    if df > 2:
        s = s / math.sqrt(df / (df - 2.0))
    return (x * s.view(-1, 1, 1)).unsqueeze(1)


S0 = powerlaw_spectrum(32, 3.0)
KK = freq_grid(32)


def perturb(kind, eps):
    """The perturbed spectrum of the field benchmark."""
    if eps == 0:
        return S0.clone()
    if kind == "slope":
        s = S0 * KK.clamp_min(1e-8).pow(-eps)
        s[0, 0] = 0.0
        return s * (S0.sum() / s.sum())
    if kind == "noise":
        return S0 + eps ** 2
    raise ValueError(kind)


def standardize(x, eps=1e-8):
    """Per-image zero mean, unit variance."""
    f = x.reshape(x.shape[0], -1)
    return ((f - f.mean(1, keepdim=True)) / f.std(1, keepdim=True).clamp_min(eps)).reshape_as(x)


# ------------------------------------------------------------- natural images
def blur(x, sigma):
    """Exact Gaussian blur of std sigma pixels (frequency domain, periodic)."""
    if sigma == 0:
        return x
    f = torch.fft.fftfreq(x.shape[-1], device=x.device)
    h = torch.exp(-2 * (math.pi * sigma) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    return torch.fft.ifft2(torch.fft.fft2(x) * h).real


def random_crops(x, n, size=32, gen=None):
    """n random size x size crops, each from a random image of x."""
    dev, (_, c, h, w) = x.device, x.shape
    idx = torch.randint(0, x.shape[0], (n,), device=dev, generator=gen)
    oy = torch.randint(0, h - size + 1, (n,), device=dev, generator=gen)
    ox = torch.randint(0, h - size + 1, (n,), device=dev, generator=gen)
    r = torch.arange(size, device=dev)
    lin = ((oy[:, None] + r)[:, :, None] * w + (ox[:, None] + r)[:, None, :]).reshape(n, 1, -1)
    return torch.gather(x[idx].reshape(n, c, -1), 2,
                        lin.expand(n, c, size * size)).reshape(n, c, size, size)


_CACHE = {}


def celeba():
    """CelebA, 16,000 images at 64 px in [0, 1]: (16000, 3, 64, 64)."""
    if "celeba" not in _CACHE:
        blob = torch.load(os.path.join(DATA, "celeba_64_16000.pt"), map_location="cpu",
                          weights_only=False)
        _CACHE["celeba"] = blob["images"].to(DEV)
        _CACHE["celeba_male"] = (blob["attr"][:, list(blob["names"]).index("Male")] == 1).to(DEV)
    return _CACHE["celeba"]


def celeba_male():
    celeba()
    return _CACHE["celeba_male"]


def cifar():
    """CIFAR-10 train split in [0, 1]: (50000, 3, 32, 32), and its labels."""
    if "cifar" not in _CACHE:
        import pyarrow.parquet as pq
        from PIL import Image
        t = pq.read_table(os.path.join(DATA, "cifar10_train.parquet"))
        arr = np.stack([np.array(Image.open(io.BytesIO(r["bytes"]))) for r in t.column("img").to_pylist()])
        _CACHE["cifar"] = (torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0).to(DEV)
        _CACHE["cifar_label"] = torch.tensor(t.column("label").to_pylist(), device=DEV)
    return _CACHE["cifar"]


def cifar_labels():
    cifar()
    return _CACHE["cifar_label"]


# ------------------------------------------------------------- the nine settings
def _fields(kind):
    draw = lambda n, g: sample_field(S0, n, generator=g)
    if kind == "marginal":
        alt = lambda e: lambda n, g: sample_field(S0, n, marginal="student", df=1.0 / e, generator=g)
    else:
        alt = lambda e: lambda n, g: sample_field(perturb(kind, e), n, generator=g)
    return draw, alt


def _t3():
    draw = lambda n, g: standardize(sample_field(S0, n, marginal="student", df=3.0, generator=g))
    alt = lambda e: lambda n, g: standardize(
        sample_field(perturb("slope", e), n, marginal="student", df=3.0, generator=g))
    return draw, alt


def _celeba(kind):
    cel = celeba()
    if kind == "shading":
        k64 = freq_grid(64)
        mask = (k64 >= 1 / 64) & (k64 < 3 / 64)
        draw = lambda n, g: cel[torch.randperm(cel.shape[0], device=DEV, generator=g)[:n]]

        def alt(e):
            def f(n, g):
                b = draw(n, g)
                z = torch.randn(n, 1, 64, 64, device=DEV, generator=g)
                sh = torch.fft.ifft2(torch.fft.fft2(z) * mask).real
                return b + e * sh / sh.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            return f
        return draw, alt
    draw = lambda n, g: random_crops(cel, n, gen=g)
    if kind == "blur":
        alt = lambda e: lambda n, g: blur(random_crops(cel, n, gen=g), e)
    else:
        alt = lambda e: lambda n, g: random_crops(cel, n, gen=g) + e * torch.randn(
            n, 3, 32, 32, device=DEV, generator=g)
    return draw, alt


def _cifar(kind):
    x = cifar()
    draw = lambda n, g: x[torch.randperm(x.shape[0], device=DEV, generator=g)[:n]]
    if kind == "blur":
        alt = lambda e: lambda n, g: blur(draw(n, g), e)
    else:
        r = torch.arange(32, device=DEV)
        P = ((-1.0) ** (r[:, None] + r[None, :])).view(1, 1, 32, 32)
        alt = lambda e: lambda n, g: (draw(n, g) + e * P).clamp(0, 1)
    return draw, alt


SETTINGS = {
    "fields_slope": (lambda: _fields("slope")),
    "fields_noise": (lambda: _fields("noise")),
    "fields_marginal": (lambda: _fields("marginal")),
    "t3_slope": _t3,
    "celeba_blur": (lambda: _celeba("blur")),
    "celeba_noise": (lambda: _celeba("noise")),
    "celeba_shading": (lambda: _celeba("shading")),
    "cifar_blur": (lambda: _cifar("blur")),
    "cifar_checker": (lambda: _cifar("checker")),
}
NAMES = {
    "fields_slope": "Fields, slope", "fields_noise": "Fields, noise",
    "fields_marginal": "Fields, marginal (1/df)", "t3_slope": "t3 fields, slope",
    "celeba_blur": "CelebA, blur", "celeba_noise": "CelebA, added noise",
    "celeba_shading": "CelebA, shading", "cifar_blur": "CIFAR-10, blur",
    "cifar_checker": "CIFAR-10, checkerboard",
}


def setting(name):
    """(draw, alt) of one of the nine benchmark settings."""
    return SETTINGS[name]()


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def null_seed(r):
    return 70000 + r


def alt_seed(r):
    return 1000 * r + 7
