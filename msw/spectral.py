"""Exact reference values for stationary Gaussian random fields on the torus.

Why this exists: for image datasets we never know the true W_2, so we cannot say
whether a sliced estimator is faithful.  For *stationary Gaussian fields* on the
discrete torus every quantity in the paper is available in closed form, so the
estimators can be validated against ground truth (the same role the Gibbs
sampler plays in Zach et al., "A Statistical Benchmark for Diffusion Posterior
Sampling Algorithms").

Conventions
-----------
A stationary field on Z_n^2 has circulant covariance
``Sigma = U* diag(S) U`` with ``U = F / n`` the unitary DFT, so

    * the eigenvalues of Sigma are exactly ``S_k`` (the power spectrum),
    * for any vector ``theta``, ``theta^T Sigma theta = (1/n^2) sum_k S_k |theta_hat_k|^2``.

Two consequences used everywhere below.

1.  Sigma_1 and Sigma_2 commute (same eigenbasis), so the Bures/W_2 distance is
    diagonal:

        W_2^2(N(0,S1), N(0,S2)) = sum_k ( sqrt(S1_k) - sqrt(S2_k) )^2.

    We report the *per-pixel* value ``W2bar = W_2 / sqrt(d)``, ``d = C n^2``,
    which is the RMS pixel displacement and the scale on which sliced
    distances live.

2.  A unit-norm slice ``theta`` sees a centred 1-D Gaussian of variance
    ``<u, S>`` with the **frequency envelope**

        u_k = |theta_hat_k|^2 / n^2,     sum_k u_k = ||theta||^2 = 1

    (Parseval).  Its per-slice ``W_2^2`` is therefore ``(sqrt<u,S1> - sqrt<u,S2>)^2``
    and *any* slicing family obeys

        D^2 = E_u ( sqrt<u,S1> - sqrt<u,S2> )^2
            <= E_u <u, (sqrt(S1) - sqrt(S2))^2>            (triangle ineq. in L2(u))
            =  (1/n^2) sum_k (sqrt(S1_k) - sqrt(S2_k))^2 = W2bar^2,

    using ``E[u_k] = 1/n^2``, which holds exactly for any zero-mean i.i.d. filter
    law.  Equality is approached when ``u`` concentrates on single frequencies.
    **The bound needs ``E[u_k] = 1/n^2``** (a frequency-unbiased family), which
    holds for dense SW directions and for a single i.i.d. k x k filter, but *not*
    for pyramid slices: blurring tilts the average envelope towards low
    frequency.  `resolution_tilt` therefore splits the sensitivity

        D / W2bar = (D / B) x (B / W2bar) = resolution x tilt,
        B^2 = sum_k E[u_k] (sqrt(S1_k) - sqrt(S2_k))^2,

    so that "how sharp are the slices in frequency" (resolution, in [0,1]) is
    separated from "does the family look where the spectra differ" (tilt).  The
    unconditional bound remains ``D <= Max-SW <= W_2``.
"""

from __future__ import annotations

import math

import torch

from .core import get_device

__all__ = [
    "freq_grid",
    "powerlaw_spectrum",
    "matern_spectrum",
    "sample_field",
    "w2_exact",
    "envelopes_from_filters",
    "dense_envelopes",
    "linear_slicer_envelopes",
    "population_slice_wpp",
    "population_distance",
    "deficiency",
    "relative_sensitivity",
    "effective_sample_size",
    "mean_envelope",
    "family_bound",
    "resolution_tilt",
    "max_sw_exact",
    "max_psw",
]


def freq_grid(size: int, device=None) -> torch.Tensor:
    """Radial frequency |k| in cycles/pixel on the fftshift-free DFT grid."""
    device = get_device() if device is None else device
    f = torch.fft.fftfreq(size, device=device)
    return torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)


def powerlaw_spectrum(size: int, p: float, total_power: float | None = 1.0,
                      device=None) -> torch.Tensor:
    """S_k proportional to |k|^-p, DC killed.

    `total_power=v` rescales so that the pixel variance ``(1/n^2) sum_k S_k``
    equals v.  Standardising this way is important: it removes the trivial
    "total power" difference between two spectra, so what remains is purely the
    *shape* of the correlation -- the regime where classic SW degenerates
    (see the paper).
    """
    device = get_device() if device is None else device
    k = freq_grid(size, device)
    s = torch.zeros_like(k)
    nz = k > 0
    s[nz] = k[nz].pow(-p)
    if total_power is not None:
        s *= total_power * size**2 / s.sum()
    return s


def matern_spectrum(size: int, rho: float, nu: float = 1.5, total_power: float | None = 1.0,
                    device=None) -> torch.Tensor:
    """Matern spectrum S_k ~ (1 + (2 pi rho |k|)^2)^{-(nu+1)} -- a light-tailed,
    single-correlation-length alternative to the scale-free power law."""
    device = get_device() if device is None else device
    k = freq_grid(size, device)
    s = (1.0 + (2 * math.pi * rho * k) ** 2).pow(-(nu + 1.0))
    s[0, 0] = 0.0
    if total_power is not None:
        s *= total_power * size**2 / s.sum()
    return s


def sample_field(spectrum: torch.Tensor, n: int, marginal: str = "gauss", df: float = 3.0,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw `n` fields with the given spectrum: (n, 1, size, size).

    `marginal` != 'gauss' applies a per-field scale mixture ``X = s * G`` which
    leaves the spectrum (hence the exact-W_2 reference for the Gaussian case
    unchanged in *shape*) but makes the pixel marginal heavy-tailed:
    'student' (chi^2 mixing, `df` degrees of freedom) or 'laplace'
    (exponential mixing).  Useful for testing robustness of the estimators;
    the closed-form W_2 above then no longer applies exactly.
    """
    size = spectrum.shape[-1]
    amp = spectrum.sqrt()  # sqrt of the covariance eigenvalues
    z = torch.randn(n, size, size, device=spectrum.device, generator=generator, dtype=torch.cfloat)
    # torch's ifft2 carries 1/n^2 and Re(.) halves the variance, hence n*sqrt(2):
    # Cov(x)_{m m'} = (1/n^2) sum_k S_k exp(i k.(m-m')), i.e. eigenvalues exactly S.
    x = torch.fft.ifft2(z * amp).real * (size * math.sqrt(2.0))
    if marginal == "gauss":
        return x.unsqueeze(1)
    if marginal == "student":
        chi2 = torch.distributions.Gamma(df / 2.0, 0.5).sample((n,)).to(x.device)
        s = torch.rsqrt(chi2 / df) / math.sqrt(df / (df - 2.0)) if df > 2 else torch.rsqrt(chi2 / df)
    elif marginal == "laplace":
        s = (-torch.log(torch.rand(n, device=x.device, generator=generator))).sqrt()
    else:
        raise ValueError(marginal)
    return (x * s.view(-1, 1, 1)).unsqueeze(1)


def w2_exact(s1: torch.Tensor, s2: torch.Tensor, channels: int = 1) -> tuple[float, float]:
    """Exact (W_2, per-pixel W2bar) between two centred stationary Gaussian fields."""
    w2sq = float(((s1.sqrt() - s2.sqrt()) ** 2).sum()) * channels
    d = channels * s1.numel()
    return math.sqrt(w2sq), math.sqrt(w2sq / d)


# ---------------------------------------------------------------------------
# frequency envelopes of a slicing family
# ---------------------------------------------------------------------------
def envelopes_from_filters(h: torch.Tensor, size: int) -> torch.Tensor:
    """Envelopes u (F, size, size) of unit-norm spatial filters h (F, C, kh, kw).

    The filter is zero-padded to the torus and ``u = |h_hat|^2 / n^2``; by
    Parseval ``u.sum() = ||h||^2``, so a unit-norm filter gives a probability
    vector over frequencies.
    """
    f, c, kh, kw = h.shape
    pad = torch.zeros(f, c, size, size, device=h.device, dtype=h.dtype)
    pad[:, :, :kh, :kw] = h
    u = torch.fft.fft2(pad).abs().pow(2).sum(1) / size**2
    return u


def dense_envelopes(size: int, n: int, channels: int = 1, zero_mean: bool = False,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Envelopes of `n` dense random unit directions (classic SW)."""
    th = torch.randn(n, channels, size, size, device=get_device(), generator=generator)
    if zero_mean:
        th -= th.mean(dim=(1, 2, 3), keepdim=True)
    th /= th.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return torch.fft.fft2(th).abs().pow(2).sum(1) / size**2


@torch.no_grad()
def linear_slicer_envelopes(slicer, size: int, channels: int = 1, batch: int = 256) -> torch.Tensor:
    """Envelopes of any *linear* slicer (e.g. `CSWSlicer`) by probing the basis.

    Feeding the d = C n^2 canonical basis images through the slicer returns the
    matrix of its directions, one column per slice; each column is then
    normalised and Fourier-transformed.  Exact, and cheap for d <= a few 10^4.
    """
    d = channels * size * size
    cols = []
    eye = torch.eye(d, device=get_device())
    for i in range(0, d, batch):
        cols.append(slicer(eye[i : i + batch].reshape(-1, channels, size, size)))
    m = torch.cat(cols)                      # (d, L) : row j = image basis j
    m = m / m.norm(dim=0, keepdim=True).clamp_min(1e-12)
    th = m.t().reshape(-1, channels, size, size)
    return torch.fft.fft2(th).abs().pow(2).sum(1) / size**2


# ---------------------------------------------------------------------------
# population values
# ---------------------------------------------------------------------------
def population_slice_wpp(u: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """Per-slice W_2^2 for Gaussian fields: (sqrt<u,S1> - sqrt<u,S2>)^2, shape (F,)."""
    v1 = (u * s1).flatten(1).sum(1).clamp_min(0).sqrt()
    v2 = (u * s2).flatten(1).sum(1).clamp_min(0).sqrt()
    return (v1 - v2) ** 2


def population_distance(u: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
                        mode: str = "mean") -> float:
    """D = ( E_u [per-slice W_2^2] )^{1/2}: the *population* value of the estimator.

    This is the quantity a sliced estimator converges to as the number of images
    and slices grows; comparing it to `w2_exact` separates the estimator's
    intrinsic blindness (bias of the slicing family) from its sampling error.
    """
    from .estimators import aggregate

    return aggregate(population_slice_wpp(u, s1, s2), mode=mode, p=2)


def mean_envelope(u: torch.Tensor) -> torch.Tensor:
    """ubar_k = E[u_k]: the average frequency weighting of a slicing family.

    ``ubar`` sums to 1.  For any i.i.d. filter law (dense SW directions, a single
    k x k random filter) ``ubar_k = 1/n^2`` exactly -- such families are
    *frequency-unbiased*.  Pyramid families are deliberately not: blurring tilts
    ``ubar`` towards low frequency, which is where the W_2 mass of a 1/f^p field
    lives.
    """
    return u.mean(0)


def family_bound(u: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> float:
    """B = sqrt( sum_k ubar_k (sqrt(S1_k) - sqrt(S2_k))^2 ).

    Cauchy-Schwarz / triangle inequality in L2(u) gives ``D <= B`` for every
    slicing family, with equality iff each envelope is concentrated on a single
    frequency.  For a frequency-unbiased family B = W2bar, which recovers the
    familiar ``D <= W_2 / sqrt(d)``.
    """
    d = (s1.sqrt() - s2.sqrt()) ** 2
    return float((mean_envelope(u) * d).sum().clamp_min(0).sqrt())


def resolution_tilt(u: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
                    channels: int = 1) -> dict:
    """Decompose the sensitivity of a slicing family:

        D / W2bar  =  (D / B)  x  (B / W2bar)
                      ^^^^^^^     ^^^^^^^^^^
                      resolution  tilt

    *resolution* in [0, 1] measures how concentrated in frequency the individual
    slices are (1 = single-frequency slices, i.e. no smoothing loss).
    *tilt* measures how well the family's average frequency weighting is matched
    to where the two spectra actually differ (1 = flat / unbiased).
    Their product is the total sensitivity relative to the true per-pixel W_2;
    values above 1 are legitimate -- the general bound is D <= Max-SW.
    """
    d_ = population_distance(u, s1, s2)
    b = family_bound(u, s1, s2)
    _, w2bar = w2_exact(s1, s2, channels)
    return {"D": d_, "B": b, "W2bar": w2bar,
            "resolution": d_ / max(b, 1e-300), "tilt": b / max(w2bar, 1e-300),
            "sensitivity": d_ / max(w2bar, 1e-300)}


def relative_sensitivity(u: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> float:
    """Scale-free per-slice discrepancy, r = RMS over slices of

        (sqrt<u,S1> - sqrt<u,S2>) / sqrt( (<u,S1> + <u,S2>) / 2 )
        = (sigma_1 - sigma_2) / sigma .

    Why this and not `deficiency`: ``D / W2bar`` is **not invariant to rescaling a
    slicing family**, so a family that simply reports larger numbers scores higher
    on it -- it cannot stand alone as a figure of merit.  Dividing each slice's
    discrepancy by that slice's own scale removes the freedom, and it is also the
    quantity detection depends on, because the sampling error of an estimated
    standard deviation is proportional to the standard deviation itself:

        Var(sigma_hat) / sigma^2 = 1 / (2 M_eff).

    Combining the two gives the detection signal-to-noise of a slice,

        SNR  ~  r * sqrt(M_eff) ,

    both factors scale-free.  See `effective_sample_size` for the second one and
    see the paper.
    """
    v1 = (u * s1).flatten(1).sum(1).clamp_min(0).sqrt()
    v2 = (u * s2).flatten(1).sum(1).clamp_min(0).sqrt()
    scale = (0.5 * (v1**2 + v2**2)).clamp_min(1e-300).sqrt()
    return float((((v1 - v2) / scale) ** 2).mean().sqrt())


@torch.no_grad()
def effective_sample_size(feature_fn, sampler, n_rep: int = 100) -> torch.Tensor:
    """M_eff per slice, measured -- the number of *independent* samples a slice has.

    `feature_fn(images) -> (S, M)` gives a slice's pooled 1-D sample; `sampler()`
    draws a fresh dataset.  Uses ``Var(sigma_hat)/sigma^2 = 1/(2 M_eff)`` over
    `n_rep` independent datasets.  For one-scalar-per-image slicers this returns
    the image count; for pooled slices it returns how much the `n^2` positions per
    image are actually worth, which is far less than `n^2` and shrinks fast with
    pyramid depth.
    """
    sig = torch.stack([feature_fn(sampler()).std(dim=1) for _ in range(n_rep)])
    return 0.5 / (sig.var(0) / sig.mean(0) ** 2).clamp_min(1e-30)


def deficiency(u: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, channels: int = 1) -> float:
    """D / W2bar: sensitivity relative to the true per-pixel W_2.

    NOT scale-free -- see `relative_sensitivity`.  Useful for *comparing bands
    within one family* (where the family's scale cancels), misleading for ranking
    families against each other.
    """
    _, w2bar = w2_exact(s1, s2, channels)
    return population_distance(u, s1, s2) / max(w2bar, 1e-300)


def max_sw_exact(s1: torch.Tensor, s2: torch.Tensor) -> float:
    """Max-SW for stationary Gaussian fields: attained at a single Fourier mode."""
    return float((s1.sqrt() - s2.sqrt()).abs().max())


def max_psw(s1: torch.Tensor, s2: torch.Tensor, k: int = 3, steps: int = 400,
            lr: float = 0.05, n_init: int = 32, pyr=None, level: int = 0) -> float:
    """Best k x k filter (optionally at pyramid level `level`): max over w of the
    per-slice value.  Ascent in R^{k^2} -- a tiny problem, unlike Max-SW in R^d."""
    size = s1.shape[-1]
    w = torch.randn(n_init, 1, k, k, device=s1.device, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        h = pyr.composite(w, level) if pyr is not None and level > 0 else w
        h = h / h.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        u = envelopes_from_filters(h, size)
        loss = -population_slice_wpp(u, s1, s2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = pyr.composite(w, level) if pyr is not None and level > 0 else w
        h = h / h.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        return float(population_slice_wpp(envelopes_from_filters(h, size), s1, s2).max().sqrt())
