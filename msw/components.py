"""The components of the two-sample statistic, and their algebra under the group.

One randomization group for the WHOLE portfolio -- the pair group of
:mod:`msw.pairgroup` -- and one pass over the data that produces every component
under every group element.

Transport rows (the W_2 part)
-----------------------------
For equal block sizes and p = 2 the four-term floor-corrected row of the v0.1
estimator is an inner product of sorted pooled samples,

    1/2[W(A1,B1) + W(A2,B2)] - 1/2[W(A1,A2) + W(B1,B2)]
        = <s(B1) - s(A2), s(B2) - s(A1)> / K ,

i.e. a rank-one cross-fit of ||mu||^2 with mu = E s(B) - E s(A).  Writing
``D_i = s(b_i) - s(a_i)`` and ``Psi_ij = <D_i, D_j> / K``, the shipped row is the
sum over L/2 DISJOINT pairs; since 0.2.0 the estimator is the U-statistic over
all L(L-1)/2 pairs, which has the same expectation and a smaller variance.  Both
p = 2 and equal block sizes are required, and both are enforced.

Under the group, ``Psi_ij -> sigma_i sigma_j Psi_ij``: the whole orbit of the
transport statistic is sign algebra on one Gram matrix.

Signed rows (the first-order part)
----------------------------------
Any per-block or per-image summary ``phi`` contributes a row
``m_i = phi(b_i) - phi(a_i)``, which the group maps to ``-m_i``.  Block tail and
shape functionals are read straight off the sorted samples the Gram matrix
already needs, so they cost nothing beyond the sort.

Standardization and weights
---------------------------
* Per-component floor: ``sd_k = max(sd_k, 1e-6 (max_e A_k - min_e A_k))`` -- each
  component is floored by ITS OWN orbit range.  A floor taken from a median
  across components is meaningless here, because the components live on
  different scales.
* Weights are orbit-INVARIANT precisions: either the pooled variance of the
  union of the 2L block values, or the exact orbit second moment of the row,
  ``q_s^2 = sum_{i != j} Psi_ijs^2 / (L(L-1))^2``.  Being constants of the orbit
  they leave the test exact, and being measured on each level's own responses
  they carry the per-level effective sample size automatically.
"""

from __future__ import annotations

import math

import torch

from .features import SliceFeatures

__all__ = ["block_rows", "per_image_u", "image_signed_rows", "pixel_u",
           "transport_components", "transport_z", "signed_z", "zcomps",
           "col_p", "p_maxz", "p_cct", "p_minp_wy",
           "BLOCK_SUMS", "IMG_SUMS", "PX_SUMS"]

DT = torch.float64
REL = 1e-6           # relative floor, used for every standard error and sd


def to_double(f: SliceFeatures) -> SliceFeatures:
    """Promote slice features to float64 in place (sorts and Gram matrices)."""
    f.mats = [m.to(DT) for m in f.mats]
    return f


def _floor(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.clamp(REL * scale, min=1e-300))


def finite(A: torch.Tensor) -> torch.Tensor:
    """Map non-finite component values to 0 -- i.e. to "no evidence".

    A degenerate column (identical inputs, a constant channel, a zero-variance
    slice) has every orbit entry equal to zero, so its precision weight 1/se^2
    overflows and the aggregate comes back NaN.  A NaN must not be read as a
    large statistic: with `>=` comparisons it would score zero exceedances and
    hand back the orbit minimum, i.e. a spurious rejection.  Mapping it to 0
    makes the column tie with itself across the whole orbit, which yields
    p = 1 for that column and contributes nothing to the max.
    """
    return torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================ block plumbing
def _shape_stats(s: torch.Tensor) -> dict:
    """Block-level functionals of SORTED samples ``s`` (L, K, S) -> name -> (L, S)."""
    K = s.shape[1]
    q = lambda a: s[:, min(K - 1, max(0, int(round(a * (K - 1))))), :]
    e2 = s.pow(2).mean(1)
    e4 = s.pow(4).mean(1)
    rng995 = (q(0.99875) - q(0.00125)).clamp_min(1e-300)
    iqr = (q(0.75) - q(0.25)).clamp_min(1e-300)
    return {
        "benergy": e2,
        "babs": s.abs().mean(1),
        "blogpow": e2.clamp_min(1e-300).log(),
        "bkurt": (e4 / e2.pow(2).clamp_min(1e-300)).clamp_min(1e-300).log(),
        "bsh": rng995.log() - iqr.log(),
        "bloc": s.mean(1),
    }


BLOCK_SUMS = ("benergy", "babs", "blogpow", "bkurt", "bsh", "bloc")


@torch.no_grad()
def block_rows(fa: SliceFeatures, fb: SliceFeatures, L: int, M: int,
               chunk: int | None = None, want_block_sums: bool = True):
    """One pass over the sorted pooled block samples.

    Returns ``(psi (L, L, S), bs {name: (L, S)}, bse {name: (S,)})``: the
    transport Gram matrix, the signed block rows ``phi(b_i) - phi(a_i)``, and
    their standard errors.

    ``bse`` comes from the variance of the 2L pooled block values
    ``{phi(a_i)} u {phi(b_i)}``.  The pair swap only permutes that multiset, so
    the standard error is exactly constant on the orbit and the test stays exact.

    The identity behind ``psi`` holds for p = 2 and EQUAL block sizes only, so
    there is no ``p`` argument, and both datasets must present the same layout:
    the same number of pooled samples per image in every slice family, and at
    least ``L * M`` images each.
    """
    if list(fa.per_image) != list(fb.per_image):
        raise ValueError("the two datasets must have identical slice layouts "
                         f"({fa.per_image} vs {fb.per_image}); equal block sizes "
                         "are what makes the transport rows exact")
    for m, q in list(zip(fa.mats, fa.per_image)) + list(zip(fb.mats, fb.per_image)):
        if m.shape[1] < L * M * q:
            raise ValueError(f"need {L * M} images per dataset, got {m.shape[1] // q}")
    psis = []
    acc_a = {k: [] for k in BLOCK_SUMS}
    acc_b = {k: [] for k in BLOCK_SUMS}
    for (ma, q), mb in zip(zip(fa.mats, fa.per_image), fb.mats):
        K = M * q
        S = ma.shape[0]
        # keep the (L, K, chunk) sorted temporaries under ~200 MB per tensor
        ck = chunk if chunk is not None else max(1, min(32, int(2.5e7 // max(L * K, 1))))
        g = torch.empty(L, L, S, device=ma.device, dtype=DT)
        pa = {k: torch.empty(L, S, device=ma.device, dtype=DT) for k in BLOCK_SUMS}
        pb = {k: torch.empty(L, S, device=ma.device, dtype=DT) for k in BLOCK_SUMS}
        for s0 in range(0, S, ck):
            s1 = min(s0 + ck, S)
            a = ma[s0:s1, :L * K].t().contiguous().view(L, K, s1 - s0).sort(1).values
            b = mb[s0:s1, :L * K].t().contiguous().view(L, K, s1 - s0).sort(1).values
            if want_block_sums:
                for k, v in _shape_stats(a).items():
                    pa[k][:, s0:s1] = v
                for k, v in _shape_stats(b).items():
                    pb[k][:, s0:s1] = v
            d = (b - a).permute(2, 0, 1).contiguous()
            del a, b
            g[:, :, s0:s1] = (torch.bmm(d, d.transpose(1, 2)) / K).permute(1, 2, 0)
            del d
        psis.append(g)
        for k in BLOCK_SUMS:
            acc_a[k].append(pa[k])
            acc_b[k].append(pb[k])
    psi = torch.cat(psis, 2)
    bs, bse = {}, {}
    for k in BLOCK_SUMS:
        A = torch.cat(acc_a[k], 1)
        B = torch.cat(acc_b[k], 1)
        bs[k] = B - A
        pool = torch.cat([A, B], 0)
        v = pool.var(0, unbiased=True)
        se = (2.0 * v).clamp_min(0).sqrt()
        bse[k] = torch.maximum(se, REL * pool.abs().mean(0).clamp_min(1e-300))
    return psi, bs, bse


IMG_SUMS = ("ienergy", "iabs", "ilogpow", "ikurt", "imin", "imax")


@torch.no_grad()
def per_image_u(f: SliceFeatures, sums=IMG_SUMS, chunk: int | None = None) -> dict:
    """name -> (S, N) per-image slice summaries, each a function of ONE image.

    Chunked over slices: the level-0 view is (S, N, q) with q up to 3600, which
    at 64 px would materialise ~1 GB per temporary in float64.
    """
    n = f.n_images
    acc = {s: [] for s in sums}
    for m, q in zip(f.mats, f.per_image):
        S = m.shape[0]
        ck = chunk if chunk is not None else max(1, min(16, int(2.5e7 // max(n * q, 1))))
        for s0 in range(0, S, ck):
            y = m[s0:s0 + ck].view(-1, n, q)
            e2 = y.pow(2).mean(2)
            if "ienergy" in acc:
                acc["ienergy"].append(e2)
            if "iabs" in acc:
                acc["iabs"].append(y.abs().mean(2))
            if "ilogpow" in acc:
                acc["ilogpow"].append(e2.clamp_min(1e-300).log())
            if "ikurt" in acc:
                e4 = y.pow(4).mean(2)
                acc["ikurt"].append((e4 / e2.pow(2).clamp_min(1e-300)).clamp_min(1e-300).log())
            if "imin" in acc:
                acc["imin"].append(y.min(2).values)
            if "imax" in acc:
                acc["imax"].append(y.max(2).values)
    return {k: torch.cat(v, 0) for k, v in acc.items()}


def image_signed_rows(ua: torch.Tensor, ub: torch.Tensor, L: int, M: int):
    """(L, S) rows ``m_i = mean_{b_i} u - mean_{a_i} u`` and the pooled se (S,).

    ``se_s = sqrt(2 var_pool(u_s) / M)`` with the variance taken over the union
    of the 2LM per-image values -- a multiset the pair swap does not move, so
    the standard error is a constant of the orbit, estimated from 2LM degrees of
    freedom instead of from L block spreads.
    """
    A = ua[:, :L * M].reshape(-1, L, M).mean(2)
    B = ub[:, :L * M].reshape(-1, L, M).mean(2)
    m = (B - A).t().contiguous()
    pool = torch.cat([ua[:, :L * M], ub[:, :L * M]], 1)
    v = pool.var(1, unbiased=True)
    se = (v * (2.0 / max(M, 1))).clamp_min(0).sqrt()
    se = torch.maximum(se, REL * pool.abs().mean(1).clamp_min(1e-300))
    return m, se


PX_SUMS = ("pmin", "pmax", "pmean", "plogvar", "pskew", "pkurt")


@torch.no_grad()
def pixel_u(x: torch.Tensor) -> torch.Tensor:
    """(6*(C+1), N) per-image RAW-PIXEL summaries.

    min, max, mean, log variance, skewness and kurtosis of the greyscale image
    and of each colour channel -- the Portilla-Simoncelli ``pixel_statistics``
    block.  Six numbers per image per channel: by three orders of magnitude the
    cheapest component in the portfolio, and the one that carries essentially
    all of the smooth-shading sensitivity (per-image pixel min and max).
    """
    n = x.shape[0]
    chans = [x.mean(1)] + ([x[:, c] for c in range(x.shape[1])] if x.shape[1] > 1 else [])
    out = []
    for y in chans:
        y = y.reshape(n, -1).to(DT)
        m = y.mean(1)
        d = y - m.unsqueeze(1)
        v = d.pow(2).mean(1).clamp_min(1e-300)
        out += [y.min(1).values, y.max(1).values, m, v.log(),
                d.pow(3).mean(1) / v.pow(1.5), d.pow(4).mean(1) / v.pow(2)]
    return torch.stack(out, 0)


# ================================================================ components
def transport_components(psi: torch.Tensor, pats: torch.Tensor, level_sizes: list[int],
                         iv_cols=None, chunk: int = 1024) -> list[torch.Tensor]:
    """``[iv, ml, ivp]`` of the U-statistic transport rows

        ``rows_i(sigma) = sigma_i sum_{j != i} sigma_j Psi_ij / (L - 1)``.

    iv   inverse-variance across slices, se from the L rows OF THAT PATTERN
         (the v0.1 rule, transplanted to the pair group);
    ml   max over pyramid levels of the level-mean z (the v0.1 max-level rule);
    ivp  inverse-variance with the orbit-invariant precision ``w_s = 1/q_s^2``,
         ``q_s^2 = E_sigma[(mean_i rows_i)^2] = sum_{i != j} Psi_ijs^2/(L(L-1))^2``
         -- the exact orbit second moment, built from L(L-1) terms instead of
         from L block spreads.
    """
    L = psi.shape[0]
    i = torch.arange(L, device=psi.device)
    p0 = psi.clone()
    p0[i, i, :] = 0.0
    q = (p0.pow(2).sum((0, 1))).sqrt() / (L * (L - 1.0))
    q = _floor(q, q.mean().expand_as(q))
    ivs, mls, ivps = [], [], []
    for s0 in range(0, pats.shape[0], chunk):
        pp = pats[s0:s0 + chunk]
        v = torch.einsum("pj,ijs->pis", pp, p0)
        rows = pp.unsqueeze(-1) * v / (L - 1)
        r = rows if iv_cols is None else rows[:, :, iv_cols]
        tb = r.mean(1)
        se = _floor(r.std(1) / L ** 0.5, r.pow(2).mean(1).sqrt())
        w = 1.0 / se ** 2
        ivs.append((w * tb).sum(1) / w.sum(1))
        qq = q if iv_cols is None else q[iv_cols]
        wp = 1.0 / qq ** 2
        ivps.append((wp * tb).sum(1) / wp.sum())
        mu, ses, j = [], [], 0
        for c in level_sizes:
            tl = rows[:, :, j:j + c].mean(2)
            mu.append(tl.mean(1))
            ses.append(_floor(tl.std(1) / L ** 0.5, tl.pow(2).mean(1).sqrt()))
            j += c
        mls.append((torch.stack(mu, 1) / torch.stack(ses, 1)).max(1).values)
        del rows, v, r
    return [torch.cat(ivs), torch.cat(mls), torch.cat(ivps)]


def transport_z(psi: torch.Tensor, pats: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """(P, S) per-slice orbit-standardised z of the transport row mean."""
    L = psi.shape[0]
    i = torch.arange(L, device=psi.device)
    p0 = psi.clone()
    p0[i, i, :] = 0.0
    q = (p0.pow(2).sum((0, 1))).sqrt() / (L * (L - 1.0))
    q = _floor(q, q.mean().expand_as(q))
    out = []
    for s0 in range(0, pats.shape[0], chunk):
        pp = pats[s0:s0 + chunk]
        v = torch.einsum("pj,ijs->pis", pp, p0)
        out.append((pp.unsqueeze(-1) * v / (L - 1)).mean(1) / q)
    return torch.cat(out)


def signed_z(m: torch.Tensor, se: torch.Tensor, pats: torch.Tensor,
             chunk: int = 4096) -> torch.Tensor:
    """(P, S) per-slice z of the signed rows under every group element."""
    L = m.shape[0]
    out = []
    for s0 in range(0, pats.shape[0], chunk):
        out.append(((pats[s0:s0 + chunk] @ m) / L) / se * (L ** 0.5))
    return torch.cat(out)


def zcomps(z: torch.Tensor, w: torch.Tensor | None = None) -> list[torch.Tensor]:
    """``[|coherent mean|, max_s |z_s|, mean_s z_s^2]`` from a (P, S) z-table.

    ``z`` is already precision-weighted (unit variance per slice under the
    orbit), so the uniform mean of z IS the ``1/se^2``-weighted mean of the rows.
    ``chi`` is the quadratic companion, and the only one of the three that
    survives when the effect has no common SIGN across slices (the +-45 degree
    chirality row, where the coherent aggregation is provably blind).
    """
    ww = torch.ones(z.shape[1], device=z.device, dtype=z.dtype) if w is None else w
    iv = ((z * ww).sum(1) / ww.sum()).abs()
    mx = z.abs().max(1).values
    chi = (z ** 2).mean(1)
    return [iv, mx, chi]


# ================================================================= combiners
def std_floor(A: torch.Tensor) -> torch.Tensor:
    """Per-component standardization floor: ``max(sd_k, 1e-6 * orbit range_k)``.

    Each component is floored by its OWN orbit range.  The components of the
    portfolio live on wildly different scales, so a floor derived from a median
    across components either does nothing or swamps whole families.
    """
    sd = A.std(0)
    return torch.maximum(sd, torch.clamp(REL * (A.max(0).values - A.min(0).values),
                                         min=1e-300))


def col_p(A: torch.Tensor) -> list[float]:
    """Per-component orbit (rank) p-values; row 0 of ``A`` is the identity."""
    A = finite(A)
    P = A.shape[0]
    return [(1.0 + float((A[1:, i] >= A[0, i]).sum())) / P for i in range(A.shape[1])]


def p_maxz(A: torch.Tensor) -> float:
    """max-z combiner: standardize each column on the orbit, take the max.

    The shipped combiner.  Measured at least as powerful as the Cauchy
    combination and as Westfall-Young everywhere, and the only one of the three
    that keeps its size near nominal at N = 8, where the orbit has 256 elements
    and rank-based combiners run out of resolution.
    """
    A = finite(A)
    Z = finite((A - A.mean(0)) / std_floor(A))
    T = Z.max(1).values
    return (1.0 + float((T[1:] >= T[0]).sum())) / A.shape[0]


def p_cct(pc: list[float]) -> float:
    """Cauchy combination of per-component orbit p-values (Liu & Xie, 2020)."""
    c = sum(math.tan((0.5 - min(max(q, 1e-12), 1 - 1e-12)) * math.pi) for q in pc) / len(pc)
    return 0.5 - math.atan(c) / math.pi


def p_minp_wy(A: torch.Tensor) -> float:
    """Westfall-Young min-p: the minimum column p, calibrated on the orbit itself.

    Exact and purely rank-based, hence immune to any standardization bug -- which
    is what makes it a useful cross-check on ``p_maxz`` rather than a rival.
    """
    A = finite(A)
    P = A.shape[0]
    asc = A.sort(0).values
    lo = torch.searchsorted(asc.t().contiguous(), A.t().contiguous(), right=False).t()
    pcol = (P - lo).double() / P
    T = -pcol.min(1).values
    return (1.0 + float((T[1:] >= T[0]).sum())) / P
