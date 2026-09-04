"""A common representation for every sliced estimator: the *slice features*.

Each estimator maps a dataset of N images to one or more matrices ``(S_i, M_i)``
whose columns are the 1-D samples of ``S_i`` slices, laid out image-major with a
fixed number ``per_image[i]`` of columns per image:

    classic SW      one matrix (n_proj, N)              per_image = 1
    CSW             one matrix (n_slicers*L, N)         per_image = 1
    pooled conv     one matrix per pyramid level,       per_image = #kept positions
                    (F_l, N * m_l)                                    at that level

Everything downstream -- the 1-D Wasserstein, the floor correction, the blocked
variance estimate, the aggregation -- then works uniformly, so comparisons
between estimators differ only in the slices themselves and not in the
statistical machinery around them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .core import w1d_sort

__all__ = ["SliceFeatures", "blocked_values", "balanced_blocked_values",
           "paired_values", "concat_features"]


def concat_features(*fs: "SliceFeatures") -> "SliceFeatures":
    """Pool several slice families into one, so the aggregation can choose between
    them.  Families may have different numbers of samples per image."""
    n = fs[0].n_images
    assert all(f.n_images == n for f in fs)
    return SliceFeatures([m for f in fs for m in f.mats],
                         [q for f in fs for q in f.per_image], n)


@dataclass
class SliceFeatures:
    mats: list[torch.Tensor]   # each (S_i, N * per_image[i])
    per_image: list[int]
    n_images: int

    @property
    def n_slices(self) -> int:
        return sum(m.shape[0] for m in self.mats)

    def subset(self, i0: int, i1: int) -> list[torch.Tensor]:
        """Columns belonging to images [i0, i1)."""
        return [m[:, i0 * q : i1 * q] for m, q in zip(self.mats, self.per_image)]


def paired_values(fa: SliceFeatures, fb: SliceFeatures, p: int = 2,
                  debias: bool = True) -> torch.Tensor:
    """Per-slice W_p^p (floor-corrected by default) using all images."""
    if not debias:
        return torch.cat([w1d_sort(x.t(), y.t(), p=p) for x, y in zip(fa.mats, fb.mats)])
    return blocked_values(fa, fb, groups=1, p=p)[0]


def blocked_values(fa: SliceFeatures, fb: SliceFeatures, groups: int = 4,
                   p: int = 2, presort: bool = True) -> torch.Tensor:
    """`groups` independent floor-corrected per-slice estimates, shape (G, S).

    Group j uses the disjoint image blocks (A_{2j}, A_{2j+1}, B_{2j}, B_{2j+1}) and
    forms

        t^{(j)} = 1/2 [W(A_2j, B_2j) + W(A_2j+1, B_2j+1)]
                - 1/2 [W(A_2j, A_2j+1) + W(B_2j, B_2j+1)] ,

    so E[t^{(j)}] = 0 under H0 and ~ the population W_p^p under H1.  The mean over
    j is the estimate and the spread over j gives a per-slice standard error --
    the ingredient the aggregation in `msw.testing` needs.  Splitting by *image*
    (never by pooled pixel) is essential: pixels inside one image are strongly
    dependent, and a pixel-level split would badly under-estimate the noise.
    """
    na, nb = fa.n_images // (2 * groups), fb.n_images // (2 * groups)
    if na < 1 or nb < 1:
        raise ValueError(f"need at least {2*groups} images per dataset")
    out = []
    for j in range(groups):
        a1 = fa.subset(2 * j * na, (2 * j + 1) * na)
        a2 = fa.subset((2 * j + 1) * na, (2 * j + 2) * na)
        b1 = fb.subset(2 * j * nb, (2 * j + 1) * nb)
        b2 = fb.subset((2 * j + 1) * nb, (2 * j + 2) * nb)
        out.append(_four_term(a1, a2, b1, b2, p=p, presort=presort))
    return torch.stack(out)


def _four_term(a1, a2, b1, b2, p: int = 2, presort: bool = True) -> torch.Tensor:
    """The per-slice four-term row for one group of blocks.

    When all four blocks have the same sample count, each block is sorted ONCE
    and the four pairings become subtractions of sorted tensors -- bit-identical
    to calling `w1d_sort` per pairing (which sorts every operand twice), at
    roughly half the sort cost.  Unequal counts fall back to `w1d_sort`, whose
    shared-quantile-grid path handles them.
    """
    vals = []
    for x1, x2, y1, y2 in zip(a1, a2, b1, b2):
        if presort and x1.shape[1] == x2.shape[1] == y1.shape[1] == y2.shape[1]:
            s1, s2 = x1.t().sort(dim=0).values, x2.t().sort(dim=0).values
            t1, t2 = y1.t().sort(dim=0).values, y2.t().sort(dim=0).values
            cross = 0.5 * ((s1 - t1).abs().pow(p).mean(0) + (s2 - t2).abs().pow(p).mean(0))
            within = 0.5 * ((s1 - s2).abs().pow(p).mean(0) + (t1 - t2).abs().pow(p).mean(0))
        else:
            cross = 0.5 * (w1d_sort(x1.t(), y1.t(), p=p) + w1d_sort(x2.t(), y2.t(), p=p))
            within = 0.5 * (w1d_sort(x1.t(), x2.t(), p=p) + w1d_sort(y1.t(), y2.t(), p=p))
        vals.append(cross - within)
    return torch.cat(vals)


def balanced_group_sizes(n_a: int, n_b: int, groups: int | None = None) -> list[int]:
    """Group block sizes that use every available image pair.

    P = min(n_a, n_b) // 2 image pairs are split over G = min(groups or 8, P)
    groups of near-equal size (they differ by at most one).  All four blocks of
    group j share the size m_j, which is what makes E[t^(j)] = 0 exact; using
    the smaller dataset's P for both sides keeps that exactness at unequal N.
    """
    P = min(n_a, n_b) // 2
    if P < 2:
        raise ValueError("need at least 4 images per dataset")
    G = max(2, min(groups if groups is not None else 8, P))
    base, extra = divmod(P, G)
    return [base + 1 if j < extra else base for j in range(G)]


def balanced_blocked_values(fa: SliceFeatures, fb: SliceFeatures,
                            groups: int | None = None, p: int = 2) -> torch.Tensor:
    """`blocked_values` with balanced splitting: uses ALL floor(N/2) image
    pairs at any N (the fixed-size splitting consumes only 2G*floor(N/2G))
    and adapts the number of groups at tiny N instead of special-casing it."""
    sizes = balanced_group_sizes(fa.n_images, fb.n_images, groups)
    out, o = [], 0
    for m in sizes:
        a1 = fa.subset(2 * o, 2 * o + m)
        a2 = fa.subset(2 * o + m, 2 * o + 2 * m)
        b1 = fb.subset(2 * o, 2 * o + m)
        b2 = fb.subset(2 * o + m, 2 * o + 2 * m)
        out.append(_four_term(a1, a2, b1, b2, p=p))
        o += m
    return torch.stack(out)
