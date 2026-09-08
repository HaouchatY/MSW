"""The randomization group of the MSW test: the pair group (Z2)^L.

The two datasets are cut into ``L`` matched blocks of ``M`` images each,
``(a_1, b_1), ..., (a_L, b_L)``.  Under H0 the 2L blocks are i.i.d., so each
generator ``pi_i = (a_i <-> b_i)`` -- swapping the i-th matched pair between the
two datasets -- is measure preserving, and the generators commute: the group is
``(Z2)^L``, represented here by sign vectors ``sigma in {+-1}^L``.

Every component of the statistic is a function of the labelled data on which the
group acts by pure sign algebra (transport rows ``Psi_ij -> sigma_i sigma_j
Psi_ij``, signed rows ``m_i -> -m_i``), so the whole orbit is obtained without
recomputing a single feature, sort or convolution.  ``msw/tests`` contains the
physical-relabelling audit that certifies this algebra against recomputation
from scratch.

Which elements to use
---------------------
``full_patterns``     all ``2^L`` elements; affordable up to L ~ 12-13.
``parity_subgroup``   an honest SUBGROUP of order ``2^d`` (the coordinates split
                      into parity blocks) -- the right object when ``2^L`` is too
                      large, because a subgroup still acts transitively on its
                      own orbit and the rank test stays EXACT.  A fixed random
                      SUBSET of the group is not closed under products and does
                      not give an exact test; that is why one is never used here.
``mc_patterns``       identity + i.i.d. uniform elements, valid only if redrawn
                      per dataset (the seed must depend on the draw).  Kept for
                      reference; not the default.
"""

from __future__ import annotations

import torch

from .core import get_device

__all__ = ["block_plan", "full_patterns", "parity_subgroup", "mc_patterns",
           "patterns_for"]

DT = torch.float64


def block_plan(n_a: int, n_b: int, L: int = 16) -> tuple[int, int]:
    """(L, M): number of matched blocks and images per block.

    ``L`` blocks of ``M = n // L`` images are cut from each dataset with
    ``n = min(n_a, n_b)``; ``L`` drops to ``n`` when the datasets are smaller
    than the requested number of blocks (one image per block).  Blocks must have
    EQUAL sizes -- that is what makes the four-term transport identity exact --
    so the last ``n mod L`` images of each dataset are not used.  At the default
    L = 16 that is at most 15 images; if N is small and awkward (say 25), a
    smaller ``L`` uses more of the data.
    """
    n = min(n_a, n_b)
    L = min(L, n)
    if L < 2:
        raise ValueError("need at least 2 images per dataset")
    return L, n // L


def full_patterns(n: int, device=None, dtype=DT) -> torch.Tensor:
    """All 2^n sign vectors, identity first.  Shape (2^n, n)."""
    device = get_device() if device is None else device
    idx = torch.arange(2 ** n, device=device)
    return (((idx.view(-1, 1) >> torch.arange(n, device=device)) & 1) * -2 + 1).to(dtype)


def parity_subgroup(n: int, n_pat: int, device=None, dtype=DT,
                    interleave: bool = True) -> torch.Tensor:
    """A subgroup of ``{+-1}^n`` of order ``2^d``, ``2^d <= n_pat``.

    The coordinates are split into ``nb = n - d`` blocks and each block is
    constrained to an even number of flips: the constraint is linear over GF(2),
    so the set is closed under products -- a genuine subgroup, hence an exact
    test.  The identity is row 0.

    ``interleave`` reorders the coordinates so that consecutive ones live in
    DIFFERENT parity blocks.  This is not cosmetic: with the contiguous layout a
    pair-aligned row set (one that only ever sees products
    ``sigma_{2j} sigma_{2j+1}``) has its sign vector pinned by the block
    parities and can never reject, which showed up as an exactly-zero rejection
    rate in the small-N audit.
    """
    device = get_device() if device is None else device
    if 2 ** n <= n_pat:
        return full_patterns(n, device, dtype)
    d = max(1, int(n_pat).bit_length() - 1)
    d = max(d, n - n // 2)
    nb = n - d
    sizes = [n // nb + (1 if i < n % nb else 0) for i in range(nb)]
    free = torch.arange(2 ** d, device=device)
    cols, f, pos, blk = [None] * n, 0, 0, [0] * n
    for bi, m in enumerate(sizes):
        bits = [(free >> (f + i)) & 1 for i in range(m - 1)]
        f += m - 1
        par = bits[0].clone()
        for bb in bits[1:]:
            par = par ^ bb
        for j, bb in enumerate(bits + [par]):
            cols[pos + j] = bb
            blk[pos + j] = bi
        pos += m
    P = (torch.stack(cols, 1) * -2 + 1).to(dtype)
    if interleave:
        buckets = [[i for i in range(n) if blk[i] == b] for b in range(nb)]
        order, r = [], 0
        while len(order) < n:
            for b in range(nb):
                if r < len(buckets[b]):
                    order.append(buckets[b][r])
            r += 1
        P = P[:, torch.tensor(order, device=device)]
    return P


def mc_patterns(n: int, n_pat: int, seed: int, device=None, dtype=DT) -> torch.Tensor:
    """Identity + (n_pat - 1) i.i.d. uniform group elements.

    Exact only when REDRAWN PER DATASET, i.e. when ``seed`` depends on the draw;
    a fixed random subset re-used across draws is not a subgroup and inflates
    the size of the test.
    """
    device = get_device() if device is None else device
    g = torch.Generator(device=device).manual_seed(int(seed))
    b = torch.randint(0, 2, (n_pat, n), device=device, generator=g)
    b[0] = 0
    return (b * -2 + 1).to(dtype)


def patterns_for(L: int, n_pat: int = 4096, mode: str = "subgroup", seed: int = 0,
                 device=None) -> torch.Tensor:
    """The group elements the test enumerates: (P, L) of +-1 with identity first.

    Full enumeration whenever ``2^L <= n_pat``; otherwise the interleaved parity
    subgroup ("subgroup", the default), the contiguous one ("contig", for the
    audit that shows why interleaving matters), or Monte-Carlo elements ("mc").
    """
    if 2 ** L <= n_pat:
        return full_patterns(L, device)
    if mode == "mc":
        return mc_patterns(L, n_pat, seed, device)
    return parity_subgroup(L, n_pat, device, interleave=(mode != "contig"))
