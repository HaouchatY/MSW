# Changes

## 0.2.0

The test keeps its name, its entry point and its meaning; what changed is how
much of the data it uses, how it randomizes, and what it looks at.

**The estimator now averages all pairs of image blocks.**  Both datasets are cut
into `L` matched blocks (16 by default).  For `p = 2` and equal block sizes the
old four-term floor-corrected row is exactly an inner product of sorted pooled
samples, `<s(b_i) - s(a_i), s(b_j) - s(a_j)> / K`, so the row that used to sum
over `L/2` disjoint pairs is now the U-statistic over all `L(L-1)/2` pairs.  Same
expectation, so distances mean exactly what they meant before -- about half the
standard deviation in a like-for-like comparison, and a correspondingly tighter
interval on `msw.distance`, whose confidence interval is now a delete-one-block
jackknife.

**The randomization group is the pair group.**  Under H0 the `2L` blocks are
i.i.d., so swapping the two halves of any matched pair is measure preserving:
the group is `(Z2)^L`, with one sign per matched pair rather than one per
disjoint image group.  Every component transforms under it by pure sign algebra,
so the whole orbit costs no extra features, sorts or convolutions.  What this
buys: at 8 images per side the group has 256 elements, so the smallest
attainable p-value is 1/256 and rejection at alpha = 0.05 is possible.  Before,
8 images gave 4 groups, 16 patterns, and a smallest attainable p-value of 1/16 =
0.0625 -- rejecting at the 5% level was arithmetically impossible, whatever the
data.  When the full group is too large to enumerate, the test uses a parity
SUBGROUP of 4096 elements (interleaved across the coordinates, which matters:
with a contiguous layout a whole class of components is pinned by the block
parities and can never reject).  A random subset of the group is never used --
it is not closed under products and would not give an exact test.

**First-order components alongside the transport rows.**  The shipped portfolio
is now `CORE` = transport (T) + block tail/shape (TL: log-kurtosis, and
log-range minus log-IQR) + raw-pixel statistics (PX: per-image min, max, mean,
log variance, skew, kurtosis, greyscale and per channel) + product slices (J:
per-image normalised means of `r(x) r(x+d)`, `r_i^2` and `|r_i| |r_j|`), on both
the per-channel DCT family and the opponent-chroma family.  The tail and shape
functionals are read straight off the sorted samples the transport Gram matrix
already needs, so they cost nothing beyond the sort; the product slices close the
+-45 degree orientation blind spot that no marginal slice of any bank can see;
the pixel statistics are the cheapest components in the portfolio by three orders
of magnitude and are what carries smooth-shading sensitivity.  Measured across
the paper's nine settings at N = 500, `CORE` certifies detection at a smaller
perturbation than the 0.1 statistic on eight of the nine and at the same rung on
the ninth -- between one and four rungs of the grid, the largest gains on the
marginal-shape and smooth-shading settings.

**Orbit-invariant weights.**  Precisions are now computed either from the pooled
union of the `2L` block values or from the exact orbit second moment of the row,
`q_s^2 = sum_{i != j} Psi_ijs^2 / (L(L-1))^2`.  Both are constants of the orbit,
so the test stays exact, and both are measured on each level's own responses, so
they carry the per-level effective sample size automatically.

**Per-component standardization floor.**  Each component is floored by its own
orbit range, `sd_k = max(sd_k, 1e-6 (max_e A_k - min_e A_k))`, instead of by a
median across components.  The components live on very different scales, so a
shared floor either does nothing or swamps whole families.

**Three combiners, max-z shipped.**  `msw.test` reports `p` (max-z), `p_cct`
(Cauchy combination) and `p_wy` (Westfall-Young min-p) from the same orbit.
Max-z measures at least as powerful as the other two everywhere and is the only
one that keeps its size near nominal at 8 images per side, where rank-based
combiners run out of resolution.

**The filter bank is deterministic.**  When a bank is subsampled the permutation
now comes from an explicit, seeded generator instead of the global RNG, so the
bank no longer depends on unrelated draws elsewhere in the program, and the
default draw is made on the CPU so that CPU and GPU select the same filters.
The default for `msw.test` is the whole DCT bank, which involves no draw at all.

### API

* `msw.test(a, b)` keeps `.T`, `.p`, `.components` and `.diagnose()`.  New:
  `.component_p` (exact p-value per component), `.p_wy`, `.L`, `.M`,
  `.group_size`, `.portfolio`.  `.G` is kept as an alias of `.L`.
* `msw.test(..., groups=)` is accepted as a deprecated alias of `L`.
  `splitting=` is accepted and ignored: blocks are always equal-sized now, which
  is what makes the transport identity exact.
* `.n_used` is now `L * M`; an N that is not a multiple of `L` leaves `N mod L`
  images unused (at most 15 at the default `L = 16`).  Lower `L` at an awkward N.
* `.per_level` is keyed by slice family ("per-channel DCT", "opponent
  luminance", "opponent chroma") and holds the per-level transport z.
* `msw.distance(a, b)` returns `(d, lo, hi)` as before; `groups=` is a deprecated
  alias of `L`.
* New public names: `msw.Portfolio`, `msw.JointSW`, `msw.portfolios`,
  `msw.core_columns`, `msw.evaluate`, `msw.block_plan`, `msw.patterns_for`,
  `msw.components`.
* New modules: `msw/pairgroup.py`, `msw/components.py`, `msw/portfolio.py`,
  `msw/joint.py`.  Nothing was removed.
