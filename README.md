# MSW

Multi-scale sliced Wasserstein tests between two sets of images, built for
stationary (texture-like) image distributions. Exact finite-sample p-values,
no trained features, certifies differences from as few as 8 images.

## Install

```bash
pip install git+https://github.com/HaouchatY/MSW.git
```

Requires `torch` and `numpy`. Runs on GPU when available, CPU otherwise.

## Quickstart

```python
import torch, msw

A = torch.rand(64, 3, 32, 32)   # two sets of images
B = torch.rand(64, 3, 32, 32)

r = msw.test(A, B)
print(r.T, r.p)                 # statistic, exact p-value
print(r.diagnose())             # e.g. "difference concentrated at level 0, opponent chroma"

d, lo, hi = msw.distance(A, B)  # distance value with 95% CI
```

See `examples/quickstart.py` for a demo on Gaussian random fields with
closed-form ground truth.

## Notes

- Designed for stationary data (textures, materials, scientific imaging,
  generator monitoring). `msw.test` warns when the input looks strongly
  non-stationary (e.g. aligned faces).
- Blind by construction to constant offsets/color casts, alignment, and
  instance identity; semantic differences need feature-based metrics.
- `msw.SequentialCertifier` (experimental): anytime-valid sequential testing —
  keep adding batches, stop as soon as `certified`.
- ~0.2 s per test at N=500 (V100), exact p included.

Reference: Y. Haouchat, *A Multi-Scale Sliced Wasserstein Distance Between
Stationary Image Distributions*, under review.

## License

MIT
