# MSW — Multi-Scale Sliced Wasserstein for stationary image distributions

`msw` measures the distance between two *sets of images* and tests whether they
come from the same distribution, using multi-scale pooled-patch sliced
Wasserstein statistics. It is built for **stationary (texture-like) image
distributions** — textures, materials, scientific imaging, sensor validation,
generator checkpoint monitoring — where it certifies differences from as few
as 8 images, with exact finite-sample calibration and no trained features.

Reference: Y. Haouchat, *A Multi-Scale Sliced Wasserstein Distance Between
Stationary Image Distributions*, under review.

## Install

```bash
pip install git+https://github.com/HaouchatY/MSW.git
```

Dependencies: `torch>=2.0`, `numpy`. GPU used automatically when available
(CPU works).

## Quickstart

```python
import torch, msw

A = torch.rand(64, 3, 32, 32)   # two sets of images, any device
B = torch.rand(64, 3, 32, 32)

r = msw.test(A, B)
print(r.T, r.p)                 # portfolio statistic, EXACT sign-flip p-value
print(r.diagnose())             # e.g. "difference concentrated at level 0, opponent chroma"

d, lo, hi = msw.distance(A, B)  # the (pseudo-)metric value with a 95% CI on D
```

See `examples/quickstart.py` for a ground-truth demo on Gaussian random fields
where the true W2 distance is known in closed form.

## API

| Call | Returns |
|---|---|
| `msw.test(A, B, groups=None, splitting="balanced")` | `TestResult`: `T`, exact `p` (sign-flip permutation subgroup, no Monte-Carlo), `p_cct` (Cauchy combination), `components`, `component_p`, `per_level` profile, `stationarity_index`, `n_used`, `G`, `.diagnose()` |
| `msw.distance(A, B)` | `(d, lo, hi)` — fixed-weight multi-scale sliced distance with a Student-t CI over image groups. The CI is on the method's value **D**, *not* a certified bound on true W2 |
| `msw.stationarity_index(A)` | Relative spread of position-pooled filter energies. Calibration: Gaussian fields ≈ 0.005, random photo crops ≈ 0.07, aligned faces ≈ 0.22. `msw.test` warns above 0.1 |
| `msw.SequentialCertifier(C, size)` | **Experimental** anytime-valid sequential test: feed fresh batches, stop the moment `certified` is True — the false-alarm guarantee holds under optional stopping |
| `msw.MultiScaleSW`, `msw.PortfolioSW`, `msw.SWSlicer`, `msw.CSWSlicer` | The underlying estimators (features / blocked estimates / statistics) |
| `msw.spectral` | Gaussian-random-field utilities: spectra, samplers, closed-form W2, frequency envelopes |

## Scope — what it sees and what it cannot

**Detects** differences expressible in translation-invariant local pixel
statistics, with sensitivity tilted toward fine scales: noise, blur, spectral
changes, texture/marginal changes, fine chromatic artifacts — including many
that feature-based metrics (FID/KID/CMMD/DINOv2) are structurally blind to
because their input resize destroys sub-resolution texture.

**Blind to, by construction:** spatially constant offsets and color casts (the
filter banks drop DC); instance identity / memorization (it is a
distribution-level statistic — two sets whose *pooled patches* match are
equal to it); alignment/registration information (it is shift-invariant — the
point). Semantic composition changes need ~10× more samples than feature
metrics.

**Stationarity:** the statistic pools patch positions, which is exact for
(torus-)stationary data and an approximation otherwise. `msw.test` computes a
stationarity index and warns above 0.1 (e.g. aligned face datasets): there,
prefer image-split nulls and interpret cross-pipeline comparisons with care.

## Calibration

- The four-term floor correction makes the statistic **exactly zero-mean under
  H0 at every N** (FID-type metrics carry an O(K/N) bias floor).
- `p` is an **exact** finite-sample p-value from a sign-flip permutation
  subgroup (2^G equally-likely relabelings under H0); min attainable p is
  1/2^G — raise `groups` for smaller levels.
- `p_cct` combines the per-component exact p-values by Cauchy combination,
  level-valid under arbitrary dependence.

## Speed

Wall-clock per evaluation at N=500, 32px, V100 (fp32):

| | msw.test (T + exact p) | SW | FID | CMMD | DINOv2-FD |
|---|---|---|---|---|---|
| time | **≈ 0.28 s** (0.16 s with a reused estimator) | 0.03 s | 4.1 s | 10.6 s | 6.7 s |

(the exact-p calibration itself costs ~5 ms; the feature metrics above return
a single uncalibrated point value.)

## Tests

```bash
pip install pytest && pytest tests/
```

## License

MIT.
