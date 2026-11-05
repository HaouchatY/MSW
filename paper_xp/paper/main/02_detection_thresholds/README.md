# Table 1 — Detection thresholds at N = 500

ε₀.₈: the smallest perturbation on each grid detected with power 0.8 (lower is better). Each baseline is thresholded at the 95th percentile of its own null, 100 null repetitions per draw family (coverage, which falls under the alternative, at the 5th); MSW is `msw.test` at α = 0.05, which draws no null. 12 alternative repetitions per rung for the baselines, 48 for MSW.

| | FID | KID | CMMD | DINOv2 | C2ST | cov. | SW | CSW | pSWD | spec. | P–S | P–S★ | **MSW** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields, slope | 0.128 | 0.064 | 0.064 | 0.128 | 0.128 | 0.256 | -- | 0.256 | -- | 0.128 | 0.032 | **0.008** | **0.008** |
| Fields, noise | 0.16 | 0.08 | 0.08 | 0.08 | 0.16 | 0.16 | -- | -- | -- | 0.16 | 0.08 | **0.01** | 0.04 |
| Fields, marginal (1/df) | 1/12 | 1/12 | 1/12 | 1/12 | 1/6 | 1/6 | 1/6 | 1/6 | 1/12 | 1/3 | 1/12 | 1/12 | **1/192** |
| t3 fields, slope | 0.128 | **0.064** | 0.128 | 0.128 | 0.128 | 0.256 | -- | 0.256 | -- | 0.128 | 0.128 | **0.064** | **0.064** |
| CelebA, blur | 0.33 | 0.33 | 0.33 | 0.22 | 0.33 | 0.5 | 0.75 | 0.5 | 0.75 | 0.5 | 0.33 | 0.22 | **0.1** |
| CelebA, added noise | 0.01 | 0.005 | 0.0025 | 0.01 | 0.005 | 0.01 | -- | 0.04 | 0.02 | 0.04 | 0.02 | 0.02 | **0.0005** |
| CelebA, shading | 0.08 | 0.04 | 0.04 | 0.04 | 0.04 | 0.08 | 0.16 | 0.16 | 0.16 | 0.16 | 0.08 | 0.04 | **0.02** |
| CIFAR-10, blur | 0.5 | 0.33 | 0.22 | 0.5 | 0.33 | -- | -- | 0.5 | 0.5 | 0.5 | 0.33 | 0.22 | **0.15** |
| CIFAR-10, checkerboard | 0.01 | 0.01 | 0.02 | 0.02 | 0.01 | -- | 0.02 | -- | 0.02 | 0.02 | -- | 0.02 | **0.000625** |

Baselines ran on the grids in `../../common/bench.py` (`BASE`); rungs below them (`FINE`) were run for MSW only, so a baseline entry at its grid floor bounds its threshold from above. `--` = not detected on the grid. P–S★ is the Portilla–Simoncelli statistics at full strength (max-z over four variants). The measured size of the MSW test is in [`../../supplement/13_test_size`](../../supplement/13_test_size).

```
python run.py && python table.py        # MSW column, ~1 GPU-hour; baselines: results/baselines.json
```
