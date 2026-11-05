# Generic two-sample tests on the same pooled slices

ε₀.₈ (smallest rung with power ≥ 0.8 at α = 0.05, kept at every larger rung; lower is better). Bold: the lowest value in the row, counting MSW's matched-budget value on the 8192 rows (MSW does not depend on the budget).

| | budget | KS-max | MMD | energy | W₂ | transport only | **MSW** | best generic / MSW |
|---|---|---|---|---|---|---|---|---|
| Fields, slope | 2048 | 0.064 | 0.064 | 0.064 | 0.032 | **0.008** | **0.008** | 4× |
| Fields, slope | 8192 | 0.064 | 0.032 | 0.032 | 0.032 | — | — | 4× |
| Fields, marginal (1/df) | 2048 | 1/12 | 1/12 | 1/12 | 1/12 | 1/48 | **1/192** | 16× |
| Fields, marginal (1/df) | 8192 | 1/12 | 1/24 | 1/24 | 1/24 | — | — | 8× |
| CelebA, noise | 2048 | 0.00125 | 0.00125 | 0.00125 | 0.00125 | **0.0005** | **0.0005** | 2.5× |
| CelebA, noise | 8192 | 0.00125 | 0.00125 | **0.0005** | **0.0005** | — | — | 1× |
| CelebA, blur | 2048 | 0.22 | 0.22 | 0.15 | 0.15 | **0.1** | **0.1** | 1.5× |
| CelebA, blur | 8192 | 0.15 | 0.15 | 0.15 | **0.1** | — | — | 1× |
| CelebA, shading | 2048 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | **0.02** | 4× |
| CelebA, shading | 8192 | 0.04 | 0.08 | 0.04 | 0.08 | — | — | 2× |
| CIFAR-10, checkerboard | 2048 | 0.00125 | 0.0025 | 0.005 | 0.02 | 0.01 | **0.000625** | 2× |
| CIFAR-10, checkerboard | 8192 | 0.00125 | 0.0025 | 0.0025 | 0.01 | — | — | 2× |

KS-max, kernel MMD, energy distance and a subsampled W₂ receive MSW's own pooled slices, L = 16 matched blocks, 4096-element pair group and max-z exact test, so only the per-slice discrepancy differs; N = 500, 24 repetitions per rung, house seeds. At the matched budget (2048 points per side per slice) MSW leads every setting by 1.5–16×; at 8192 points the best generic test ties it on CelebA noise and blur and gains a rung on four of six rows. With only the transport rows swapped and TL+PX+J kept (`table.tex`, block c), every generic test matches MSW on the marginal, shading and checkerboard rows and stays 1.5–8× behind on fields slope, CelebA noise and CelebA blur; measured sizes lie in 0.030–0.066 (400–700 nulls, block e), and removing the [0, 1] clamp from the CIFAR checkerboard leaves every power unchanged up to one repetition (`data/pooled_clampcontrol.csv`).

```
python run.py grid && python run.py k512 && python run.py clamp && python table.py
# ~5 + ~2 GPU-hours + ~15 GPU-minutes (estimates in run.py; no timing recorded in results/)
```
