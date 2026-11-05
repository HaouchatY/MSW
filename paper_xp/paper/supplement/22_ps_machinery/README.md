# Tab. "Representation or inference?" — Portilla–Simoncelli inside MSW's machinery

| setting | P–S, plain | P–S, standardised | MSW transport only | ratio (std / T) |
|---|---|---|---|---|
| fields, slope | 0.032 | 0.016 | **0.008** | 2× |
| fields, marginal | >1/12 | >1/12 | **1/48** | >4× |
| CelebA, noise | >0.005 | >0.005 | **0.0005** | >10× |
| CelebA, shading | 0.08 | **0.02** | 0.08 | 0.25× |
| CIFAR, checker | >0.01 | >0.01 | **0.01** | >1× |

ε₀.₈ (smallest rung rejected at 0.05 in ≥ 80 % of 15 repetitions, N = 500) of the P–S statistic vector (plenoptic, 2 scales, 4 orientations, channel-mean image) placed in MSW's 16 blocks, its 4096-element pair group and exact test — "plain" is one U-statistic over the vector, "standardised" is per-coordinate U over its orbit RMS, max-combined — against `msw.test(..., portfolio='T_all')` on the same draws. Standardised P–S needs 2× to more than 10× larger perturbations on four of five rows and is 4× more sensitive on CelebA shading (0.02 against 0.08). The group is MSW's own parity subgroup; the original run used a hand-built subgroup.

```
python run.py && python table.py      # ~15 GPU-minutes
```
