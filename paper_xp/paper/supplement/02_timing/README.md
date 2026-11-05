# Table S1 — Wall-clock per evaluation

Median seconds for one evaluation of one pair of sets of N images (random 32-px 3-channel CelebA crops), feature extraction of both sets included, one-time setup (network loading, SW directions, CSW kernels, P–S pool, MSW's cached portfolio) excluded. 2 warm-up and 25 timed calls per cell, synchronised, on one NVIDIA H200 (torch 2.10.0+cu128, msw 0.3.0). Fastest value per column in bold.

| Metric | N = 200 | N = 500 | N = 1000 |
|---|---|---|---|
| FID | 0.089 s | 0.224 s | 0.495 s |
| KID | 0.083 s | 0.199 s | 0.391 s |
| CMMD | 0.182 s | 0.459 s | 0.920 s |
| DINOv2 | 0.186 s | 0.479 s | 1.02 s |
| C2ST | 0.193 s | 0.310 s | 0.493 s |
| coverage | 0.078 s | 0.194 s | 0.386 s |
| SW | 0.0017 s | 0.0035 s | 0.0047 s |
| CSW | 0.014 s | 0.018 s | 0.024 s |
| pSWD | 0.014 s | 0.027 s | 0.048 s |
| spectrum | **0.00016 s** | **0.00017 s** | **0.00022 s** |
| P–S | 0.0099 s | 0.019 s | 0.035 s |
| MSW distance | 0.0081 s | 0.018 s | 0.032 s |
| MSW test (with exact p-value) | 0.044 s | 0.066 s | 0.105 s |

`msw.distance` costs 0.018 s at N = 500 and `msw.test`, statistic plus exact orbit p-value, 0.066 s; the six feature metrics cost 0.19–0.48 s, dominated by extraction, so the full MSW test is 3–7× cheaper than any of them and the distance alone 11–27×. The raw-pixel statistics (SW, CSW, pSWD, spectrum, P–S) cost 0.0002–0.027 s at N = 500, the same order as the MSW distance or below. The paper's original table timed the baselines under a different protocol on a shared V100 and listed the MSW distance only; every row here uses one protocol and one GPU.

```
python run.py && python table.py        # timed calls total 3 min on one H200, plus model loading
```
