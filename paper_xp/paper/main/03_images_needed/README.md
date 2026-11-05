# Table 2 — Images needed

N★: the fewest images per set at which a method reaches power 0.8 and keeps it at every larger rung of the ladder 8, 16, …, 1024 (2048 for CelebA noise), at a fixed perturbation per row. Baselines: 95th percentile of 50 null repetitions per rung, 10 alternative repetitions, from 16 images; MSW: `msw.test` at α = 0.05, 96 repetitions, from 8. `>` = never reached.

| | FID | KID | CMMD | DINOv2 | C2ST | cov. | SW | CSW | pSWD | spec. | P–S | P–S★ | **MSW** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields, slope | 1024 | 256 | 512 | 1024 | > | > | > | > | > | > | 64 | 16 | **8** |
| Fields, noise | > | 512 | 256 | 512 | 1024 | > | > | > | > | > | 64 | 16 | **8** |
| Fields, marginal (1/df) | 512 | 1024 | 512 | 512 | > | > | > | > | 256 | > | 512 | 512 | **32** |
| t3 fields, slope | 128 | 128 | 128 | 512 | 256 | > | > | > | > | 512 | 256 | 128 | **64** |
| CelebA, blur | 256 | 128 | 128 | 64 | 128 | > | > | > | > | 1024 | 256 | 32 | **8** |
| CelebA, added noise | > | > | 512 | > | > | > | > | > | > | > | > | > | **8** |
| CelebA, shading | 64 | 64 | 64 | 128 | 64 | 256 | 1024 | > | 1024 | 512 | 64 | **16** | **16** |
| CIFAR-10, blur | 1024 | 512 | 128 | 1024 | 512 | > | > | 1024 | > | 512 | 256 | 64 | **16** |
| CIFAR-10, checkerboard | 512 | 128 | 1024 | > | 256 | > | > | > | > | > | > | > | **8** |

```
python run.py && python table.py        # MSW column, ~2 GPU-hours; baselines: results/baselines.json
```
