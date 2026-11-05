# App. (app:dtd) — Held-out textures (DTD)

| | FID | KID | CMMD | DINOv2 | C2ST | cov. | SW | CSW | MSW |
|---|---|---|---|---|---|---|---|---|---|
| A: 47-category mix, max power | 0.17 | 0.17 | 0.58 | 0.17 | 0.50 | 0.17 | 0.42 | 0.25 | **1.00** |
| B: single category, max power | 0.17 | 0.42 | 0.33 | 0.33 | 0.25 | 0.25 | 0.17 | 0.17 | **1.00** |
| C: woven noise, ε₀.₈ (N = 50) | 0.02 | 0.01 | **0.005** | -- | 0.02 | 0.02 | -- | -- | **0.005** |
| C: woven blur, ε₀.₈ (N = 50) | 1.5 | 0.75 | **0.5** | 1.5 | 1 | -- | -- | -- | **0.5** |
| D: woven noise 0.02, N = 8 | 0.00 | 0.17 | **1.00** | 0.00 | 0.17 | 0.00 | 0.17 | 0.00 | **1.00** |
| D: woven noise 0.02, N = 16 | 0.00 | 0.92 | **1.00** | 0.25 | 0.67 | 0.17 | 0.00 | 0.08 | **1.00** |
| D: woven noise 0.02, N = 25 | 0.58 | **1.00** | **1.00** | 0.00 | 0.50 | 0.75 | 0.08 | 0.08 | **1.00** |
| D: woven blur 1, N = 8 | 0.00 | 0.00 | 0.17 | 0.08 | 0.25 | 0.00 | 0.00 | 0.00 | **1.00** |
| D: woven blur 1, N = 16 | 0.00 | 0.42 | **1.00** | 0.17 | 0.75 | 0.00 | 0.00 | 0.00 | **1.00** |
| D: woven blur 1, N = 25 | 0.00 | 0.83 | **1.00** | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | **1.00** |

32-px crops of DTD r1.0.1 (47 categories, 5640 images, not used in development), two disjoint pools of 2000 crops; null = pool against pool, alternative = pool against the other pool with added noise or Gaussian blur. A: one crop per image from 4000 images of all categories, noise 0.0025 and blur 0.33 (the CelebA-calibrated amplitudes), N = 8–100; B: woven, marbled and honeycombed separately, same amplitudes, N = 8–50; C: woven at N = 50 on a grid of amplitudes; D: woven at N = 8–25. 12 repetitions per cell; baselines (archived exp67) detect beyond the 95th percentile of the cell's 12 nulls, MSW when `msw.test` rejects at α = 0.05. Per-cell powers are in `table.csv`.

MSW reaches power 0.8 on A only for blur (0.92 at N = 50, 1.00 at N = 100; noise peaks at 0.67), and on B for blur at N = 50 in all three categories (1.00) and for noise on marbled and honeycombed (0.83–0.92 from N = 16; woven at most 0.42); no baseline exceeds 0.58 on A or B. On C, MSW and CMMD both sit at the lowest grid amplitude for noise (0.005) and blur (0.5), so both values are upper bounds. On D, MSW has power 1.00 at every N, CMMD from N = 8 on noise and from N = 16 on blur. MSW rejects 18 of 252 null repetitions (0.071; 6/60 on the mixed pools).

```
python run.py && python table.py     # MSW column, ~10 GPU-minutes; baselines: results/baselines.json
```
