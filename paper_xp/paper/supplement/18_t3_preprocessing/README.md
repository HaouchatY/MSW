# Preprocessing on the t3 row

ε₀.₈ on the t3-fields slope row at N = 500, with and without per-image standardization:

| inputs | FID | KID | CMMD | DINOv2 | C2ST | cov. | SW | CSW | pSWD | spec. | P–S | MSW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standardized | 0.128 | **0.064** | 0.128 | 0.128 | 0.128 | 0.256 | -- | 0.256 | -- | 0.128 | 0.128 | **0.064** |
| raw | 0.128 | **0.064** | 0.128 | 0.128 | 0.128 | 0.256 | -- | -- | -- | -- | 0.256 | **0.064** |

32-px fields with spectrum |k|⁻³ as a Student-t(3) scale mixture, slope perturbation, house seeds; the standardized row is Table 1's (`main/02`), the raw row is this folder's rerun of every metric on the same draws without the standardization step. Baselines: 100 null and 12 alternative repetitions, thresholded at the 95th percentile of their own null (coverage at the 5th); MSW: `msw.test` at α = 0.05, 48 repetitions per rung, null size 0.06 (3/48). On raw inputs MSW rejects with power 0.44 at ε = 0.032 and 1.00 at 0.064, as on standardized inputs; FID, KID, CMMD, DINOv2, C2ST and coverage are unchanged, and standardization helps only CSW (-- → 0.256), the spectrum test (-- → 0.128) and P–S (0.256 → 0.128).

P–S★ has no raw run. The archived raw run (`results/baselines.json`, exp33: 15 repetitions, mean-z ≥ 2.4865 rule, no DINOv2, pSWD, spectrum or P–S, and the old MSW statistic) is superseded by `results/baselines_raw_rerun.json`.

```
python run.py && python run.py baselines && python table.py   # ~6 + ~40 GPU-minutes (run.py estimate; not logged)
```
