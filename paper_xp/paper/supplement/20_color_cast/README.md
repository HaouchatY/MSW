# App. "Global color cast" — constant chromatic cast on CelebA crops

| ε | FID | KID | CMMD | DINOv2 | C2ST | coverage | SW | CSW | MSW | MSW−PX |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0008 | | | | | | | | | 0.02 | 0.02 |
| 0.0025 | | | | | | | | | 0.02 | 0.02 |
| 0.005 | −0.50 | +0.31 | +0.24 | −0.28 | −0.13 | −0.05 | +1.16 | +0.43 | 0.02 | 0.02 |
| 0.01 | −0.57 | +0.50 | +1.43 | −0.26 | +0.11 | −0.06 | +1.29 | +0.52 | 0.06 | 0.02 |
| 0.02 | −0.63 | +1.26 | **+4.74** | −0.16 | +1.07 | −0.15 | +1.80 | +0.88 | 0.53 | 0.02 |
| 0.04 | −0.47 | **+4.13** | **+9.93** | +0.23 | **+3.55** | −0.10 | **+3.31** | +2.12 | **1.00** | 0.02 |
| 0.08 | | | | | | | | | **1.00** | 0.02 |
| null (size) | | | | | | | | | 0.073 | 0.067 |

A cast ε(1,−1,0)/√2 is added to every 32-px CelebA crop of set B (no clamping), N = 500. Baselines: deflection z over 12 repetitions, bold when z ≥ 2.4865 (archived). MSW: rejection rate of the exact test at 0.05 over 100 repetitions per amplitude (300 nulls), with the shipped portfolio and without the raw-pixel row (MSW−PX); bold when ≥ 0.8. With the pixel row MSW detects from ε = 0.04 (0.53 at 0.02); without it the test stays at 0.02 over the whole 100× grid (0.0008–0.08).

Without the pixel row the p-value is not bit-identical across amplitudes as the original text claimed: it is identical in 84 of 100 repetitions and otherwise moves by at most 2/4096 (T moves by ≤ 5·10⁻⁵). The cause is float32 arithmetic, not a component that sees the cast: the shipped filters are zero-mean only to ~1·10⁻⁶ per channel (built and centred in float32) and x + εc is rounded. On CPU (N = 128, 256 group elements) the transport, tail/shape and product columns change by ≤ 3·10⁻⁴ in float64 with the float32 filters (growing linearly in ε), and by ≤ 2·10⁻¹³ once the filters are re-centred in float64, at ε up to 1; p is unchanged in every case. Clamping plays no role (none is applied).

```
python run.py      # ~25 GPU-minutes
python table.py
```
