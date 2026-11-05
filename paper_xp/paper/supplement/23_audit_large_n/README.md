# Tab. "Systematic shift at N = 800" — preprocessing audit at large N

| pair | ΔFID | ΔDINOv2-FD | ΔCMMD | MSW T | MSW d |
|---|---|---|---|---|---|
| bilinear vs. nearest | +0.830 | +3.14 | +0.130 | 20.2 | +0.0213 |
| lanczos vs. nearest | +0.721 | +2.10 | +0.090 | 9.9 | +0.0081 |
| bicubic vs. JPEG q90 | +0.719 | +1.14 | +0.265 | 19.7 | −0.0026 |
| nearest vs. box | +0.130 | +0.01 | +0.018 | 7.7 | +0.0037 |
| lanczos vs. torchvision AA | +0.031 | +0.00 | +0.028 | 8.5 | +0.0038 |

COCO validation images through eight resize pipelines (1600 per pipeline); set A is the first 800 images through pipeline i, set B the last 800 through pipeline j, one evaluation per cell; `table.csv` holds all 28 pairs and the 8 same-pipeline nulls. Baseline columns are archived shifts against the mean of the two same-pipeline nulls; FID's null varies by 0.435 (population std) across the eight pipelines. MSW (`msw.test`, `msw.distance`) rejects 27 of 28 pairs at 0.05 (26 under Bonferroni at 0.05/28; the miss is PIL bicubic vs. torchvision bicubic with antialiasing, p = 0.64) and 0 of 8 nulls. The distance's 95 % interval excludes 0 on 17 of 28 pairs only, and on bicubic vs. JPEG q90 (T = 19.7) it sits at the null level (−0.0026 against −0.0033 on every null).

```
python ../../main/05_pipeline_audit/prep.py && python run.py    # ~30 GPU-minutes
python table.py
```
