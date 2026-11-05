# Table 3 — Which preprocessing pipeline made these images?

Eight real pipelines (five PIL resizes, torch bicubic with and without antialiasing, PIL bicubic then JPEG 90) applied to the same 1600 COCO images at 384 px; nothing is injected. A pair of pipelines counts as resolved when the method separates it in at least half of 30 repetitions, each comparing N images through one pipeline with N different images through the other.

| | N = 16 | N = 64 | N = 128 | permutation, N = 16 | permutation, N = 64 |
|---|---|---|---|---|---|
| FID | 0/28 | 0/28 | 0/28 | 0/28 | 0/28 |
| KID | 0/28 | 0/28 | 0/28 | 0/28 | 0/28 |
| CMMD | 7/28 | 16/28 | 23/28 | 7/28 | 17/28 |
| DINOv2-FD | 0/28 | 0/28 | 0/28 | 0/28 | 0/28 |
| C2ST | 0/28 | 0/28 | 0/28 | | |
| spectrum | 0/28 | 0/28 | | | |
| **MSW** | **16/28** | **23/28** | **26/28** | | |

Baselines: a two-sided threshold from a held-out half of the same-pipeline nulls (60 repetitions per pipeline), and an exact permutation test on the method's own features (199 relabellings). MSW: `msw.test` at α = 0.05, no null drawn; its measured size on the same-pipeline nulls is 0.052, 0.035 and 0.046.

```
python prep.py && python run.py && python table.py     # ~2 GPU-hours; prep streams COCO from Hugging Face
```
