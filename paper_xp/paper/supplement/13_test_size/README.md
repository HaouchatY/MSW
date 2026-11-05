# Sec. 5 / App. size — Size of the MSW test

| Family | N | reps | rejections | size | 95% Wilson CI |
|---|---|---|---|---|---|
| Gaussian fields | 16 | 1000 | 39 | 0.039 | [0.029, 0.053] |
| Gaussian fields | 500 | 1000 | 53 | 0.053 | [0.041, 0.069] |
| t3 fields | 16 | 1000 | 40 | 0.040 | [0.030, 0.054] |
| t3 fields | 500 | 1000 | 43 | 0.043 | [0.032, 0.057] |
| CelebA, 32 px crops | 16 | 1000 | 50 | 0.050 | [0.038, 0.065] |
| CelebA, 32 px crops | 500 | 1000 | 58 | 0.058 | [0.045, 0.074] |
| CelebA, 64 px | 16 | 1000 | 50 | 0.050 | [0.038, 0.065] |
| CelebA, 64 px | 500 | 1000 | 65 | 0.065 | [0.051, 0.082] |
| CIFAR-10 | 16 | 1000 | 50 | 0.050 | [0.038, 0.065] |
| CIFAR-10 | 500 | 1000 | 51 | 0.051 | [0.039, 0.066] |
| **pooled** | all | 10000 | 499 | **0.0499** | [0.0458, 0.0543] |

Rejection rate of `msw.test` at α = 0.05 when both sets are independent draws of N images from the same null draw (the null draw of fields slope, t3 slope, CelebA blur, CelebA shading and CIFAR-10 blur), 1000 repetitions per cell with house seeds 70000 + r. Per cell the size lies between 0.039 and 0.065, and pooled over the ten cells it is 499/10000 = 0.0499, with a Wilson interval that contains 0.05; one cell (CelebA 64 px, N = 500) has a lower bound of 0.051, about what one expects among ten intervals. The paper's earlier figures (0.037–0.065 per family, 403/7780 = 0.0518 pooled) were measured on a different set of null cells (7780 draws); this rerun uses a fixed grid of five families × N ∈ {16, 500}.

```
python run.py && python table.py        # ~20 GPU-minutes
```
