# Tab. "Implementation checks" — baselines against the official packages

| quantity | ours | official | package | relative gap |
|---|---|---|---|---|
| FID (null pair) | 24.4293585 | 24.4290304 | pytorch-fid | 1.3e-05 |
| FID (noise pair) | 32.7259544 | 32.7258389 | pytorch-fid | 3.5e-06 |
| density | 0.944042981 | 0.944042969 | prdc | 1.3e-08 |
| coverage | 0.961914062 | 0.961914062 | prdc | **0** |
| FID (null pair) | 24.4293585 | 24.5435259 | torch-fidelity | 4.7e-03 |
| FID (noise pair) | 32.7259544 | 32.8755484 | torch-fidelity | 4.6e-03 |
| KID (null pair) | 7.35e-05 | 4.08e-05 | torch-fidelity | 0.80 |
| KID (noise pair) | 0.00916972 | 0.00930731 | torch-fidelity | 1.5e-02 |

Fixed CIFAR-10 pools of 2048 images (A, B, and B plus white noise 0.01, clamped), written as 8-bit PNGs; the official packages read the PNGs and ours sees the same 8-bit values (round(255x)/255, bit-identical to the decoded PNGs). The GPU rerun gives a FID gap of 1.3·10⁻⁵ to pytorch-fid, against 2·10⁻⁷ in the paper. The cause is GPU arithmetic in the feature extractor, not quantisation or the Fréchet formula: on CPU (float32) the two Inception passes give bit-identical features (batch 200 vs 50) and the two formulas agree to 2·10⁻¹¹ (24.4258735, the paper's value), while on the GPU each side drifts by ~1.4·10⁻⁴ from that value (TF32/cuDNN kernels that depend on batch size). clean-fid failed in the rerun (DataLoader error), so its row is missing; KID on the null pair is a difference of two numbers near zero, and its relative gap is meaningless. DINOv2-FD signal excess (noise minus null pair): +14.59 with the straight 224 resize, +13.39 with the reference 256-then-crop.

```
python run.py && python table.py      # a few GPU-minutes
```
