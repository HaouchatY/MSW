# The CMMD experiments, rerun with MSW

The experiments of Jayasumana et al., *Rethinking FID: Towards a Better Evaluation Metric for Image Generation* ([arXiv:2401.09603](https://arxiv.org/abs/2401.09603)), rerun with public models, with MSW evaluated beside FID and CMMD on the same images.

| | experiment | CMMD paper |
|---|---|---|
| [01](01_moment_matched_mixtures) | moment-matched mixtures | Table 2 |
| [02](02_embedding_normality) | normality of the embeddings | Fig. 2, App. A |
| [03](03_sd_iterations) | late Stable Diffusion iterations | Fig. 6 |
| [04](04_sd_iterations_few_images) | the same, from few images | — |
| [05](05_sample_size) | number of images | Fig. 8 |
| [06](06_latent_noise_coarse) | VQGAN latent noise, 16-pixel tokens | Figs. 1, 7 |
| [07](07_latent_noise_fine) | VQGAN latent noise, 8-pixel tokens | Figs. 1, 7 |
| [08](08_amused_iterations) | refinement iterations of an open Muse | Figs. 3–4, Table 3 |
| [09](09_runtime) | runtime | Table 4 |

Reference set: COCO-30k (val 2014). FID follows clean-fid (antialiased PIL resize to 299); CMMD follows the official implementation (CLIP ViT-L/14@336px, RBF σ = 10); MSW is `msw.distance` at native resolution. Muse is not public: aMUSEd-256 stands in for it, and the human ratings of Table 3 are not reproduced.

## Running

```
pip install -r requirements.txt
export MSW_XP_WORK=/data/msw_xp      # ~110 GB: images, latents, tokens, features
```

`common/` holds the shared pipeline (data, generation, metrics, measurement). Each experiment folder holds its code (`run.sh` or `run.py`), its results (`results/`), and its figure (`figure.py` writes `data/`, `figure.tex` builds `figure.pdf`; `figure_body.tex` is the bare TikZ picture).
