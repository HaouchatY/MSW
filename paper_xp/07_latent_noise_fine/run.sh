#!/bin/bash
# VQGAN latent noise with 8-pixel tokens (VQ-Diffusion ITHQ VQGAN, 64x64 tokens at
# 512 px), on COCO-30k and on Stable Diffusion 1.4 images (generated in
# 03_sd_iterations); then MSW's per-level decomposition on 2,000 images.
set -e
cd "$(dirname "$0")/../common"
python measure.py coco512 cocoB_p0.00 cocoB_p0.01 cocoB_p0.02 cocoB_p0.05 cocoB_p0.10 cocoB_p0.20 cocoB_p0.30 sdB_p0.00 sdB_p0.01 sdB_p0.02 sdB_p0.05 sdB_p0.10 sdB_p0.20 sdB_p0.30 
python analyze.py cocoB_vq sdB_vq --out ../07_latent_noise_fine/results
python ../07_latent_noise_fine/levels.py
