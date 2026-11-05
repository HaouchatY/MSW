#!/bin/bash
# VQGAN latent noise with 16-pixel tokens (aMUSEd-256 VQGAN), on COCO-30k and on
# aMUSEd-256 images; nested random token replacement at p = 0 ... 0.4.
set -e
cd "$(dirname "$0")/../common"
python prep_coco.py
python gen_amused.py
python vq_tokens.py
python measure.py coco256 amused_s12 coco_vq_p0.00 coco_vq_p0.05 coco_vq_p0.10 coco_vq_p0.15 coco_vq_p0.20 coco_vq_p0.25 coco_vq_p0.30 coco_vq_p0.35 coco_vq_p0.40 amused_vq_p0.05 amused_vq_p0.10 amused_vq_p0.15 amused_vq_p0.20 amused_vq_p0.25 amused_vq_p0.30 amused_vq_p0.35 amused_vq_p0.40 
python analyze.py coco_vq amused_vq --out ../06_latent_noise_coarse/results
