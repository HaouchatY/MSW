#!/bin/bash
# Stable Diffusion 1.4 on the COCO-30k captions, iterations 95-100 of 100 (DDIM),
# against COCO-30k.  Generation is ~33 V100-hours: start one worker per GPU
# (CUDA_VISIBLE_DEVICES=k python gen_sd.py --worker --compile &); shards are
# claimed atomically and a restart resumes.
set -e
cd "$(dirname "$0")/../common"
python prep_coco.py
python gen_sd.py --worker --compile
python measure.py coco512 sd_s95 sd_s96 sd_s97 sd_s98 sd_s99 sd_s100
python analyze.py sd_iter --out ../03_sd_iterations/results
