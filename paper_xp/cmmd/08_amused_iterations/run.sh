#!/bin/bash
# aMUSEd-256 (open Muse) on the COCO-30k captions: the full-image prediction after
# each of its 12 refinement iterations, against COCO-30k at 256 px.
set -e
cd "$(dirname "$0")/../common"
python prep_coco.py
python gen_amused.py
python measure.py coco256 amused_s1 amused_s2 amused_s3 amused_s4 amused_s5 amused_s6 amused_s7 amused_s8 amused_s9 amused_s10 amused_s11 amused_s12 
python analyze.py amused_iter --out ../08_amused_iterations/results
