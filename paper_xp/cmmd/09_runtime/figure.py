"""results/runtime.json -> data/features.csv, data/distance.csv"""
import json, os

H = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(f"{H}/results/runtime.json"))
os.makedirs(f"{H}/data", exist_ok=True)
with open(f"{H}/data/features.csv", "w") as f:            # one batch of 32 images, ms
    f.write("y;label;ms\n")
    for i, (k, lab) in enumerate((("msw_features_batch32_256px_ms", "MSW, 256 px"),
                                  ("msw_features_batch32_512px_ms", "MSW, 512 px"),
                                  ("inception_batch32_ms", "Inception (FID)"),
                                  ("clip336_fp16_batch32_ms", "CLIP-336 fp16 (CMMD)"),
                                  ("clip336_fp32_batch32_ms", "CLIP-336 fp32"))):
        f.write(f"{i};{lab};{T[k][0]:.3f}\n")
with open(f"{H}/data/distance.csv", "w") as f:            # n = 30,000 per set, s
    f.write("y;label;s\n")
    for i, (v, lab) in enumerate(((T["frechet_gpu_ms"][0] / 1000, "Fr\\'echet, GPU"),
                                  (T["mmd_gpu_ms"][0] / 1000, "MMD, GPU (CMMD)"),
                                  (T["frechet_scipy_cpu_ms"][0] / 1000, "Fr\\'echet, SciPy CPU"),
                                  (T["msw_distance_n30000_512px_ms"] / 1000, "MSW, 512 px"))):
        f.write(f"{i};{lab};{v:.3f}\n")
