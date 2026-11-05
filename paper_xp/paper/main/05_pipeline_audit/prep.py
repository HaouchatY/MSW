"""Eight real preprocessing pipelines applied to the same 1600 COCO images.

Source: the first 1600 images of sayakpaul/coco-30-val-2014 (streamed from
Hugging Face) whose shorter side is at least 384 px, centre-cropped to a square.
Each pipeline delivers 384 px uint8 RGB, a genuine downsample:
  pil_{bicubic,bilinear,lanczos,nearest,box}   PIL resize
  tv_bicubic_{aa,noaa}                         torch bicubic with / without antialias
  pil_bicubic_jpeg90                           PIL bicubic, then JPEG quality 90
Two pipelines differ iff they are different code paths; nothing is injected.
Writes $MSW_XP_DATA/grid/<pipeline>.npy, (1600, 384, 384, 3) uint8, 5.7 GB.
"""
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from draws import DATA

OUT = os.path.join(DATA, "grid")
S, NIM = 384, 1600
PIPELINES = ["pil_bicubic", "pil_bilinear", "pil_lanczos", "pil_nearest",
             "pil_box", "tv_bicubic_aa", "tv_bicubic_noaa", "pil_bicubic_jpeg90"]
PILF = {"pil_bicubic": Image.BICUBIC, "pil_bilinear": Image.BILINEAR,
        "pil_lanczos": Image.LANCZOS, "pil_nearest": Image.NEAREST, "pil_box": Image.BOX}


def deliver(crop, v):
    if v in PILF:
        return np.asarray(crop.resize((S, S), PILF[v]), dtype=np.uint8)
    if v == "pil_bicubic_jpeg90":
        b = io.BytesIO()
        crop.resize((S, S), Image.BICUBIC).save(b, "JPEG", quality=90)
        b.seek(0)
        return np.asarray(Image.open(b).convert("RGB"), dtype=np.uint8)
    t = torch.from_numpy(np.asarray(crop, dtype=np.uint8)).permute(2, 0, 1)[None].float()
    r = F.interpolate(t, size=(S, S), mode="bicubic", align_corners=False,
                      antialias=v.endswith("_aa"))
    return r.clamp(0, 255).round().to(torch.uint8)[0].permute(1, 2, 0).numpy()


if __name__ == "__main__":
    from datasets import load_dataset
    os.makedirs(OUT, exist_ok=True)
    buf = {v: [] for v in PIPELINES}
    for ex in load_dataset("sayakpaul/coco-30-val-2014", split="train", streaming=True):
        im = ex["image"].convert("RGB")
        w, h = im.size
        m = min(w, h)
        if m < S:
            continue
        crop = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
        for v in PIPELINES:
            buf[v].append(deliver(crop, v))
        if len(buf[PIPELINES[0]]) >= NIM:
            break
    for v in PIPELINES:
        np.save(os.path.join(OUT, f"{v}.npy"), np.stack(buf[v]))
