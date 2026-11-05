"""COCO-30k (val 2014) as the CMMD paper's reference set.

sayakpaul/coco-30-val-2014 is the standard 30k random image-caption sample of
COCO 2014 val used for text-to-image FID.  Each image is centre-cropped to its
largest square and resized with PIL bicubic -- exactly the official CMMD loader
(io_util._center_crop_and_resize) -- to 512 px (reference for Stable Diffusion,
which generates at 512) and 256 px (reference for the 256-px Muse stand-in and
for the VQGAN distortion experiments).  Global index = shard order, so caption i
is the prompt of generated image i in every experiment.
"""
import glob, io, json, os, sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from huggingface_hub import snapshot_download

from paths import DATA

SNAP = f"{snapshot_download('sayakpaul/coco-30-val-2014', repo_type='dataset')}/data"
SHARDS = sorted(glob.glob(f"{SNAP}/train-*.parquet"))
OUT = DATA
N = 30000
SIZES = (512, 256)


def crop_resize(im, size):
    w, h = im.size
    s = min(w, h)
    box = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
    return np.asarray(im.crop(box).resize((size, size), resample=Image.BICUBIC))


def work(task):
    shard, r0, r1, g0 = task
    t = pq.read_table(shard, columns=["image"]).slice(r0, r1 - r0)
    mm = {s: np.memmap(f"{OUT}/coco{s}.u8", np.uint8, "r+", shape=(N, s, s, 3)) for s in SIZES}
    for j, cell in enumerate(t.column("image").to_pylist()):
        im = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        for s in SIZES:
            mm[s][g0 + j] = crop_resize(im, s)
    for m in mm.values():
        m.flush()
    return r1 - r0


if __name__ == "__main__":
    counts = [pq.ParquetFile(s).metadata.num_rows for s in SHARDS]
    assert sum(counts) == N, counts
    caps = []
    for s in SHARDS:
        caps += pq.read_table(s, columns=["caption"]).column("caption").to_pylist()
    json.dump(caps, open(f"{OUT}/captions.json", "w"))
    for s in SIZES:
        np.memmap(f"{OUT}/coco{s}.u8", np.uint8, "w+", shape=(N, s, s, 3)).flush()
    tasks, g0 = [], 0
    for shard, c in zip(SHARDS, counts):
        step = -(-c // 8)
        for r0 in range(0, c, step):
            r1 = min(r0 + step, c)
            tasks.append((shard, r0, r1, g0 + r0))
        g0 += c
    done = 0
    with Pool(40) as p:
        for k in p.imap_unordered(work, tasks):
            done += k
            print(f"{done}/{N}", flush=True)
    print("COCO PREPARED", flush=True)
