"""CMMD Sec. 6.3 / Fig. 8: the metrics against the number of images.

Stable Diffusion 1.4 (final images) against COCO-30k, on random caption subsets
of N images (the same captions on both sides, as the paper evaluates
text-to-image models).  The paper's range is 5k-30k with one draw per N; we
add N = 1k and 2k, and take R = 5 independent subsets per N < 30k so that
every metric's spread is visible, not just its trend.

FID and CMMD come from the cached features; MSW streams the images again,
because the number of positions it keeps per image depends on N.
"""
import json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA, GEN, FEAT, results                         # noqa: E402
from feats import cmmd, fid                               # noqa: E402
import msw                                                 # noqa: E402
from msw_stream import MSWStream                           # noqa: E402

DEV = torch.device("cuda:0")
msw.set_device(DEV)
NS = (1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000)
R = 5
FULL = 30000


def msw_subset(real, fake, idx):
    st = MSWStream(3, 512, len(idx), store="cpu")
    feats = []
    for mm in (real, fake):
        for i in range(0, len(idx), 64):
            sel = np.sort(idx[i:i + 64])
            x = torch.as_tensor(mm[sel], device=DEV).permute(0, 3, 1, 2).float().div_(255)
            st.add(x)
        feats.append(st.features())
    return st.distance(*feats)


if __name__ == "__main__":
    real = np.memmap(f"{DATA}/coco512.u8", np.uint8, "r", shape=(FULL, 512, 512, 3))
    fake = np.memmap(f"{GEN}/sd_final512.u8", np.uint8, "r", shape=(FULL, 512, 512, 3))
    ri, rc = np.load(f"{FEAT}/coco512_inc.npy"), np.load(f"{FEAT}/coco512_clip.npy")
    fi, fc = np.load(f"{FEAT}/sd_s100_inc.npy"), np.load(f"{FEAT}/sd_s100_clip.npy")
    path = f"{results(os.path.dirname(os.path.abspath(__file__)))}/sample_size.json"
    rows = json.load(open(path)) if os.path.exists(path) else []
    have = {(r["N"], r["rep"]) for r in rows}
    rng = np.random.default_rng(7)
    for N in NS:
        for rep in range(1 if N == FULL else R):
            idx = np.arange(FULL) if N == FULL else rng.permutation(FULL)[:N]
            if (N, rep) in have:
                continue
            t0 = time.time()
            # the pairing: image i of both sets belongs to caption i; keep it,
            # and keep caption order inside the subset (blocks are contiguous)
            idx = np.sort(idx)
            m = msw_subset(real, fake, idx)
            rows.append({"N": N, "rep": rep, "fid": fid(ri[idx], fi[idx]), "cmmd": cmmd(rc[idx], fc[idx]),
                         "msw": m["d"], "msw_lo": m["lo"], "msw_hi": m["hi"], "sec": time.time() - t0})
            json.dump(rows, open(path, "w"), indent=1)
            r = rows[-1]
            print(f"N={N:6d} rep={rep}  FID {r['fid']:.3f}  CMMD {r['cmmd']:.4f}  "
                  f"MSW {r['msw']:.5f} [{r['msw_lo']:.5f}, {r['msw_hi']:.5f}]  ({r['sec']:.0f}s)", flush=True)
    print("SAMPLE SIZE DONE", flush=True)
