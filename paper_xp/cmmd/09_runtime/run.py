"""CMMD Table 4: runtime of the distances and of feature extraction, plus MSW.

The paper times FD and MMD at n = 30,000, d = 2048 on a TPUv4 (JAX), and one
batch of 32 images through Inception and CLIP.  Same quantities here on one
V100, run on an otherwise idle GPU, median of 5 after one warm-up.  MSW has no
feature extractor in that sense: its "features" are the pyramid and filter
responses of the images themselves, and its distance is the sort-and-Gram pass.
"""
import json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA, GEN, results                              # noqa: E402
from feats import CLIP336, Inception, cmmd, fid          # noqa: E402
import msw                                                # noqa: E402
from msw_stream import MSWStream                          # noqa: E402

DEV = torch.device("cuda:0")
msw.set_device(DEV)


def timed(f, reps=5):
    f(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); f(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000), float(np.std(ts) * 1000)


def fid_scipy(a, b):
    from scipy import linalg
    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cm, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cm.real))


if __name__ == "__main__":
    out = {}
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30000, 2048)).astype(np.float32)
    b = rng.standard_normal((30000, 2048)).astype(np.float32)
    an, bn = a / np.linalg.norm(a, axis=1, keepdims=True), b / np.linalg.norm(b, axis=1, keepdims=True)
    out["frechet_gpu_ms"] = timed(lambda: fid(a, b), 3)
    t = time.perf_counter(); fid_scipy(a, b); out["frechet_scipy_cpu_ms"] = ((time.perf_counter() - t) * 1000, 0.0)
    out["mmd_gpu_ms"] = timed(lambda: cmmd(an, bn), 3)
    x = torch.rand(32, 3, 299, 299, device=DEV)
    inc = Inception(); out["inception_batch32_ms"] = timed(lambda: inc(x))
    x = torch.rand(32, 3, 336, 336, device=DEV)
    clip = CLIP336(dtype=torch.float16); out["clip336_fp16_batch32_ms"] = timed(lambda: clip(x))
    clip32 = CLIP336(dtype=torch.float32); out["clip336_fp32_batch32_ms"] = timed(lambda: clip32(x))
    for res in (256, 512):
        st = MSWStream(3, res, 30000, store="cpu")
        x = torch.rand(32, 3, res, res, device=DEV)
        out[f"msw_features_batch32_{res}px_ms"] = timed(lambda: (st.add(x), st.reset()))
    # the distance itself at n = 30,000, 512 px: features from 30,000 cached images
    real = np.memmap(f"{DATA}/coco512.u8", np.uint8, "r", shape=(30000, 512, 512, 3))
    fake = np.memmap(f"{GEN}/sd_final512.u8", np.uint8, "r", shape=(30000, 512, 512, 3))
    st = MSWStream(3, 512, 30000, store="cpu")
    fs = []
    t = time.perf_counter()
    for mm in (real, fake):
        for i in range(0, 30000, 64):
            st.add(torch.as_tensor(np.ascontiguousarray(mm[i:i + 64]), device=DEV).permute(0, 3, 1, 2).float().div_(255))
        fs.append(st.features())
    out["msw_features_60000_images_512px_s"] = time.perf_counter() - t
    t = time.perf_counter(); d = st.distance(*fs); out["msw_distance_n30000_512px_ms"] = (time.perf_counter() - t) * 1000
    out["gpu"] = torch.cuda.get_device_name(0)
    json.dump(out, open(f"{results(os.path.dirname(os.path.abspath(__file__)))}/runtime.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
