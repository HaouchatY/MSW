"""Ordering late SD iterations at small N: which metric is stable?

Stable Diffusion 1.4, iterations 95..100 of 100 (CMMD Fig. 6 benchmark).  From a
pool of the first P = 4096 COCO-30k captions we draw random subsets of N
captions; on each subset every metric scores the six iteration sets against the
COCO images of the same captions, on identical images.  The truth improves with
the iteration, so per subset we record Kendall tau between the iteration and
-metric (+1 = the six sets perfectly ordered) and whether metric(95) > metric(100).

Iterations 95-99 of the pool are decoded once and stored (uint8); iteration 100
is the stored final image.  FID: clean-fid inputs; CMMD: official
configuration; MSW: msw.distance at native 512 px (msw_stream.py).
"""
import itertools, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA, GEN, FEAT, results                         # noqa: E402
import measure as M                                               # noqa: E402
from feats import CLIP336, Inception, cmmd, fid
from msw_stream import MSWStream

N30, P = 30000, 4096
ITERS = (95, 96, 97, 98, 99, 100)
GRID = {8: 100, 16: 100, 32: 100, 64: 100, 128: 100, 256: 50, 512: 30}
OUT = f"{results(os.path.dirname(os.path.abspath(__file__)))}/smalln_sd.json"
POOL = f"{GEN}/sd_pool_iters.u8"
DEV = M.DEV


def build_pool():
    """Decode iterations 95..99 for the pool once; iteration 100 is already stored."""
    shape = (5, P, 512, 512, 3)
    done_flag = POOL + ".done"
    if os.path.exists(done_flag):
        return np.memmap(POOL, np.uint8, "r", shape=shape)
    lat = np.memmap(f"{GEN}/sd_lat.f16", np.float16, "r", shape=(N30, 6, 4, 64, 64))
    out = np.memmap(POOL, np.uint8, "w+", shape=shape)
    t0 = time.time()
    for j in range(5):
        for i in range(0, P, 32):
            out[j, i:i + 32] = M.decode_latents(lat[i:i + 32, j]).permute(0, 2, 3, 1).cpu().numpy()
        out.flush()
        print(f"pool: iteration {ITERS[j]} decoded ({time.time() - t0:.0f}s)", flush=True)
    open(done_flag, "w").write("ok")
    return np.memmap(POOL, np.uint8, "r", shape=shape)


def set_images(pool, final, k):
    return final if k == 100 else pool[ITERS.index(k)]


def pool_features(pool, final, coco):
    path = f"{FEAT}/smalln_pool_feats.npz"
    if os.path.exists(path):
        z = np.load(path); return {k: z[k] for k in z.files}
    inc, clip = Inception(), CLIP336(dtype=torch.float16)
    F = {}
    for k in ITERS:
        imgs = set_images(pool, final, k)
        fi, fc = [], []
        for i in range(0, P, 64):
            u8 = torch.as_tensor(np.ascontiguousarray(imgs[i:i + 64]), device=DEV).permute(0, 3, 1, 2)
            xc, xf = M.metric_inputs(u8); fi.append(inc(xf).cpu()); fc.append(clip(xc).cpu())
        F[f"inc{k}"], F[f"clip{k}"] = torch.cat(fi).numpy(), torch.cat(fc).numpy()
        print(f"features: iteration {k}", flush=True)
    F["inc_coco"] = np.load(f"{FEAT}/coco512_inc.npy")[:P]
    F["clip_coco"] = np.load(f"{FEAT}/coco512_clip.npy")[:P]
    np.savez(path, **F)
    return F


def kendall(v):
    """tau between the iteration order and -v (+1: strictly decreasing values)."""
    c = d = 0
    for i, j in itertools.combinations(range(len(v)), 2):
        s = np.sign(v[i] - v[j]); c += s > 0; d += s < 0
    return (c - d) / max(c + d, 1)


def msw_values(idx, pool, final, coco):
    n = len(idx)
    st = MSWStream(3, 512, n, store="cpu")

    def feats(imgs):
        for i in range(0, n, 16):
            st.add(torch.as_tensor(imgs[idx[i:i + 16]], device=DEV).permute(0, 3, 1, 2).float().div_(255))
        return st.features()

    fr = feats(coco)
    return [st.distance(fr, feats(set_images(pool, final, k)))["d"] for k in ITERS]


if __name__ == "__main__":
    final = np.memmap(f"{GEN}/sd_final512.u8", np.uint8, "r", shape=(N30, 512, 512, 3))
    coco = np.memmap(f"{DATA}/coco512.u8", np.uint8, "r", shape=(N30, 512, 512, 3))
    pool = build_pool()
    F = pool_features(pool, final, coco)
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    have = {(r["N"], r["rep"]) for r in res}
    for N, R in GRID.items():
        rng = np.random.default_rng(1000 + N)
        for rep in range(R):
            idx = np.sort(rng.permutation(P)[:N])          # drawn even if cached: keeps the stream
            if (N, rep) in have:
                continue
            t0 = time.time()
            fidv = [fid(F["inc_coco"][idx], F[f"inc{k}"][idx]) for k in ITERS]
            cmv = [cmmd(F["clip_coco"][idx], F[f"clip{k}"][idx]) for k in ITERS]
            mv = msw_values(idx, pool, final, coco)
            row = {"N": N, "rep": rep, "fid": fidv, "cmmd": cmv, "msw": mv}
            for m in ("fid", "cmmd", "msw"):
                row[f"tau_{m}"] = kendall(row[m])
                row[f"first_last_{m}"] = bool(row[m][0] > row[m][-1])
            res.append(row)
            if rep % 10 == 9 or rep == R - 1:
                json.dump(res, open(OUT, "w"))
                rows = [r for r in res if r["N"] == N]
                print(f"N={N:4d} rep {rep + 1}/{R}  mean tau  FID {np.mean([r['tau_fid'] for r in rows]):+.3f}  "
                      f"CMMD {np.mean([r['tau_cmmd'] for r in rows]):+.3f}  MSW {np.mean([r['tau_msw'] for r in rows]):+.3f}"
                      f"   ({time.time() - t0:.1f}s/rep)", flush=True)
    json.dump(res, open(OUT, "w"))
    print("SMALL-N DONE", flush=True)
