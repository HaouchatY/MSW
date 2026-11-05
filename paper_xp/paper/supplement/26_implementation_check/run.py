"""Implementation check (tab:xval): the baselines of common/baselines.py against
the official packages, on fixed CIFAR-10 pools.

Pools: CIFAR-10 train permuted by a generator seeded 0; A = images 0..2047,
B = 2048..4095, Bn = clamp(B + 0.01 * white noise (generator seeded 1), 0, 1).
Each pool is written as 8-bit PNGs, and ours is computed on the same 8-bit
images (round(255 x) / 255), so both sides see identical pixels.
  FID       ours vs pytorch-fid (A-B, A-Bn) and torch-fidelity, clean-fid (A-B)
  KID       ours vs torch-fidelity (100 subsets of 1000, A-B and A-Bn)
  density / coverage   Naeem et al., k = 5, ours vs prdc on the same features
  DINOv2-FD ours (bicubic straight to 224) vs the reference resize (256, then
            centre crop 224): A-B and A-Bn
Needs pytorch-fid, torch-fidelity, prdc, clean-fid (each skipped if missing).

    python run.py          # a few GPU-minutes
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import baselines as B
from draws import DEV, cifar, gen


def density_coverage(real, fake, k=5):
    """Naeem et al. (2020) density and coverage, real = reference set."""
    a = torch.as_tensor(real, dtype=torch.float32, device=DEV)
    b = torch.as_tensor(fake, dtype=torch.float32, device=DEV)
    ra = torch.cdist(a, a).topk(k + 1, largest=False).values[:, -1]
    inside = torch.cdist(b, a) <= ra[None, :]                     # (fake, real)
    return float(inside.sum(1).float().mean() / k), float(inside.any(0).float().mean())


_DINO = {}


def dino_crop(x, bs=100):
    """DINOv2-large CLS with the reference preprocessing: bicubic 256, centre crop 224."""
    if "m" not in _DINO:
        from transformers import AutoModel
        _DINO["m"] = AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.float16).to(DEV).eval()
    m = _DINO["m"]
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            z = F.interpolate(x[i:i + bs].float().to(DEV), size=(256, 256), mode="bicubic",
                              align_corners=False).clamp(0, 1)[:, :, 16:240, 16:240]
            out.append(m(pixel_values=((z - mean) / std).half()).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def official(res, key, fn):
    try:
        res.update(fn())
        print(key, "done", flush=True)
    except Exception as ex:                           # package missing or failing: record why
        res[f"{key}_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
        print(key, "FAILED:", res[f"{key}_error"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    x = cifar()
    p = torch.randperm(x.shape[0], device=DEV, generator=gen(0))
    A, Bp = x[p[:a.n]], x[p[a.n:2 * a.n]]
    Bn = (Bp + 0.01 * torch.randn(Bp.shape, device=DEV, generator=gen(1))).clamp(0, 1)
    pools = {"A": A, "B": Bp, "Bn": Bn}
    dirs = {}
    for k, t in pools.items():
        d = dirs[k] = os.path.join(a.out, f"png{k}")
        os.makedirs(d, exist_ok=True)
        arr = (t.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        for i in range(arr.shape[0]):
            Image.fromarray(arr[i]).save(os.path.join(d, f"{i:05d}.png"))
    ks = min(1000, a.n)                               # KID subset size (1000 at the default n)
    q8 = lambda t: (t.clamp(0, 1) * 255).round() / 255.0

    res = {"n": a.n}
    fa, fb, fbn = B.inception(q8(A)), B.inception(q8(Bp)), B.inception(q8(Bn))
    res["ours_fid_AB"], res["ours_fid_ABn"] = B.fid(fa, fb), B.fid(fa, fbn)
    res["ours_kid_AB"] = B.kid(fa, fb, seed=0, subsets=100, subset_size=ks)
    res["ours_kid_ABn"] = B.kid(fa, fbn, seed=0, subsets=100, subset_size=ks)
    res["ours_density_AB"], res["ours_coverage_AB"] = density_coverage(fa, fb)
    assert res["ours_coverage_AB"] == B.coverage(fa, fb, k=5)
    print({k: round(v, 6) for k, v in res.items()}, flush=True)

    def ptfid():
        dev = ["--device", "cuda"] if DEV.type == "cuda" else ["--device", "cpu"]
        r = {}
        for tag, d in [("AB", dirs["B"]), ("ABn", dirs["Bn"])]:
            o = subprocess.run([sys.executable, "-m", "pytorch_fid", dirs["A"], d] + dev,
                               capture_output=True, text=True, check=True)
            r[f"ptfid_{tag}"] = float(o.stdout.strip().split()[-1])
        return r

    def tfid():
        from torch_fidelity import calculate_metrics
        r = {}
        for tag, d in [("AB", dirs["B"]), ("ABn", dirs["Bn"])]:
            m = calculate_metrics(input1=dirs["A"], input2=d, cuda=DEV.type == "cuda", fid=True, kid=True,
                                  kid_subsets=100, kid_subset_size=ks, verbose=False)
            r[f"tf_fid_{tag}"] = float(m["frechet_inception_distance"])
            r[f"tf_kid_{tag}"] = float(m["kernel_inception_distance_mean"])
        return r

    def prdc():
        from prdc import compute_prdc
        o = compute_prdc(real_features=fa, fake_features=fb, nearest_k=5)
        return {"prdc_density_AB": float(o["density"]), "prdc_coverage_AB": float(o["coverage"])}

    def cfid():
        from cleanfid import fid as cf
        return {"cleanfid_AB": float(cf.compute_fid(dirs["A"], dirs["B"], device=DEV, verbose=False))}

    for key, fn in [("pytorch-fid", ptfid), ("torch-fidelity", tfid), ("prdc", prdc), ("clean-fid", cfid)]:
        official(res, key, fn)

    for tag, emb in [("straight", B.dino), ("cropref", dino_crop)]:
        da, db, dbn = emb(A), emb(Bp), emb(Bn)
        res[f"dino_{tag}_fd_AB"], res[f"dino_{tag}_fd_ABn"] = B.fid(da, db), B.fid(da, dbn)
    json.dump(res, open(os.path.join(a.out, "xval.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
