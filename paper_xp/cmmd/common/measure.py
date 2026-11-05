"""One pass per image set: Inception, CLIP-336 and MSW features together.

Every set has 30,000 images, index-aligned with the COCO-30k captions.  Images
are decoded on the fly (SD latents, aMUSEd tokens, distorted tokens) or read
from a uint8 store, and each chunk feeds all three extractors, so no set is
ever written out as images.  The MSW distance to the set's reference is
computed at the end of the pass, against reference features cached once.

    python measure.py SET [SET ...]

Sets
    coco256, coco512                 the references (also cache their MSW features)
    amused_s{1..12}                  aMUSEd iteration k            ref coco256
    amused_vq_p{p}                   aMUSEd final, tokens distorted ref coco256
    coco_vq_p{p}                     COCO tokenised, distorted     ref coco256
    sd_s{95..100}                    SD after k of 100 iterations   ref coco512
"""
from __future__ import annotations

import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import CLIP336, Inception          # noqa: E402
import msw                                     # noqa: E402
from msw_stream import MSWStream               # noqa: E402

from paths import DATA, GEN, FEAT, WORK
N, CHUNK = 30000, 64
if os.environ.get("SMOKE"):                       # quick end-to-end test on a prefix
    N, FEAT = int(os.environ["SMOKE"]), f"{WORK}/feats_smoke"
DEV = torch.device("cuda:0")
msw.set_device(DEV)
os.makedirs(FEAT, exist_ok=True)


# ----------------------------------------------------------------- resizing
# Each metric is fed exactly as its reference implementation feeds it:
#   CMMD  official io_util: PIL bicubic on the uint8 RGB image, to 336
#   FID   clean-fid ("as suggested in [Parmar et al.]"): PIL bicubic on each
#         channel in float mode (no uint8 rounding), to 299
#   MSW   native resolution, no resize
from PIL import Image                                           # noqa: E402
_POOL = ThreadPoolExecutor(24)


def _one(img):
    c = np.asarray(Image.fromarray(img).resize((336, 336), Image.BICUBIC))
    f = np.stack([np.asarray(Image.fromarray(img[..., ch].astype(np.float32), mode="F")
                             .resize((299, 299), Image.BICUBIC)) for ch in range(3)], -1)
    return c, np.clip(f, 0, 255)


def metric_inputs(u8: torch.Tensor):
    """uint8 (B,3,H,W) on GPU -> (CLIP input 336, Inception input 299), float [0,1] on GPU."""
    hwc = u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    outs = list(_POOL.map(_one, hwc))
    c = torch.as_tensor(np.stack([o[0] for o in outs]), device=DEV).permute(0, 3, 1, 2).float() / 255
    f = torch.as_tensor(np.stack([o[1] for o in outs]), device=DEV).permute(0, 3, 1, 2).float() / 255
    return c, f


# ----------------------------------------------------------------- decoders
_amused = {}


def vqvae():
    if "vq" not in _amused:
        from diffusers import VQModel
        _amused["vq"] = VQModel.from_pretrained("amused/amused-256", subfolder="vqvae").to(DEV).eval()
    return _amused["vq"]


@torch.no_grad()
def decode_tokens(tok: np.ndarray, sub: int = 16) -> torch.Tensor:
    """Token grids -> uint8 images, in sub-batches (the fp32 decoder is memory-hungry)."""
    t = torch.as_tensor(np.asarray(tok).astype(np.int64), device=DEV)
    out = []
    for i in range(0, t.shape[0], sub):
        ti = t[i:i + sub]
        x = vqvae().decode(ti, force_not_quantize=True, shape=(ti.shape[0], 16, 16, 64)).sample.clamp(0, 1)
        out.append((x * 255).round().to(torch.uint8))
    return torch.cat(out)                                            # (B,3,256,256)


_sd = {}


@torch.no_grad()
def decode_latents(lat: np.ndarray) -> torch.Tensor:
    if "vae" not in _sd:
        from diffusers import AutoencoderKL
        _sd["vae"] = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",
                                                   torch_dtype=torch.float16).to(DEV).eval()
    vae = _sd["vae"]
    z = torch.as_tensor(np.asarray(lat), device=DEV, dtype=torch.float16) / vae.config.scaling_factor
    out = []
    for i in range(0, z.shape[0], 8):
        x = vae.decode(z[i:i + 8]).sample
        out.append(((x / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8))
    return torch.cat(out)                                            # (B,3,512,512)


def u8_store(path: str, s: int):
    return np.memmap(path, np.uint8, "r", shape=(N, s, s, 3))


_ithq = {}
PB = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)


@torch.no_grad()
def vqB_distort(u8_np: np.ndarray, i0: int, p: float) -> torch.Tensor:
    """8-px tokens at 512 px (VQ-Diffusion ITHQ VQGAN, the Muse-like tokenizer).

    Common random numbers keyed by the image index, so the corruption is nested
    across p exactly as in vq_tokens.py, without storing any tokens.
    """
    if "vq" not in _ithq:
        from diffusers import VQModel
        _ithq["vq"] = VQModel.from_pretrained("microsoft/vq-diffusion-ithq", subfolder="vqvae").to(DEV).eval()
    vq = _ithq["vq"]
    x = torch.as_tensor(np.ascontiguousarray(u8_np), device=DEV).permute(0, 3, 1, 2).float() / 255
    b = x.shape[0]
    tok = vq.quantize(vq.encode(2 * x - 1).latents)[2][2].reshape(b, 64, 64)
    rng = np.random.default_rng(777_000 + i0)
    u = torch.as_tensor(rng.random((b, 64, 64)), device=DEV)
    r = torch.as_tensor(rng.integers(0, 4096, (b, 64, 64)), device=DEV)
    tok = torch.where(u < p, r, tok)
    emb = vq.quantize.get_codebook_entry(tok.reshape(b, -1), shape=(b, 64, 64, 128))
    y = vq.decode(emb, force_not_quantize=True).sample
    return ((y / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)


def source(name: str):
    """(resolution, reference name, chunk iterator yielding uint8 (B,3,H,W) on GPU)."""
    if name in ("coco256", "coco512"):
        s = int(name[4:])
        mm = u8_store(f"{DATA}/{name}.u8", s)
        it = (torch.as_tensor(np.ascontiguousarray(mm[i:i + CHUNK]), device=DEV).permute(0, 3, 1, 2)
              for i in range(0, N, CHUNK))
        return s, None, it
    if name.startswith("amused_s"):
        k = int(name[len("amused_s"):])
        tok = np.memmap(f"{GEN}/amused_tok.i16", np.int16, "r", shape=(N, 12, 16, 16))
        return 256, "coco256", (decode_tokens(tok[i:i + CHUNK, k - 1]) for i in range(0, N, CHUNK))
    if name.startswith("amused_vq_p") or name.startswith("coco_vq_p"):
        tok = np.memmap(f"{GEN}/{name}.i16", np.int16, "r", shape=(N, 16, 16))
        return 256, "coco256", (decode_tokens(tok[i:i + CHUNK]) for i in range(0, N, CHUNK))
    if name.startswith("cocoB_p") or name.startswith("sdB_p"):
        p = float(name.split("_p")[1])
        path = f"{DATA}/coco512.u8" if name.startswith("cocoB") else f"{GEN}/sd_final512.u8"
        mm = u8_store(path, 512)
        return 512, "coco512", (vqB_distort(mm[i:i + 8], i, p) for i in range(0, N, 8))
    if name == "sd_s100":
        mm = u8_store(f"{GEN}/sd_final512.u8", 512)
        it = (torch.as_tensor(np.ascontiguousarray(mm[i:i + CHUNK]), device=DEV).permute(0, 3, 1, 2)
              for i in range(0, N, CHUNK))
        return 512, "coco512", it
    if name.startswith("sd_s"):
        k = int(name[len("sd_s"):])
        j = (95, 96, 97, 98, 99, 100).index(k)
        lat = np.memmap(f"{GEN}/sd_lat.f16", np.float16, "r", shape=(N, 6, 4, 64, 64))
        return 512, "coco512", (decode_latents(lat[i:i + 16, j]) for i in range(0, N, 16))
    raise ValueError(name)


# ----------------------------------------------------------------- the pass
def layout_signature(st: MSWStream) -> str:
    import hashlib
    h = hashlib.sha1()
    for la in st.est.layouts:
        h.update(la.pos.cpu().numpy().tobytes())
        h.update(la.filters.cpu().numpy().tobytes())
    return h.hexdigest()


def run(name: str, inc: Inception, clip: CLIP336):
    out_json = f"{FEAT}/{name}.json"
    if os.path.exists(out_json):
        print(f"[{name}] already measured", flush=True)
        return
    res, ref, it = source(name)
    st = MSWStream(3, res, N, store="cpu")
    fi, fc, seen, t0 = [], [], 0, time.time()
    for u8 in it:
        xc, xf = metric_inputs(u8)
        fi.append(inc(xf).cpu())
        fc.append(clip(xc).cpu())
        for j in range(0, u8.shape[0], 16):           # bounds the level-0 conv at 512 px
            st.add(u8[j:j + 16].float().div_(255.0))
        seen += u8.shape[0]
        if seen % 3200 < u8.shape[0]:
            print(f"[{name}] {seen}/{N}  {seen / (time.time() - t0):.0f} img/s", flush=True)
    assert seen == N, seen
    np.save(f"{FEAT}/{name}_inc.npy", torch.cat(fi).numpy())
    np.save(f"{FEAT}/{name}_clip.npy", torch.cat(fc).numpy())
    f_set = st.features()
    rec = {"name": name, "res": res, "ref": ref, "n": N, "sec_pass": time.time() - t0,
           "layout": layout_signature(st)}
    if ref is None:                                   # a reference: cache its MSW features
        torch.save({"mats": f_set.mats, "per": f_set.per_image, "layout": rec["layout"]},
                   f"{FEAT}/msw_ref_{name}_n{N}.pt")
    else:
        r = torch.load(f"{FEAT}/msw_ref_{ref}_n{N}.pt")
        assert r["layout"] == rec["layout"], "reference and set used different slice layouts"
        from msw.features import SliceFeatures
        f_ref = SliceFeatures(r["mats"], r["per"], N)
        t1 = time.time()
        rec["msw"] = st.distance(f_ref, f_set)
        rec["sec_msw_distance"] = time.time() - t1
        del r, f_ref
    json.dump(rec, open(out_json, "w"), indent=1)
    print(f"[{name}] done in {time.time() - t0:.0f}s  msw={rec.get('msw', {}).get('d')}", flush=True)


if __name__ == "__main__":
    inc = Inception()
    # fp16 CLIP: CMMD within 0.05% of fp32 on a blur check (0.6276 vs 0.6273),
    # embedding cosine >= 0.99958 per image; 4.5x faster (check_clip.py).
    clip = CLIP336(dtype=torch.float32 if os.environ.get("CLIP_FP32") == "1" else torch.float16)
    for name in sys.argv[1:]:
        run(name, inc, clip)
