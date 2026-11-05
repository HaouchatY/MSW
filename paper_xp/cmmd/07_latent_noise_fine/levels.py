"""Where MSW's dip comes from: its squared distance split by pyramid level.

COCO images at 512 px through the 8-pixel VQGAN (VQ-Diffusion ITHQ, 64x64
tokens), a fraction p of the tokens replaced at random (nested draws, seed
424242), against the clean images; first 2,000 COCO-30k images.  For each p,
u = sum_q lambda_q U_q is split exactly into one term per pyramid level.

    python levels.py         -> results/levels.json   {p: [u_level0, ..., u_level5]}
"""
import json, os, sys

import numpy as np
import torch
from diffusers import VQModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA, results                                   # noqa: E402
import msw                                                         # noqa: E402
from msw import components as C                                    # noqa: E402
from msw.features import SliceFeatures                             # noqa: E402
from msw.pairgroup import block_plan                               # noqa: E402
from msw.weights import slice_weights                              # noqa: E402
from msw_stream import MSWStream                                   # noqa: E402

DEV = torch.device("cuda:0")
msw.set_device(DEV)
NP, GRID, V = 2000, 64, 4096
vq = VQModel.from_pretrained("microsoft/vq-diffusion-ithq", subfolder="vqvae").to(DEV).eval()


@torch.no_grad()
def encode(u8):
    return vq.quantize(vq.encode(2 * u8.float() / 255 - 1).latents)[2][2].reshape(u8.shape[0], GRID, GRID)


@torch.no_grad()
def decode(idx):
    b = idx.shape[0]
    emb = vq.quantize.get_codebook_entry(idx.reshape(b, -1), shape=(b, GRID, GRID, 128))
    x = vq.decode(emb, force_not_quantize=True).sample / 2 + 0.5
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def msw_features(chunks):
    st = MSWStream(3, 512, NP, store="cpu")
    for u8 in chunks:
        for j in range(0, u8.shape[0], 16):
            st.add(u8[j:j + 16].float().div_(255))
    return st.features(), st


def per_level(st, fa, fb):
    L, M = block_plan(NP, NP, 16)
    w = np.array(slice_weights(3, st.est.level_sizes, scheme="P1s"))
    out, s0 = [], 0
    for ma, mb, q in zip(fa.mats, fb.mats, fa.per_image):
        sa = C.to_double(SliceFeatures([ma.to(DEV)], [q], NP))
        sb = C.to_double(SliceFeatures([mb.to(DEV)], [q], NP))
        psi, _, _ = C.block_rows(sa, sb, L, M, want_block_sums=False)
        S = psi.shape[2]
        g = (psi * torch.as_tensor(w[s0:s0 + S], device=psi.device, dtype=psi.dtype)).sum(2)
        g.fill_diagonal_(0.0)
        out.append(float(g.sum()) / (L * (L - 1))); s0 += S
    return out


if __name__ == "__main__":
    src = np.memmap(f"{DATA}/coco512.u8", np.uint8, "r", shape=(30000, 512, 512, 3))
    clean = lambda: (torch.as_tensor(np.ascontiguousarray(src[i:i + 16]), device=DEV).permute(0, 3, 1, 2)
                     for i in range(0, NP, 16))
    ref, _ = msw_features(clean())
    tok = torch.cat([encode(u) for u in clean()]).cpu().numpy()
    rng = np.random.default_rng(424242)
    u = rng.random(tok.shape); r = rng.integers(0, V, tok.shape)
    res = {}
    for p in (0.0, 0.01, 0.02, 0.05):
        tp = np.where(u < p, r, tok)
        f, st = msw_features(decode(torch.as_tensor(tp[i:i + 8], device=DEV)) for i in range(0, NP, 8))
        res[str(p)] = per_level(st, ref, f)
        print(f"p={p:.2f}  u per level (0 = finest): " + "  ".join(f"{v:+.2e}" for v in res[str(p)]), flush=True)
    json.dump(res, open(f"{results(os.path.dirname(os.path.abspath(__file__)))}/levels.json", "w"), indent=1)
