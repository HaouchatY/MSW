"""VQGAN latent-space noise (CMMD Sec. 6.2, Figs. 1, 5 and 7).

"For each image, we obtain VQGAN tokens, replace them with random tokens with
probability p, and reconstruct the image with the VQGAN detokenizer."  Two
series, as in the paper:

  amused_vq_p*   the generator's own final tokens, distorted       (Fig. 7)
  coco_vq_p*     COCO-30k tokenised by the same VQGAN, distorted   (Fig. 1),
                 compared against the clean COCO images

The draws are common random numbers: one uniform u and one replacement token r
per token position, shared by every p, and token t becomes r where u < p.  The
distortion is therefore nested -- every token replaced at p is also replaced at
p' > p -- so the curves differ between levels only by the extra distortion,
not by fresh randomness.  The paper does not say how it drew; nesting is the
lower-variance choice and changes no expectation.

At p = 0 the aMUSEd series is the generated image itself (amused_s12), and the
COCO series is the plain VQGAN reconstruction.
"""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusers import VQModel

from paths import DATA, GEN
N, V = 30000, 8192
P = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
DEV = torch.device("cuda:0")


@torch.no_grad()
def encode_coco():
    path = f"{GEN}/coco_tok.i16"
    if os.path.exists(path):
        return np.memmap(path, np.int16, "r", shape=(N, 16, 16))
    vq = VQModel.from_pretrained("amused/amused-256", subfolder="vqvae").to(DEV).eval()
    img = np.memmap(f"{DATA}/coco256.u8", np.uint8, "r", shape=(N, 256, 256, 3))
    out = np.memmap(path + ".tmp", np.int16, "w+", shape=(N, 16, 16))
    for i in range(0, N, 32):
        x = torch.as_tensor(np.ascontiguousarray(img[i:i + 32]), device=DEV).permute(0, 3, 1, 2).float() / 255
        lat = vq.encode(x).latents
        idx = vq.quantize(lat)[2][2].reshape(x.shape[0], 16, 16)
        out[i:i + 32] = idx.cpu().numpy().astype(np.int16)
        if i % 6400 == 0:
            print(f"encode {i}/{N}", flush=True)
    out.flush(); del out
    os.replace(path + ".tmp", path)
    return np.memmap(path, np.int16, "r", shape=(N, 16, 16))


def distort(tok: np.ndarray, prefix: str, seed: int, skip_zero: bool):
    rng = np.random.default_rng(seed)
    u = rng.random(tok.shape)
    r = rng.integers(0, V, tok.shape).astype(np.int16)
    for p in P:
        if p == 0 and skip_zero:
            continue
        t = np.where(u < p, r, tok).astype(np.int16)
        frac = float((t != tok).mean())
        t.tofile(f"{GEN}/{prefix}_vq_p{p:.2f}.i16")
        print(f"{prefix} p={p:.2f}: {frac:.4f} of tokens changed", flush=True)


if __name__ == "__main__":
    coco = np.asarray(encode_coco())
    distort(coco, "coco", seed=20240101, skip_zero=False)
    am = np.memmap(f"{GEN}/amused_tok.i16", np.int16, "r", shape=(N, 12, 16, 16))
    assert np.load(f"{GEN}/amused_done.npy").all(), "aMUSEd generation not finished"
    distort(np.asarray(am[:, -1]), "amused", seed=20240102, skip_zero=True)
    print("VQ TOKENS DONE", flush=True)
