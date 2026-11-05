"""Feature extractors and metrics for the CMMD reproduction.

FID      pytorch-fid Inception-v3 pool3 (the TF FID weights); images resized to
         299 with antialiased bicubic, as the CMMD paper prescribes ("high-
         quality bicubic resizing with anti-aliasing as suggested in [Parmar]").
CMMD     the official google-research configuration: OpenAI CLIP ViT-L/14@336px,
         bicubic resize to 336, CLIP normalisation, L2-normalised projected
         embedding; Gaussian RBF with sigma = 10, V-statistic (diagonal
         included, as the released code computes it), times 1000.
FID_inf  Chong & Forsyth: FID at 15 subset sizes from 5000 to N, linear fit in
         1/N, intercept.
MSW      msw.distance (msw_stream.py, verified bit-identical), at native
         resolution: no resize.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def resize(x: torch.Tensor, s: int) -> torch.Tensor:
    if x.shape[-1] == s and x.shape[-2] == s:
        return x
    return F.interpolate(x, size=(s, s), mode="bicubic", antialias=True,
                         align_corners=False).clamp_(0, 1)


class Inception:
    def __init__(self, device="cuda"):
        from pytorch_fid.inception import InceptionV3
        self.net = InceptionV3([3], resize_input=False, normalize_input=True).to(device).eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:          # x in [0,1], fp32
        return self.net(resize(x, 299).contiguous())[0].flatten(1).float()


class CLIP336:
    NAME = "openai/clip-vit-large-patch14-336"

    def __init__(self, device="cuda", dtype=torch.float32):
        from transformers import CLIPVisionModelWithProjection
        self.dtype = dtype
        self.net = CLIPVisionModelWithProjection.from_pretrained(self.NAME, torch_dtype=dtype).to(device).eval()
        self.mean, self.std = CLIP_MEAN.to(device), CLIP_STD.to(device)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        z = (resize(x, 336) - self.mean) / self.std
        e = self.net(pixel_values=z.to(self.dtype)).image_embeds.float()
        return e / e.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------- metrics
def _sqrtm_psd_eigs(a: torch.Tensor) -> torch.Tensor:
    w, v = torch.linalg.eigh((a + a.T) / 2)
    return (v * w.clamp_min(0).sqrt()) @ v.T


def fid(f1: np.ndarray, f2: np.ndarray, device="cuda") -> float:
    """Frechet distance between Gaussian fits (float64 on the GPU).

    tr((S1 S2)^{1/2}) is computed as tr((S1^{1/2} S2 S1^{1/2})^{1/2}), symmetric
    PSD throughout, so the result is real by construction.
    """
    x = torch.as_tensor(f1, dtype=torch.float64, device=device)
    y = torch.as_tensor(f2, dtype=torch.float64, device=device)
    m1, m2 = x.mean(0), y.mean(0)
    s1 = torch.cov(x.T)
    s2 = torch.cov(y.T)
    r1 = _sqrtm_psd_eigs(s1)
    cm = torch.linalg.eigvalsh(r1 @ s2 @ r1).clamp_min(0).sqrt().sum()
    return float(((m1 - m2) ** 2).sum() + torch.trace(s1) + torch.trace(s2) - 2 * cm)


def fid_inf(f_real: np.ndarray, f_fake: np.ndarray, num_points: int = 15,
            min_n: int = 5000, seed: int = 0, device="cuda") -> float:
    """Chong & Forsyth's FID_infinity, with their protocol."""
    rng = np.random.default_rng(seed)
    ns = np.linspace(min_n, f_fake.shape[0], num_points).astype(int)
    vals = []
    for n in ns:
        sel = rng.permutation(f_fake.shape[0])[:n]
        vals.append(fid(f_real, f_fake[sel], device))
    A = np.stack([np.ones(len(ns)), 1.0 / ns], 1)
    coef, *_ = np.linalg.lstsq(A, np.array(vals), rcond=None)
    return float(coef[0])


def cmmd(f1: np.ndarray, f2: np.ndarray, sigma: float = 10.0, scale: float = 1000.0,
         device="cuda", chunk: int = 8192) -> float:
    """The released CMMD estimator: E k(x,x') + E k(y,y') - 2 E k(x,y), all means
    over every pair including the diagonal, gamma = 1 / (2 sigma^2), x 1000."""
    x = torch.as_tensor(f1, dtype=torch.float64, device=device)
    y = torch.as_tensor(f2, dtype=torch.float64, device=device)
    gamma = 1.0 / (2 * sigma ** 2)

    def kmean(a, b):
        an, bn = (a * a).sum(1), (b * b).sum(1)
        tot = 0.0
        for i in range(0, a.shape[0], chunk):
            d = an[i:i + chunk, None] + bn[None, :] - 2 * a[i:i + chunk] @ b.T
            tot += float(torch.exp(-gamma * d).sum())
        return tot / (a.shape[0] * b.shape[0])

    return scale * (kmean(x, x) + kmean(y, y) - 2 * kmean(x, y))
