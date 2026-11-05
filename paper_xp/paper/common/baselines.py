"""The baseline metrics of the paper, numerically identical to the original runs.

make_measure(draw, C, size) returns measure(a, b, seed) -> dict of eleven values
for two image batches a, b of shape (N, C, size, size); seed is the repetition.
Deep features see to01(x): one fixed affine map to [0, 1] (pool mean +- 4 sd of
500 reference images, draw seed 1), 1-channel images repeated to 3 channels.

  fid       Frechet distance of Inception pool3 features (pytorch_fid, fp32)
  kid       unbiased polynomial-kernel MMD^2 on the same features, mean over 20
            random subsets of min(500, N) (subset generator seeded `seed`)
  cmmd      unbiased RBF MMD^2 (sigma 10, x1000) on CLIP ViT-L/14 embeddings
  dino      Frechet distance of DINOv2-large CLS embeddings
  c2st      held-out accuracy of a logistic regression on Inception features
  coverage  Naeem et al. coverage, k = 5, a as the real set (drops under H1)
  sw        classic SW, 8192 random directions, raw images
  csw       convolution SW (Nguyen & Ho 2022), 64 stacks x 32 slices, raw images
            -- sw and csw are blocked floor-corrected per-slice values over 4
            groups combined by inverse variance, 0 in expectation under H0
  pswd      Karras et al. patch SWD on a Laplacian pyramid, on to01 images
  spec      L2 distance of the square-root mean periodograms, raw images
  ps        Portilla-Simoncelli statistics (2 scales, grayscale = channel mean),
            z-scored on a 500-image pool, L2 between the mean z-vectors

PSStrong gives the four P-S variants of the full-strength P-S* (n_scales 3 at
64 px); ps_star standardizes them by their null moments and takes the max.

Versions of the original runs: plenoptic 2.1.0, pytorch_fid 0.3.0, open_clip
with the ViT-L-14-quickgelu config, transformers >= 4.56 (AutoModel dtype=).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------- deep features
_EXT = {}


def _load(name, build):
    # lazily, without touching the global RNG (the original loaded these before
    # reseeding it, so their initialisation never reached the draws)
    if name not in _EXT:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEV.type == "cuda" else []):
            _EXT[name] = build()
    return _EXT[name]


def _inception():
    from pytorch_fid.inception import InceptionV3
    net = InceptionV3([3], resize_input=True, normalize_input=True).to(DEV).eval()

    @torch.no_grad()
    def feats(x, bs=200):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return torch.cat([net(x[i:i + bs].contiguous())[0].flatten(1)
                          for i in range(0, x.shape[0], bs)]).cpu().numpy()
    return feats


def _clip():
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    visual = model.visual.to(DEV).half().eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(x, bs=256):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        out = []
        for i in range(0, x.shape[0], bs):
            z = F.interpolate(x[i:i + bs], size=(224, 224), mode="bicubic",
                              align_corners=False).clamp(0, 1)
            out.append(F.normalize(visual(((z - mean) / std).half()).float(), dim=-1))
        return torch.cat(out).cpu().numpy()
    return embed


def _dino():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.float16).to(DEV).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(x, bs=100):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        out = []
        for i in range(0, x.shape[0], bs):
            z = F.interpolate(x[i:i + bs].float().to(DEV), size=(224, 224),
                              mode="bicubic", align_corners=False).clamp(0, 1)
            out.append(m(pixel_values=((z - mean) / std).half()).last_hidden_state[:, 0]
                       .float().cpu().numpy())
        return np.concatenate(out)
    return embed


def inception(x):
    return _load("inc", _inception)(x)


def clip(x):
    return _load("clip", _clip)(x)


def dino(x):
    return _load("dino", _dino)(x)


# ------------------------------------------------------------ feature metrics
def _sqrtm_psd(a):
    a = (a + a.T) / 2
    w, v = np.linalg.eigh(a)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def fid(f1, f2):
    """Frechet distance; computed in the sample subspace when N <= d."""
    f1, f2 = np.asarray(f1, dtype=np.float64), np.asarray(f2, dtype=np.float64)
    n1, n2, d = f1.shape[0], f2.shape[0], f1.shape[1]
    mu1, mu2 = f1.mean(0), f2.mean(0)
    diff = mu1 - mu2
    if min(n1, n2) > d:
        s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
        s1h = _sqrtm_psd(s1)
        return float(diff @ diff + np.trace(s1) + np.trace(s2)
                     - 2 * np.trace(_sqrtm_psd(s1h @ s2 @ s1h)))
    a1, a2 = f1 - mu1, f2 - mu2
    t1 = float((a1 * a1).sum()) / (n1 - 1)
    t2 = float((a2 * a2).sum()) / (n2 - 1)
    c = a1 @ a2.T
    m = (c @ c.T) / ((n1 - 1) * (n2 - 1))
    ev = np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)
    return float(diff @ diff + t1 + t2 - 2 * np.sqrt(ev).sum())


def _mmd2_u(kxx, kyy, kxy):
    m, n = kxx.shape[0], kyy.shape[0]
    return ((kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
            + (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1)) - 2 * kxy.mean())


def kid(f1, f2, seed, subsets=20, subset_size=500):
    x = torch.as_tensor(np.asarray(f1), dtype=torch.float32, device=DEV)
    y = torch.as_tensor(np.asarray(f2), dtype=torch.float32, device=DEV)
    m = min(subset_size, x.shape[0], y.shape[0])
    k = lambda u, v: (u @ v.t() / u.shape[1] + 1.0) ** 3
    g = torch.Generator(device="cpu").manual_seed(seed)
    vals = []
    for _ in range(subsets):
        xs = x[torch.randperm(x.shape[0], generator=g)[:m].to(DEV)]
        ys = y[torch.randperm(y.shape[0], generator=g)[:m].to(DEV)]
        vals.append(float(_mmd2_u(k(xs, xs), k(ys, ys), k(xs, ys))))
    return float(np.asarray(vals).mean())


def cmmd(f1, f2, sigma=10.0, scale=1000.0):
    x = torch.as_tensor(np.asarray(f1), dtype=torch.float32, device=DEV)
    y = torch.as_tensor(np.asarray(f2), dtype=torch.float32, device=DEV)
    k = lambda u, v: torch.exp(-torch.cdist(u, v).pow(2) / (2 * sigma ** 2))
    return float(scale * _mmd2_u(k(x, x), k(y, y), k(x, y)))


def coverage(f_real, f_fake, k=5):
    a = torch.as_tensor(np.asarray(f_real), dtype=torch.float32, device=DEV)
    b = torch.as_tensor(np.asarray(f_fake), dtype=torch.float32, device=DEV)
    ra = torch.cdist(a, a).topk(k + 1, largest=False).values[:, -1]
    return (torch.cdist(b, a) <= ra[None, :]).any(0).float().mean().item()


def c2st(f1, f2, seed, epochs=300, lr=1e-2, wd=1e-2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.as_tensor(np.concatenate([f1, f2]), dtype=torch.float32, device=DEV)
    y = torch.cat([torch.zeros(len(f1)), torch.ones(len(f2))]).to(DEV)
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    perm = torch.randperm(x.shape[0], generator=g).to(DEV)
    x, y = x[perm], y[perm]
    cut = x.shape[0] // 2
    xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
    w = torch.zeros(x.shape[1], 1, device=DEV, requires_grad=True)
    b = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    with torch.enable_grad():                 # callers may run under no_grad
        for _ in range(epochs):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits((xtr @ w).squeeze(1) + b, ytr).backward()
            opt.step()
    with torch.no_grad():
        return ((((xte @ w).squeeze(1) + b) > 0).float() == yte).float().mean().item()


# ------------------------------------------------------------ sliced (SW, CSW)
def _w2(a, b):
    """Column-wise W_2^2 between 1-D samples (M, K); shared quantile grid if sizes differ."""
    if a.shape[0] == b.shape[0]:
        return (a.sort(dim=0).values - b.sort(dim=0).values).abs_().pow_(2).mean(dim=0)

    def quant(s, q):
        m = s.shape[0]
        pos = (torch.arange(m, device=s.device, dtype=s.dtype) + 0.5) / m
        idx = torch.searchsorted(pos, q.to(pos.dtype)).clamp_(1, m - 1)
        lo, hi = idx - 1, idx
        wgt = ((q - pos[lo]) / (pos[hi] - pos[lo]).clamp_min(1e-12)).unsqueeze(1)
        return s[lo] + wgt * (s[hi] - s[lo])
    n_q = min(min(a.shape[0], b.shape[0]), 200_000)
    q = (torch.arange(n_q, device=a.device, dtype=a.dtype) + 0.5) / n_q
    return (quant(a.sort(dim=0).values, q) - quant(b.sort(dim=0).values, q)).abs_().pow_(2).mean(dim=0)


def invvar_statistic(fa, fb, groups=4):
    """fa, fb: (S, N) one scalar per slice and image.  Group j: cross minus within
    W_2^2 on disjoint image blocks; per-slice means over groups combined with
    weights 1 / se^2."""
    na, nb = fa.shape[1] // (2 * groups), fb.shape[1] // (2 * groups)
    rows = []
    for j in range(groups):
        a1, a2 = fa[:, 2 * j * na:(2 * j + 1) * na].t(), fa[:, (2 * j + 1) * na:(2 * j + 2) * na].t()
        b1, b2 = fb[:, 2 * j * nb:(2 * j + 1) * nb].t(), fb[:, (2 * j + 1) * nb:(2 * j + 2) * nb].t()
        rows.append(0.5 * (_w2(a1, b1) + _w2(a2, b2)) - 0.5 * (_w2(a1, a2) + _w2(b1, b2)))
    tg = torch.stack(rows)
    se = (tg.std(0) / groups ** 0.5).clamp_min(1e-12)
    w = 1.0 / se ** 2
    return float((w * tg.mean(0)).sum() / w.sum())


class SW:
    def __init__(self, C, size, n_proj=8192, seed=0, chunk=2048):
        g = torch.Generator(device=DEV).manual_seed(seed)
        th = torch.randn(n_proj, C * size * size, device=DEV, generator=g)
        th -= th.mean(1, keepdim=True)
        self.dirs, self.chunk = th / th.norm(dim=1, keepdim=True), chunk

    @torch.no_grad()
    def features(self, x):
        flat = x.reshape(x.shape[0], -1)
        return torch.cat([flat @ self.dirs[i:i + self.chunk].t()
                          for i in range(0, self.dirs.shape[0], self.chunk)], dim=1).t().contiguous()

    def statistic(self, a, b):
        return invvar_statistic(self.features(a), self.features(b))


class CSW:
    """n_slicers stacks of unit-norm stride-2 convolutions, each image -> L scalars.

    Seeds torch's GLOBAL generator (torch.manual_seed) and builds nn.Conv2d
    layers on the CPU before redrawing their weights on DEV, exactly as the
    original did, so the global RNG state it leaves behind is the same."""

    def __init__(self, C, size, n_slicers=64, L=32, seed=0, bottom_width=8, image_batch=4096):
        torch.manual_seed(seed)
        self.stacks = []
        for _ in range(n_slicers):
            convs, in_ch, s, first = [], C, size, True
            while s > bottom_width:
                convs.append(nn.Conv2d(in_ch, L, 4, 2, 1, bias=False, groups=1 if first else L))
                in_ch, first, s = L, False, (s + 2 - 4) // 2 + 1
            convs.append(nn.Conv2d(in_ch, L, s, 1, 0, bias=False, groups=1 if first else L))
            stack = []
            for c in convs:
                c.to(DEV)
            for c in convs:
                with torch.no_grad():
                    w = torch.randn_like(c.weight)
                    stack.append((w / w.flatten(1).norm(dim=1).view(-1, 1, 1, 1),
                                  c.stride, c.padding, c.groups))
            self.stacks.append(stack)
        self.image_batch = image_batch

    @torch.no_grad()
    def _stack(self, stack, x):
        for w, st, pad, gr in stack:
            x = F.conv2d(x, w, None, st, pad, 1, gr)
        return x.reshape(x.shape[0], -1)

    @torch.no_grad()
    def features(self, x):
        cols = [torch.cat([self._stack(st, x[i:i + self.image_batch])
                           for i in range(0, x.shape[0], self.image_batch)]) for st in self.stacks]
        return torch.cat(cols, dim=1).t().contiguous()

    def statistic(self, a, b):
        return invvar_statistic(self.features(a), self.features(b))


# --------------------------------------------------------- Karras SWD, spectrum
_K2 = (lambda g: g[:, None] * g[None, :])(torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0)


def _blur(x):
    C = x.shape[1]
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), _K2.to(x.device).expand(C, 1, 5, 5), groups=C)


def _laplacian_pyramid(x, n_levels):
    pyr = [x]
    for _ in range(n_levels - 1):
        d = _blur(pyr[-1])[:, :, ::2, ::2]
        up = torch.zeros(*d.shape[:2], d.shape[2] * 2, d.shape[3] * 2, device=d.device, dtype=d.dtype)
        up[:, :, ::2, ::2] = d * 4.0
        pyr[-1] = pyr[-1] - _blur(up)
        pyr.append(d)
    return pyr


def _patches(level, n, g, k=7):
    B, C, H, W = level.shape
    ys = torch.randint(0, H - k + 1, (B, n), device=level.device, generator=g)
    xs = torch.randint(0, W - k + 1, (B, n), device=level.device, generator=g)
    unf = F.unfold(level, k)
    lin = ys * (W - k + 1) + xs
    return torch.gather(unf, 2, lin[:, None, :].expand(-1, unf.shape[1], -1)) \
        .permute(0, 2, 1).reshape(B * n, C, k, k)


def _finalize(desc):
    m = desc.mean(dim=(0, 2, 3), keepdim=True)
    s = desc.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    return ((desc - m) / s).reshape(desc.shape[0], -1)


def _swd_pair(da, db, g, repeats=4, dirs_per_repeat=128):
    n = min(da.shape[0], db.shape[0])
    da, db = da[:n], db[:n]
    dists = []
    for _ in range(repeats):
        dirs = torch.randn(da.shape[1], dirs_per_repeat, device=da.device, generator=g)
        dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-12)
        dists.append(((da @ dirs).sort(dim=0).values - (db @ dirs).sort(dim=0).values).abs().mean())
    return float(torch.stack(dists).mean())


def karras_swd(a, b, seed, n_per_img=128, cap=16384):
    """Mean over Laplacian levels (down to 16 px) of patch sliced W1."""
    n_levels = int(math.log2(a.shape[-1])) - 3
    g = torch.Generator(device=a.device).manual_seed(90_000 + seed)
    per_level = []
    for la, lb in zip(_laplacian_pyramid(a, n_levels), _laplacian_pyramid(b, n_levels)):
        da, db = _patches(la, n_per_img, g), _patches(lb, n_per_img, g)
        if da.shape[0] > cap:
            da = da[torch.randperm(da.shape[0], device=a.device, generator=g)[:cap]]
            db = db[torch.randperm(db.shape[0], device=a.device, generator=g)[:cap]]
        per_level.append(_swd_pair(_finalize(da), _finalize(db), g))
    return float(np.mean(per_level))


def spec_dist(a, b):
    S = lambda x: (torch.fft.fft2(x.float()).abs() ** 2 / (x.shape[-1] * x.shape[-2])).mean(dim=(0, 1))
    return float(((S(a).sqrt() - S(b).sqrt()) ** 2).sum().sqrt())


# ------------------------------------------------------------ Portilla-Simoncelli
_PS = {}


def ps_stats(x, n_scales=2, bs=256):
    """P-S statistic vectors (plenoptic), grayscale = channel mean: (N, D) on DEV."""
    if x.shape[1] == 3:
        x = x.mean(1, keepdim=True)
    key = (x.shape[-1], n_scales)
    if key not in _PS:
        import plenoptic as po
        _PS[key] = po.models.PortillaSimoncelli(key[:1] * 2, n_scales=n_scales, n_orientations=4,
                                                spatial_corr_width=7).to(DEV)
    with torch.no_grad():
        return torch.cat([_PS[key](x[i:i + bs].float().to(DEV)).squeeze(1)
                          for i in range(0, x.shape[0], bs)])


def _pool_z(v):
    return v.mean(0), v.std(0).clamp_min(1e-3 * float(v.std(0).median()) + 1e-12)


class PSDist:
    """The `ps` value: L2 between mean pool-z-scored statistic vectors (2 scales)."""

    def __init__(self, draw):
        self.m, self.s = _pool_z(ps_stats(draw(500, torch.Generator(device=DEV).manual_seed(1))).cpu())

    def __call__(self, a, b):
        za = (ps_stats(a).cpu() - self.m) / self.s
        zb = (ps_stats(b).cpu() - self.m) / self.s
        return float((za.mean(0) - zb.mean(0)).norm())


class PSStrong:
    """The four P-S* variants; n_scales 3 at 64 px, 2 at 32 px.

    ps_l2   as PSDist (identical to it at 32 px)
    ps_chi  mean_d (dmean_d / se_d)^2, se_d the two-sample standard error
    ps_max  max_d |dmean_d / se_d|
    ps_ivs  |sum_d w_d dmean_d| / sum_d w_d, w_d = 1 / se_d^2
    """

    VARIANTS = ("ps_l2", "ps_chi", "ps_max", "ps_ivs")

    def __init__(self, draw, size):
        self.ns = 3 if size >= 64 else 2
        self.m, self.s = _pool_z(self._stats(draw(500, torch.Generator(device=DEV).manual_seed(1))))

    def _stats(self, x):
        return ps_stats(x, self.ns, bs=128)

    def __call__(self, a, b):
        va, vb = self._stats(a), self._stats(b)
        za, zb = (va - self.m) / self.s, (vb - self.m) / self.s
        dm = va.mean(0) - vb.mean(0)
        se = (va.var(0) / va.shape[0] + vb.var(0) / vb.shape[0]).sqrt()
        se = se.clamp_min(1e-3 * float(se.median()) + 1e-12)
        z, w = dm / se, 1.0 / se ** 2
        return {"ps_l2": float((za.mean(0) - zb.mean(0)).norm()),
                "ps_chi": float((z ** 2).mean()), "ps_max": float(z.abs().max()),
                "ps_ivs": float(((w * dm).sum() / w.sum()).abs())}


def ps_star(null_rows, rows):
    """P-S* value of each row: max over the four variants of (v - mean_null) / sd_null
    (null moments over null_rows, sd with ddof=1).  Detection as for every baseline:
    above the 95th percentile of ps_star(null_rows, null_rows)."""
    A0 = np.array([[r[v] for v in PSStrong.VARIANTS] for r in null_rows])
    A = np.array([[r[v] for v in PSStrong.VARIANTS] for r in rows])
    mu, sd = A0.mean(0), np.maximum(A0.std(0, ddof=1), 1e-30)
    return ((A - mu) / sd).max(1)


# -------------------------------------------------------------------- measure
_PORT_RNG = set()


def _replay_port_rng(C, size):
    # The original also built its own estimator here (once per (C, size) per
    # process), whose DCT bank subsampling drew one global torch.randperm per
    # pyramid level; replayed so the global RNG state matches downstream.
    if (C, size) in _PORT_RNG:
        return
    _PORT_RNG.add((C, size))
    for _ in range(4 if size == 64 else 3):
        torch.randperm(24 * C, device=DEV)


def make_measure(draw, C, size):
    sw = SW(C, size, 8192, seed=0)
    csw = CSW(C, size, 64, 32, seed=0)
    _replay_port_rng(C, size)
    psd = PSDist(draw)
    pool = draw(500, torch.Generator(device=DEV).manual_seed(1))
    mu, sd = float(pool.mean()), float(pool.std())
    to01 = lambda t: ((t - (mu - 4 * sd)) / (8 * sd)).clamp(0, 1)

    def measure(a, b, seed):
        a3 = to01(a).expand(-1, 3, -1, -1) if C == 1 else to01(a)
        b3 = to01(b).expand(-1, 3, -1, -1) if C == 1 else to01(b)
        fa, fb = inception(a3), inception(b3)
        return {"fid": fid(fa, fb),
                "kid": kid(fa, fb, seed, subsets=20, subset_size=min(500, a.shape[0])),
                "cmmd": cmmd(clip(a3), clip(b3)),
                "dino": fid(dino(a3), dino(b3)),
                "c2st": c2st(fa, fb, seed), "coverage": coverage(fa, fb, k=5),
                "sw": sw.statistic(a, b), "csw": csw.statistic(a, b),
                "pswd": karras_swd(a3, b3, seed), "spec": spec_dist(a, b),
                "ps": psd(a, b)}
    return measure
