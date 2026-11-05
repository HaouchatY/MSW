"""CMMD Sec. 3.3 / Appendix A / Fig. 2: are Inception embeddings Gaussian?

Mardia's skewness and kurtosis tests and the Henze-Zirkler test on the
Inception embeddings of COCO-30k (n = 30,000, d = 2048), and on its CLIP
embeddings, which the paper says fail too; plus the t-SNE of Fig. 2.

At d = 2048 the Henze-Zirkler constants (1 + 2 beta^2)^(-d/2) are ~1e-312 and
underflow double precision, so the statistic and its log-normal moments are
computed in log space (mpmath for the constants, logsumexp for the sums).

There is no MSW row: MSW fits no distribution.  Each slice is compared through
its full empirical quantile function, so the question "is the normality
assumption violated" does not arise for it.
"""
import json, math, os, sys

import mpmath as mp
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import FEAT, results                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = results(HERE)
DEV = torch.device("cuda:0")
mp.mp.dps = 80


def whiten(x):
    """z_i = S^{-1/2}(x_i - mean), S the biased sample covariance (as in Mardia)."""
    x = torch.as_tensor(x, dtype=torch.float64, device=DEV)
    xc = x - x.mean(0)
    S = xc.T @ xc / x.shape[0]
    w, v = torch.linalg.eigh(S)
    keep = w > w.max() * 1e-12
    if not bool(keep.all()):
        print(f"  covariance rank {int(keep.sum())}/{len(w)}: whitening on its range", flush=True)
    return xc @ (v[:, keep] / w[keep].sqrt()), int(keep.sum())


def mardia(z):
    n, d = z.shape
    s3, s4 = 0.0, 0.0
    for i in range(0, n, 2048):
        g = z[i:i + 2048] @ z.T                       # (b, n) inner products z_i^T z_j
        s3 += float((g ** 3).sum())
    q = (z * z).sum(1)                                # z_i^T z_i
    b1 = s3 / n ** 2
    b2 = float((q ** 2).mean())
    A = n * b1 / 6
    dfA = d * (d + 1) * (d + 2) / 6
    zA = (A - dfA) / math.sqrt(2 * dfA)               # chi^2 with huge df -> normal
    Bz = (b2 - d * (d + 2)) / math.sqrt(8 * d * (d + 2) / n)
    return {"skew_A": A, "skew_df": dfA, "skew_z": zA, "skew_p_log10": log10_sf(zA),
            "kurt_b2": b2, "kurt_z": Bz, "kurt_p_log10": log10_sf(abs(Bz)) + math.log10(2)}


def henze_zirkler(z):
    """Henze-Zirkler, computed so that it still means something at d = 2048.

    HZ = 1 + (1/n) sum_{i != j} exp(-b^2/2 D_ij) - t2 + t3: the diagonal gives
    exactly 1, and every other term is ~exp(-1000) here, so HZ - 1 and the
    null moments mu - 1, var are ~1e-300.  Double precision rounds all of it to
    0 (the reason a naive implementation divides by zero, or returns p = 0).
    We keep every term as a logarithm (logsumexp in float64 is accurate there)
    and form HZ - 1, mu - 1 and var in mpmath, then the log-normal z-score.
    """
    n, d = z.shape
    beta = (1 / math.sqrt(2)) * ((n * (2 * d + 1)) / 4) ** (1 / (d + 4))
    b2 = beta ** 2
    q = (z * z).sum(1)                                # D_ii = |z_i|^2 to the mean
    lse = []
    for i in range(0, n, 2048):
        dij = (q[i:i + 2048, None] + q[None, :] - 2 * z[i:i + 2048] @ z.T).clamp_min(0)
        e = -0.5 * b2 * dij
        rows = torch.arange(i, min(i + 2048, n), device=z.device)
        e[rows - i, rows] = -float("inf")             # drop the diagonal (it is exactly 1 each)
        lse.append(torch.logsumexp(e.flatten(), 0))
    log_off = float(torch.logsumexp(torch.stack(lse), 0)) - math.log(n)
    log_t2 = math.log(2) - (d / 2) * math.log1p(b2) + float(torch.logsumexp(-b2 / (2 * (1 + b2)) * q, 0))
    log_t3 = math.log(n) - (d / 2) * math.log1p(2 * b2)
    hz_m1 = mp.e ** log_off - mp.e ** log_t2 + mp.e ** log_t3      # HZ - 1
    B2 = mp.mpf(b2)
    a = 1 + 2 * B2
    mu_m1 = -(a ** (-mp.mpf(d) / 2)) * (1 + d * B2 / a + d * (d + 2) * B2 ** 2 / (2 * a ** 2))
    mu = 1 + mu_m1
    w = (1 + B2) * (1 + 3 * B2)
    var = 2 * (1 + 4 * B2) ** (-mp.mpf(d) / 2) \
        + 2 * a ** (-mp.mpf(d)) * (1 + 2 * d * B2 ** 2 / a ** 2 + 3 * d * (d + 2) * B2 ** 4 / (4 * a ** 4)) \
        - 4 * w ** (-mp.mpf(d) / 2) * (1 + 3 * d * B2 ** 2 / (2 * w) + d * (d + 2) * B2 ** 4 / (2 * w ** 2))
    r = var / mu ** 2
    psi = mp.sqrt(mp.log1p(r))
    pmu = mp.log1p(mu_m1) - mp.log1p(r) / 2
    zlog = (mp.log1p(hz_m1) - pmu) / psi
    return {"beta": beta, "HZ_minus_1": mp.nstr(hz_m1, 6), "null_mean_minus_1": mp.nstr(mu_m1, 6),
            "null_sd": mp.nstr(mp.sqrt(var), 6), "log_offdiag": log_off,
            "z": mp.nstr(zlog, 8), "p_log10": log10_sf(zlog)}


def log10_sf(z):
    """log10 of the standard normal upper tail, valid for astronomically large z
    (Mills ratio: sf(z) ~ phi(z)/z), where mpmath's erfc overflows."""
    z = mp.mpf(z)
    if z < 30:
        return float(mp.log10(mp.ncdf(-z)))
    return float((-z * z / 2 - mp.log(z * mp.sqrt(2 * mp.pi))) / mp.log(10))


def tsne(x, path):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    y = PCA(50, random_state=0).fit_transform(x)
    e = TSNE(2, init="pca", random_state=0, n_jobs=32).fit_transform(y)
    np.save(path, e)
    return e


if __name__ == "__main__":
    res = {}
    for name in ("inc", "clip"):
        x = np.load(f"{FEAT}/coco512_{name}.npy")
        print(f"{name}: n={x.shape[0]} d={x.shape[1]}", flush=True)
        z, rank = whiten(x)
        r = {"n": x.shape[0], "d": x.shape[1], "rank": rank}
        r.update(mardia(z)); res[name] = r
        json.dump(res, open(f"{OUT}/normality.json", "w"), indent=1, default=str)
        r.update({"hz_" + k: v for k, v in henze_zirkler(z).items()})
        res[name] = r
        json.dump(res, open(f"{OUT}/normality.json", "w"), indent=1, default=str)
        print(json.dumps(r, indent=1, default=str), flush=True)
        del z; torch.cuda.empty_cache()
    json.dump(res, open(f"{OUT}/normality.json", "w"), indent=1)
    e = tsne(np.load(f"{FEAT}/coco512_inc.npy"), f"{OUT}/tsne_inception_coco30k.npy")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(e[:, 0], e[:, 1], s=0.4, c="k", alpha=0.35, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{HERE}/figure.png", dpi=220)
    print("NORMALITY DONE", flush=True)
