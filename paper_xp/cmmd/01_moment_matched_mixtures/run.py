"""CMMD Table 2: a Gaussian against a mixture with the same mean and covariance.

Reference N(0, I) in 2-D.  Alternative: four Gaussians at (+-lam, 0), (0, +-lam)
with covariance tau^2 I, tau^2 = 1 - lam^2 / 2, so mean and covariance never
move.  lam = 0 and six values on [0.80, sqrt 2]; n = 100,000 per side; five
independent repeats.  Metrics: Frechet distance (FD) and FD_inf on the raw
points, unbiased RBF-MMD^2 (bandwidth 1, x 1000), and the MSW estimator on 16
fixed projection directions with its exact sign-flip test over all 2^16
patterns.  Exact population values of MMD and of MSW's target are included.

    python run.py            -> results/table2.json
"""
import json, math, os, sys

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from feats import fid, fid_inf                                   # noqa: E402
from paths import results                                        # noqa: E402
import msw                                                        # noqa: E402
from msw import components as C                                   # noqa: E402
from msw.features import SliceFeatures                            # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
msw.set_device(DEV)
S, BW, N, R, L, J = 1.0, 1.0, 100_000, 5, 16, 16
LAMS = [0.0] + list(np.linspace(0.80, math.sqrt(2), 6))


def sample_ref(n, rng):
    return rng.normal(0, S, (n, 2))


def sample_mix(lam, n, rng):
    tau = math.sqrt(max(S ** 2 - lam ** 2 / 2, 0.0))
    centres = np.array([[lam, 0], [0, lam], [-lam, 0], [0, -lam]])
    k = rng.integers(0, 4, n)
    return centres[k] + rng.normal(0, 1, (n, 2)) * tau


def mmd_u(x, y, bw=BW, scale=1000.0, chunk=1024):
    """Unbiased MMD^2 (CMMD paper Eq. 4) with a Gaussian RBF kernel."""
    x = torch.as_tensor(x, dtype=torch.float64, device=DEV)
    y = torch.as_tensor(y, dtype=torch.float64, device=DEV)
    g = 1.0 / (2 * bw ** 2)

    def ksum(a, b):
        return sum(float(torch.exp(-g * torch.cdist(a[i:i + chunk], b) ** 2).sum())
                   for i in range(0, a.shape[0], chunk))

    m, n = x.shape[0], y.shape[0]
    return scale * ((ksum(x, x) - m) / (m * (m - 1)) + (ksum(y, y) - n) / (n * (n - 1))
                    - 2 * ksum(x, y) / (m * n))


def msw_points(x, y):
    """MSW's estimator on points: fixed directions, block U-statistic, exact test."""
    th = np.arange(J) * math.pi / J
    dirs = np.stack([np.cos(th), np.sin(th)], 1)
    fa = SliceFeatures([torch.as_tensor(dirs @ x.T, device=DEV)], [1], x.shape[0])
    fb = SliceFeatures([torch.as_tensor(dirs @ y.T, device=DEV)], [1], y.shape[0])
    M = min(x.shape[0], y.shape[0]) // L
    psi, _, _ = C.block_rows(C.to_double(fa), C.to_double(fb), L, M, want_block_sums=False)
    g = psi.mean(2)
    g.fill_diagonal_(0.0)
    u = float(g.sum()) / (L * (L - 1))
    bits = (torch.arange(2 ** L, device=DEV)[:, None] >> torch.arange(L, device=DEV)) & 1
    sig = (1 - 2 * bits).to(torch.float64)                        # row 0 is the identity
    t = ((sig @ g) * sig).sum(1)
    p = float((t >= t[0] - 1e-15 * abs(float(t[0]))).double().mean())
    return math.copysign(math.sqrt(abs(u)), u), p


def _ek(m, a2, n, b2, s):
    v = s * s + a2 + b2
    return s * s / v * math.exp(-float(np.sum((np.asarray(m) - np.asarray(n)) ** 2)) / (2 * v))


def pop_mmd(lam, s=BW):
    """Exact MMD^2 x 1000 between the two mixtures (closed form for Gaussians)."""
    tau2 = max(S * S - lam * lam / 2, 0.0)
    c = [(lam, 0), (0, lam), (-lam, 0), (0, -lam)]
    kxx = _ek((0, 0), S * S, (0, 0), S * S, s)
    kyy = np.mean([_ek(a, tau2, b, tau2, s) for a in c for b in c])
    kxy = np.mean([_ek((0, 0), S * S, a, tau2, s) for a in c])
    return 1000 * (kxx + kyy - 2 * kxy)


def pop_msw(lam):
    """Exact target of the 16-direction estimator, from the quantile functions."""
    tau = math.sqrt(max(S * S - lam * lam / 2, 0.0))
    c = np.array([(lam, 0), (0, lam), (-lam, 0), (0, -lam)])
    u = (np.arange(200000) + 0.5) / 200000
    qref = norm.ppf(u) * S
    tot = 0.0
    for j in range(J):
        th = np.array([math.cos(j * math.pi / J), math.sin(j * math.pi / J)])
        mu = c @ th
        if tau < 1e-9:
            qm = np.sort(mu)[np.minimum((u * 4).astype(int), 3)]
        else:
            x = np.linspace(mu.min() - 8 * tau, mu.max() + 8 * tau, 400001)
            cdf = np.mean([norm.cdf((x - m) / tau) for m in mu], axis=0)
            qm = np.interp(u, cdf, x)
        tot += np.mean((qm - qref) ** 2)
    return math.sqrt(tot / J)


if __name__ == "__main__":
    rows = []
    for k, lam in enumerate(LAMS):
        vals = {"fd": [], "fd_inf": [], "mmd": [], "msw": [], "p": []}
        for r in range(R):
            rng = np.random.default_rng(5000 + 100 * k + r)
            x, y = sample_ref(N, rng), sample_mix(lam, N, rng)
            vals["fd"].append(fid(x, y)); vals["fd_inf"].append(fid_inf(x, y, min_n=5000))
            vals["mmd"].append(mmd_u(x, y))
            d, p = msw_points(x, y); vals["msw"].append(d); vals["p"].append(p)
        row = {"lam": float(lam), "pop_mmd": pop_mmd(lam), "pop_msw": pop_msw(lam)}
        for key, v in vals.items():
            row[key] = float(np.mean(v)); row[key + "_sd"] = float(np.std(v, ddof=1))
            row[key + "_reps"] = [float(t) for t in v]
        rows.append(row)
        print(f"lam={lam:.3f}  FD {row['fd']:.5f}  MMD {row['mmd']:.4f} (exact {row['pop_mmd']:.4f})  "
              f"MSW {row['msw']:.4f} (exact {row['pop_msw']:.4f})", flush=True)
    json.dump(rows, open(os.path.join(results(os.path.dirname(os.path.abspath(__file__))), "table2.json"), "w"),
              indent=1)
