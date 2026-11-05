"""Generic two-sample discrepancies on MSW's own pooled slices, given the same
blocks, pair group and calibration as MSW's transport rows (msw internals).

For each slice family (per-channel DCT bank; opponent chroma bank on colour
data) both datasets are cut into the same L matched blocks of M images.  Per
slice s and block i the pooled 1-D samples a_i, b_i are subsampled to K points
with one index set shared by both sides, so the pair swap a_i <-> b_i stays
measure preserving and the test stays exact.  Each competitor is a Gram matrix
over blocks, Psi_ij = <phi(b_i) - phi(a_i), phi(b_j) - phi(a_j)>, with

  w2      phi = the quantile function (msw's own transport row, subsampled)
  energy  phi = the CDF, L2(dt) inner product, times 2 (1-D energy distance)
  mmd     phi = the Gaussian-kernel mean embedding, median-heuristic bandwidth
          from the sorted pooled 2LK values (orbit-invariant), by Gauss-Hermite
          quadrature of the kernel's spectral measure (R = 128 nodes)

handed to msw.components.transport_components (the same iv, ml, ivp components
as MSW's transport rows).  KS is a sup: u_s(sigma) = max_t |sum_i sigma_i
D_i(t)| / L, whose identity row is the pooled two-sample KS statistic of slice
s, standardized on the orbit, with components mean over slices, max over level
means, max over slices.  Every column set is combined by max-z on the orbit.
"""
import math

import numpy as np
import torch

from msw import components as C
from msw.pairgroup import block_plan, patterns_for
from msw.portfolio import Portfolio, core_columns, groups

DT = torch.float64
KS_GRID = 1024
RIVALS = ("ks", "mmd", "energy", "w2")


def sub_blocks(mat, q, L, M, Ksub, seed):
    """(S, L, K) sorted block samples; the index set is shared by both sides."""
    S, Kf = mat.shape[0], M * q
    x = mat[:, :L * M * q].view(S, L, Kf)
    if Ksub is None or Ksub >= Kf:
        return x.to(DT).sort(-1).values
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.stack([torch.randperm(Kf, generator=g)[:Ksub].sort().values
                       for _ in range(L)]).to(mat.device)
    return torch.gather(x, 2, idx.unsqueeze(0).expand(S, -1, -1)).to(DT).sort(-1).values


def cdf_diff(A, B):
    """F_b - F_a on the pooled grid (S, L, G) and the grid steps (S, G)."""
    S, L, K = A.shape
    gr = torch.cat([A.reshape(S, L * K), B.reshape(S, L * K)], 1).sort(-1).values
    G = gr.shape[1]
    t = gr.unsqueeze(1).expand(S, L, G).contiguous()
    fa = torch.searchsorted(A.contiguous(), t, right=True).to(DT) / K
    fb = torch.searchsorted(B.contiguous(), t, right=True).to(DT) / K
    w = torch.zeros_like(gr)
    w[:, :-1] = gr[:, 1:] - gr[:, :-1]
    return fb - fa, w


def psi_w2(A, B):
    d = B - A
    return torch.einsum("slk,smk->lms", d, d) / A.shape[-1]


def psi_energy(A, B):
    D, w = cdf_diff(A, B)
    return 2.0 * torch.einsum("slg,smg->lms", D * w.unsqueeze(1), D)


def median_bw(A, B, cap=512):
    S, L, K = A.shape
    pool = torch.cat([A.reshape(S, L * K), B.reshape(S, L * K)], 1).sort(-1).values
    if pool.shape[1] > cap:
        pool = pool[:, torch.linspace(0, pool.shape[1] - 1, cap, device=pool.device).round().long()]
    med = (pool.unsqueeze(2) - pool.unsqueeze(1)).abs().reshape(S, -1).median(dim=1).values
    return torch.maximum(med, torch.maximum(1e-3 * pool.std(dim=1), torch.full_like(med, 1e-12)))


def psi_mmd(A, B, R=128):
    S = A.shape[0]
    t, h = np.polynomial.hermite.hermgauss(R)
    t = torch.tensor(t, dtype=DT, device=A.device)
    om = torch.tensor(h / math.sqrt(math.pi), dtype=DT, device=A.device)
    w = (math.sqrt(2.0) * t).view(1, R) / median_bw(A, B).view(S, 1)

    def emb(X):
        ph = X.unsqueeze(-1) * w.view(S, 1, 1, R)
        return torch.complex(ph.cos().mean(2), ph.sin().mean(2))
    d = (emb(B) - emb(A)) * om.view(1, 1, R).sqrt().to(torch.complex128)
    return (torch.einsum("slr,smr->lms", d.real, d.real)
            + torch.einsum("slr,smr->lms", d.imag, d.imag))


def ks_rows(A, B, pats, chunk=1024):
    """(P, S) orbit values of max_t |sum_i sigma_i D_i(t)| / L (grid thinned to 1024)."""
    D, _ = cdf_diff(A, B)
    S, L, G = D.shape
    if G > KS_GRID:
        D = D[:, :, torch.linspace(0, G - 1, KS_GRID, device=D.device).round().long()]
    out = torch.empty(pats.shape[0], S, device=D.device, dtype=DT)
    for s in range(S):
        for p0 in range(0, pats.shape[0], chunk):
            out[p0:p0 + chunk, s] = (pats[p0:p0 + chunk] @ D[s]).abs().max(1).values / L
    return out


def p_and_T(A):
    """max-z combiner (= msw.components.p_maxz) and the observed statistic."""
    A = C.finite(A)
    Z = C.finite((A - A.mean(0)) / C.std_floor(A))
    T = Z.max(1).values
    return (1.0 + float((T[1:] >= T[0]).sum())) / A.shape[0], float(T[0])


class Bench:
    """One pass over a pair of datasets -> (p, T) for MSW and every rival."""

    def __init__(self, channels, size, Ksub=128, L=16, n_patterns=4096, seed=0):
        self.P = Portfolio(channels, size, L=L, n_patterns=n_patterns, seed=seed)
        self.Ksub = Ksub
        names = self.P.names
        G = groups(names)
        self.core = core_columns(names)
        self.T_idx = [names.index(n) for n in G["T"]]
        self.rest_idx = [names.index(n) for n in G["TL"] + G["PX"] + G["J"]]

    def rival_tables(self, a, b, pats, L, M):
        port = self.P.port
        fams = [(port.base, None)] + ([(port.opp, self.P.chroma)] if port.opp is not None else [])
        cols = {m: [] for m in RIVALS}
        for est, cmask in fams:
            fa, fb = est.features(a), est.features(b)
            psis = {m: [] for m in ("w2", "energy", "mmd")}
            ks, lsz = [], []
            for li, (ma, q) in enumerate(zip(fa.mats, fa.per_image)):
                A = sub_blocks(ma, q, L, M, self.Ksub, seed=1000 * li + 7)
                B = sub_blocks(fb.mats[li], q, L, M, self.Ksub, seed=1000 * li + 7)
                psis["w2"].append(psi_w2(A, B))
                psis["energy"].append(psi_energy(A, B))
                psis["mmd"].append(psi_mmd(A, B))
                ks.append(ks_rows(A, B, pats))
                lsz.append(A.shape[0])
            for m in ("w2", "energy", "mmd"):
                cols[m] += C.transport_components(torch.cat(psis[m], 2), pats, lsz, iv_cols=cmask)
            z = torch.cat(ks, 1)
            q_ = z.pow(2).mean(0).sqrt()
            z = z / torch.maximum(q_, torch.clamp(C.REL * q_.mean().expand_as(q_), min=1e-300))
            zm = z if cmask is None else z[:, cmask]
            mu, j = [], 0
            for c in lsz:
                mu.append(z[:, j:j + c].mean(1))
                j += c
            cols["ks"] += [zm.mean(1), torch.stack(mu, 1).max(1).values, zm.max(1).values]
        return {m: torch.stack(v, 1) for m, v in cols.items()}

    @torch.no_grad()
    def run(self, a, b, hybrids=True):
        """{method: [p, T]}: MSW_CORE (= msw.test), MSW_T (transport rows only),
        the four rivals, and with hybrids each rival + TL + PX + J."""
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n].to(self.P.device).float(), b[:n].to(self.P.device).float()
        L, M = block_plan(n, n, self.P.L)
        pats = patterns_for(L, self.P.n_patterns, self.P.pattern_mode, device=self.P.device)
        A = self.P.table(a, b)
        out = {"MSW_CORE": p_and_T(A[:, self.core]), "MSW_T": p_and_T(A[:, self.T_idx])}
        for m, tab in self.rival_tables(a, b, pats, L, M).items():
            out[m] = p_and_T(tab)
            if hybrids:
                out[m + "+TLPXJ"] = p_and_T(torch.cat([tab, A[:, self.rest_idx]], 1))
        return {k: list(v) for k, v in out.items()}
