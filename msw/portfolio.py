"""The component portfolio: slice families, the component menu, and the orbit table.

``Portfolio.table(a, b)`` makes ONE pass over a pair of datasets and returns the
matrix ``A`` of shape ``(P, n_components)``: row 0 is the observed statistic and
row ``e`` is the same statistic evaluated on the data relabelled by group element
``e`` (see :mod:`msw.pairgroup`).  Every candidate portfolio is a column subset
of ``A``, so an arbitrary number of them can be evaluated after the fact from the
one expensive pass -- which is how the shipped choice was selected.

Two slice families are carried:

  ``b``  the per-channel DCT bank (the default estimator of :mod:`msw.estimators`)
  ``c``  the opponent bank restricted to its chroma atoms (3-channel data only)

Each family contributes the transport components and, from the same sorted
samples, a menu of signed first-order components.  Two families do not depend on
a bank: the raw-pixel statistics ``px`` and the product slices ``j``.

Component groups
----------------
Each signed family produces the same three aggregations of its (P, S) z-table:
``iv`` = |mean_s z_s| (coherent, precision-weighted), ``mx`` = max_s |z_s| (one
slice carries the signal), ``chi`` = mean_s z_s^2 (no common sign across slices).

  T    transport U-statistic over all L(L-1)/2 block pairs: ``iv``, ``ml``
       (max over pyramid levels), ``ivp`` (orbit-invariant precisions)
  E    block and per-image power contrasts (benergy, babs, ienergy, iabs)
  B    log power (blogpow, ilogpow)
  TL   tail and shape: ``bkurt`` = log(mean s^4 / (mean s^2)^2) and
       ``bsh`` = log(q_.99875 - q_.00125) - log(IQR), a scale-free shape contrast
  EXT  per-image extremes (imin, imax) and per-image kurtosis (ikurt)
  LOC  block location (bloc)
  MIX  the per-slice mean over all the above z-types, then the three aggregations
  PX   per-image RAW-PIXEL statistics (min, max, mean, log var, skew, kurtosis,
       for the greyscale image and for each colour channel)
  J    signed product slices: per-image normalised means of r(x) r(x+d) for six
       lags, r_i^2, and |r_i| |r_j| across orientations and scales

The shipped portfolio is ``CORE = T + TL + PX + J``: 15 columns on 1-channel
data, 24 on 3-channel data.
"""

from __future__ import annotations

import time

import torch

from . import components as C
from .estimators import PortfolioSW
from .joint import JointSW, LAGSETS
from .pairgroup import block_plan, patterns_for

__all__ = ["Portfolio", "groups", "portfolios", "core_columns", "evaluate",
           "CORE_GROUPS"]

CORE_GROUPS = ("T", "TL", "PX", "J")


def _level_z(psi: torch.Tensor, level_sizes: list[int]) -> list[float]:
    """Per-level z of the identity transport rows -- the diagnostic read-out.

    ``rows_i = sum_{j != i} Psi_ij / (L - 1)``; within a pyramid level the rows
    are averaged over slices and turned into a z by their spread over the L
    blocks.  This is the identity row of the ``ml`` component, split by level.
    """
    L = psi.shape[0]
    i = torch.arange(L, device=psi.device)
    p0 = psi.clone()
    p0[i, i, :] = 0.0
    rows = p0.sum(1) / (L - 1.0)                       # (L, S)
    out, j = [], 0
    for c in level_sizes:
        tl = rows[:, j:j + c].mean(1)
        se = torch.maximum(tl.std() / L ** 0.5,
                           torch.clamp(C.REL * tl.pow(2).mean().sqrt(), min=1e-300))
        out.append(float(tl.mean() / se))
        j += c
    return out


@torch.no_grad()
def _joint_u(est: JointSW, x: torch.Tensor) -> torch.Tensor:
    """(S', N) per-IMAGE normalised product-slice means (correlation form).

    Each column is a function of ONE image, so a pair swap permutes columns
    between the two datasets and the pooled variance used for the precision
    weights is a constant of the orbit.
    """
    im = est.image_means(x).to(C.DT)
    d = (im[:, est.den1] * im[:, est.den2]).clamp_min(1e-30).sqrt()
    return (im / d)[:, est.norm_rows].t().contiguous()


class Portfolio:
    """The slice families and the component menu, held together.

    Parameters
    ----------
    channels, size   image shape (C, size, size).
    levels           pyramid depth (default: 5 at 64 px, 4 below).
    k, n_filters     filter size and filters per level of the DCT family;
                     ``n_filters=None`` (the default) keeps the whole bank, which
                     is deterministic and identical on CPU and GPU.
    use_joint        include the product-slice family J.
    L                number of matched image blocks (see `msw.pairgroup`).
    n_patterns       group elements to enumerate: the full 2^L when it is no
                     larger, otherwise a parity subgroup of this order.
    pattern_mode     'subgroup' (default), 'contig' or 'mc'; see `patterns_for`.
    seed             seed of the estimators (bank subsets, kept positions).
    """

    def __init__(self, channels: int, size: int, levels: int | None = None,
                 k: int = 5, n_filters: int | None = None, use_joint: bool = True,
                 L: int = 16, n_patterns: int = 4096, pattern_mode: str = "subgroup",
                 seed: int = 0):
        levels = levels or (5 if size == 64 else 4)
        self.port = PortfolioSW(channels, size, levels=levels, k=k,
                                n_filters=n_filters, seed=seed)
        self.channels, self.size, self.levels = channels, size, levels
        self.L, self.n_patterns, self.pattern_mode = L, n_patterns, pattern_mode
        self.use_joint = use_joint
        self.pattern_seed = 0
        self.jest = (JointSW(channels, size, levels=levels, k=k, gray=True, seed=seed,
                             families=("var", "autoprod", "magprod"),
                             lags=LAGSETS["6"]) if use_joint else None)
        dev = self.port.base.layouts[0].filters.device
        self.device = dev
        self.chroma = (torch.tensor(self.port._chroma_idx, device=dev)
                       if self.port.opp is not None else None)

    # ---------------------------------------------------------------- names
    def _family_names(self, tag: str) -> list[str]:
        n = [f"{tag}_T_iv", f"{tag}_T_ml", f"{tag}_T_ivp"]
        for k in C.BLOCK_SUMS + C.IMG_SUMS + ("mixA",):
            n += [f"{tag}_{k}_iv", f"{tag}_{k}_mx", f"{tag}_{k}_chi"]
        return n

    @property
    def names(self) -> list[str]:
        """Column names of the orbit table, in order."""
        n = self._family_names("b")
        if self.port.opp is not None:
            n += self._family_names("c")
        n += ["px_iv", "px_mx", "px_chi"]
        if self.use_joint:
            n += ["j_iv", "j_mx", "j_chi"]
        return n

    # ------------------------------------------------------------------ table
    @torch.no_grad()
    def table(self, a: torch.Tensor, b: torch.Tensor, timing: dict | None = None,
              diagnostics: dict | None = None):
        """(P, n_components) orbit table; row 0 is the identity (observed) row.

        Also returns nothing else: ``L`` and the group size are recovered from
        ``plan(a, b)`` and from the number of rows.
        """
        t0 = time.time()
        n = min(a.shape[0], b.shape[0])
        a = a[:n].to(self.device).float()
        b = b[:n].to(self.device).float()
        L, M = block_plan(n, n, self.L)
        pats = patterns_for(L, self.n_patterns, self.pattern_mode,
                            seed=self.pattern_seed, device=self.device)
        cols = []
        fams = [("b", self.port.base, None)]
        if self.port.opp is not None:
            fams.append(("c", self.port.opp, self.chroma))
        for tag, est, cmask in fams:
            t = time.time()
            fa = C.to_double(est.features(a))
            fb = C.to_double(est.features(b))
            if timing is not None:
                timing[f"feat_{tag}"] = timing.get(f"feat_{tag}", 0) + time.time() - t
            t = time.time()
            psi, bs, bse = C.block_rows(fa, fb, L, M)
            if timing is not None:
                timing[f"sort_{tag}"] = timing.get(f"sort_{tag}", 0) + time.time() - t
            t = time.time()
            if diagnostics is not None:
                self._diagnostics(tag, est, psi, diagnostics)
            cols += C.transport_components(psi, pats, est.level_sizes, iv_cols=cmask)
            zs = [C.transport_z(psi, pats)]
            del psi
            for k in C.BLOCK_SUMS:
                z = C.signed_z(bs[k], bse[k], pats)
                zs.append(z)
                cols += C.zcomps(z if cmask is None else z[:, cmask])
            ua, ub = C.per_image_u(fa), C.per_image_u(fb)
            del fa, fb
            for k in C.IMG_SUMS:
                m, se = C.image_signed_rows(ua[k], ub[k], L, M)
                z = C.signed_z(m, se, pats)
                zs.append(z)
                cols += C.zcomps(z if cmask is None else z[:, cmask])
            del ua, ub
            zmix = torch.stack(zs, 0).mean(0)
            cols += C.zcomps(zmix if cmask is None else zmix[:, cmask])
            del zs, zmix
            if timing is not None:
                timing[f"comp_{tag}"] = timing.get(f"comp_{tag}", 0) + time.time() - t
        t = time.time()
        m, se = C.image_signed_rows(C.pixel_u(a), C.pixel_u(b), L, M)
        cols += C.zcomps(C.signed_z(m, se, pats))
        if timing is not None:
            timing["px"] = timing.get("px", 0) + time.time() - t
        if self.use_joint:
            t = time.time()
            m, se = C.image_signed_rows(_joint_u(self.jest, a), _joint_u(self.jest, b), L, M)
            cols += C.zcomps(C.signed_z(m, se, pats))
            if timing is not None:
                timing["joint"] = timing.get("joint", 0) + time.time() - t
        if timing is not None:
            timing["total"] = timing.get("total", 0) + time.time() - t0
        return torch.stack(cols, 1)

    def _diagnostics(self, tag, est, psi, out: dict):
        """Per-level transport z, by slice family, for the plain-language read-out."""
        if tag == "b":
            out["per-channel DCT"] = _level_z(psi, est.level_sizes)
            return
        lum, chr_, i = [], [], 0
        for c in est.level_sizes:
            lum += [i + j for j in range(c) if j % 3 == 0]
            chr_ += [i + j for j in range(c) if j % 3 != 0]
            i += c
        per_lvl = [c // 3 for c in est.level_sizes]
        out["opponent luminance"] = _level_z(psi[:, :, lum], per_lvl)
        out["opponent chroma"] = _level_z(psi[:, :, chr_], [2 * v for v in per_lvl])

    def plan(self, a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
        """(L, M) for this pair of datasets: blocks, and images per block."""
        return block_plan(a.shape[0], b.shape[0], self.L)


# ========================================================= named portfolios
def _idx(names, keys):
    return [names.index(k) for k in keys if k in names]


def groups(names: list[str]) -> dict:
    """Component group name -> the column names it contains."""
    G = {}
    G["T"] = [n for n in names if "_T_" in n]
    G["Tc"] = [n for n in names if n in ("b_T_iv", "b_T_ml", "c_T_iv", "c_T_ml")]
    G["Tp"] = [n for n in names if n in ("b_T_ivp", "b_T_ml", "c_T_ivp", "c_T_ml")]
    G["E"] = [n for n in names if "_benergy_" in n or "_babs_" in n]
    G["B"] = [n for n in names if "_blogpow_" in n or "_bkurt_" in n]
    G["SH"] = [n for n in names if "_bsh_" in n]
    G["TL"] = [n for n in names if "_bsh_" in n or "_bkurt_" in n]
    G["POW"] = [n for n in names if "_blogpow_" in n]
    G["LOC"] = [n for n in names if "_bloc_" in n]
    G["IE"] = [n for n in names if "_ienergy_" in n or "_iabs_" in n]
    G["I"] = [n for n in names if "_ilogpow_" in n or "_ikurt_" in n]
    G["EXT"] = [n for n in names if "_imin_" in n or "_imax_" in n]
    G["MIX"] = [n for n in names if "_mixA_" in n]
    G["PX"] = [n for n in names if n.startswith("px_")]
    G["J"] = [n for n in names if n.startswith("j_")]
    return G


def portfolios(names: list[str]) -> dict:
    """Named portfolio -> column indices.  Columns absent on 1-channel data drop."""
    G = groups(names)
    P = {}

    def add(key, *gs):
        cols = []
        for g in gs:
            cols += G[g]
        P[key] = _idx(names, cols)

    P["v01_like"] = _idx(names, G["Tc"])
    P["v01_pooledw"] = _idx(names, G["Tp"])
    add("T_all", "T")
    add("T+SH", "T", "SH")
    add("T+PX", "T", "PX")
    add("T+J", "T", "J")
    add("T+B", "T", "B")
    add("T+E", "T", "E")
    add("T+E+SH", "T", "E", "SH")
    add("T+E+SH+PX", "T", "E", "SH", "PX")
    add("T+SH+PX", "T", "SH", "PX")
    add("LEAN", "T", "SH", "PX", "J")
    add("LEAN+B", "T", "B", "SH", "PX", "J")
    add("LEAN+MIX", "T", "SH", "PX", "J", "MIX")
    add("LEAN+EXT", "T", "SH", "PX", "J", "EXT")
    add("WIDE", "T", "E", "SH", "PX", "J")
    add("WIDE-T", "E", "SH", "PX", "J")
    add("WIDE-E", "T", "SH", "PX", "J")
    add("WIDE-SH", "T", "E", "PX", "J")
    add("WIDE-PX", "T", "E", "SH", "J")
    add("WIDE-J", "T", "E", "SH", "PX")
    add("WIDE+B", "T", "E", "B", "SH", "PX", "J")
    add("WIDE+MIX", "T", "E", "SH", "PX", "J", "MIX")
    add("WIDE+EXT", "T", "E", "SH", "PX", "J", "EXT")
    add("WIDE+IE", "T", "E", "SH", "PX", "J", "IE")
    add("WIDE+ALLIMG", "T", "E", "SH", "PX", "J", "IE", "I", "EXT")
    add("C_T+TL", "T", "TL")
    add("C_T+TL+PX", "T", "TL", "PX")
    add("C_T+TL+J", "T", "TL", "J")
    add("CORE", *CORE_GROUPS)
    add("CORE+POW", "T", "TL", "POW", "PX", "J")
    add("CORE+E", "T", "TL", "E", "PX", "J")
    add("CORE+MIX", "T", "TL", "PX", "J", "MIX")
    add("CORE+EXT", "T", "TL", "PX", "J", "EXT")
    add("CORE-T", "TL", "PX", "J")
    add("CORE-TL", "T", "PX", "J")
    add("CORE-PX", "T", "TL", "J")
    add("CORE-J", "T", "TL", "PX")
    P["TL_only"] = _idx(names, G["TL"])
    P["POW_only"] = _idx(names, G["POW"])
    P["FULL"] = list(range(len(names)))
    for g in ("T", "E", "B", "SH", "LOC", "IE", "I", "EXT", "MIX", "PX", "J"):
        P[g + "_only"] = _idx(names, G[g])
    return {k: v for k, v in P.items() if v}


def core_columns(names: list[str]) -> list[int]:
    """Columns of the shipped portfolio CORE = T + TL + PX + J."""
    G = groups(names)
    return _idx(names, [n for g in CORE_GROUPS for n in G[g]])


def evaluate(A: torch.Tensor, names: list[str], pf: dict | None = None):
    """(portfolio -> (p_cct, p_maxz, p_wy), per-component p-values)."""
    pc_all = C.col_p(A)
    pf = pf or portfolios(names)
    out = {}
    for k, idx in pf.items():
        S = A[:, idx]
        out[k] = (C.p_cct([pc_all[i] for i in idx]), C.p_maxz(S), C.p_minp_wy(S))
    return out, pc_all
