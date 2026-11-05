"""From cached features to the numbers of every CMMD figure and table.

For every measured set: FID, FID_inf and CMMD against its reference, and a
bootstrap standard deviation for each (B resamples of the 30,000 caption
indices, the same indices for the set and the reference, since image i of both
belongs to caption i).  MSW comes from the measurement pass with its jackknife
interval.  Then, per series, the question the paper asks of every metric: does
it move in the right direction at every step, and is any wrong-way step larger
than the metric's own sampling noise?
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import cmmd, fid, fid_inf                    # noqa: E402

from paths import FEAT

OUT = FEAT                     # per-set metrics cache; series go to --out
B = int(os.environ.get("BOOT", "20"))
P = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
PB = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)

SERIES = {
    # name: (x values, set names, direction the truth moves: +1 = gets worse with x)
    "amused_iter": (list(range(1, 13)), [f"amused_s{k}" for k in range(1, 13)], -1),
    "sd_iter": ([95, 96, 97, 98, 99, 100], [f"sd_s{k}" for k in (95, 96, 97, 98, 99, 100)], -1),
    "amused_vq": (list(P), ["amused_s12"] + [f"amused_vq_p{p:.2f}" for p in P[1:]], +1),
    "coco_vq": (list(P), [f"coco_vq_p{p:.2f}" for p in P], +1),
    # 8-px tokens at 512 (VQ-Diffusion ITHQ), the Muse-like granularity
    "cocoB_vq": (list(PB), [f"cocoB_p{p:.2f}" for p in PB], +1),
    "sdB_vq": (list(PB), [f"sdB_p{p:.2f}" for p in PB], +1),
}


def load(name):
    return np.load(f"{FEAT}/{name}_inc.npy"), np.load(f"{FEAT}/{name}_clip.npy")


def measure(name, cache={}):
    path = f"{OUT}/metrics_{name}.json"
    if os.path.exists(path):
        return json.load(open(path))
    meta = json.load(open(f"{FEAT}/{name}.json"))
    ref = meta["ref"]
    if ref not in cache:
        cache.clear(); cache[ref] = load(ref)
    ri, rc = cache[ref]
    si, sc = load(name)
    n = len(si)
    out = {"name": name, "ref": ref, "n": n,
           "fid": fid(ri, si), "fid_inf": fid_inf(ri, si), "cmmd": cmmd(rc, sc),
           "msw": meta["msw"]["d"], "msw_lo": meta["msw"]["lo"], "msw_hi": meta["msw"]["hi"]}
    rng = np.random.default_rng(0)
    bf, bc = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        bf.append(fid(ri[idx], si[idx]))
        bc.append(cmmd(rc[idx], sc[idx]))
    out["fid_sd"], out["cmmd_sd"] = float(np.std(bf, ddof=1)), float(np.std(bc, ddof=1))
    # rng is re-seeded identically for every set and n is the same, so draw b
    # uses the same caption indices in every set: draws are paired across sets
    out["fid_boot"], out["cmmd_boot"] = [float(v) for v in bf], [float(v) for v in bc]
    out["msw_loo_u"] = meta["msw"].get("loo_u")
    # the jackknife half-width is a 95% interval; quote it as an sd for comparability
    out["msw_sd"] = (out["msw_hi"] - out["msw_lo"]) / (2 * 1.96)
    json.dump(out, open(path, "w"), indent=1)
    print(f"{name:18s} FID {out['fid']:8.3f}±{out['fid_sd']:.3f}  FIDinf {out['fid_inf']:8.3f}  "
          f"CMMD {out['cmmd']:7.4f}±{out['cmmd_sd']:.4f}  MSW {out['msw']:.5f}±{out['msw_sd']:.5f}",
          flush=True)
    return out


def kendall(x, y):
    c = d = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s = np.sign(x[j] - x[i]) * np.sign(y[j] - y[i])
            c += s > 0; d += s < 0
    return (c - d) / max(c + d, 1)


def _root(u):
    return np.sign(u) * np.sqrt(np.abs(u))


def step_noise(a, b, m):
    """sd of metric(b) - metric(a) for two sets on the same captions.

    FID, CMMD: paired bootstrap (draw k of both sets used the same indices).
    MSW: paired jackknife over the L matched blocks (same blocks in both sets).
    FID_inf: none (its 15-point regression has no resampling here).
    """
    if m in ("fid", "cmmd"):
        return float(np.std(np.array(b[m + "_boot"]) - np.array(a[m + "_boot"]), ddof=1))
    if m == "msw" and a.get("msw_loo_u") and b.get("msw_loo_u"):
        dk = _root(np.array(b["msw_loo_u"])) - _root(np.array(a["msw_loo_u"]))
        L = len(dk)
        return float(np.sqrt((L - 1) / L * ((dk - dk.mean()) ** 2).sum()))
    return float("nan")


def series(key):
    xs, names, sign = SERIES[key]
    have = [n for n in names if os.path.exists(f"{FEAT}/{n}.json")]
    if len(have) < len(names):
        print(f"{key}: {len(have)}/{len(names)} sets measured, skipped"); return None
    rows = [measure(n) for n in names]
    res = {"x": xs, "sets": names, "rows": rows, "summary": {}}
    for m in ("fid", "fid_inf", "cmmd", "msw"):
        v = np.array([r[m] for r in rows])
        sd = np.array([r.get(f"{m}_sd", np.nan) for r in rows])
        tau = kendall(np.array(xs, float), v) * sign            # +1 = perfectly right direction
        steps = np.diff(v) * sign                                # > 0 = right direction
        wrong = [int(i) for i in np.where(steps < 0)[0]]
        noise = np.array([step_noise(rows[i], rows[i + 1], m) for i in range(len(steps))])
        signif = [int(i) for i in wrong if np.isfinite(noise[i]) and -steps[i] > 2 * noise[i]]
        res["summary"][m] = {"tau": float(tau), "wrong_steps": wrong,
                             "wrong_steps_beyond_2sd": signif, "monotone": not wrong,
                             "step_noise": [float(x) for x in noise],
                             "steps_signed": [float(x) for x in steps]}
        print(f"  {key:12s} {m:8s} tau={tau:+.3f}  wrong-way steps {wrong}  beyond 2sd {signif}")
    json.dump(res, open(f"{SERIES_OUT}/series_{key}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score measured series against their references.")
    ap.add_argument("series", nargs="+", choices=sorted(SERIES))
    ap.add_argument("--out", required=True, help="the experiment's results/ folder")
    a = ap.parse_args()
    SERIES_OUT = a.out
    os.makedirs(SERIES_OUT, exist_ok=True)
    for k in a.series:
        series(k)
