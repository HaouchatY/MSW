"""Quickstart: certify a spectral difference between two Gaussian random
fields whose true W2 distance is known in closed form."""
import torch

import msw
from msw.spectral import powerlaw_spectrum, sample_field, w2_exact

SIZE, N = 32, 500
s0 = powerlaw_spectrum(SIZE, 3.0, total_power=1.0)     # reference spectrum
s1 = powerlaw_spectrum(SIZE, 3.1, total_power=1.0)     # slightly steeper slope

g = torch.Generator(device=msw.get_device()).manual_seed(0)
A = sample_field(s0, N, generator=g)                   # (N, 1, 32, 32)
B = sample_field(s1, N, generator=g)

r = msw.test(A, B)
print(f"portfolio statistic T = {r.T:.2f}")
print(f"exact p               = {r.p:.4f}   (CCT {r.p_cct:.4f}, WY {r.p_wy:.4f})")
print(f"{r.L} blocks of {r.M} images, {r.group_size} group elements enumerated")
print(f"components            = { {k: round(v, 4) for k, v in r.components.items()} }")
print(f"per-component p       = { {k: v for k, v in r.component_p.items() if v <= 0.05} }")
print(f"diagnosis             : {r.diagnose()}")
print(f"stationarity index    = {r.stationarity_index:.3f}")

d, lo, hi = msw.distance(A, B)
print(f"distance D = {d:.4f}  (95% CI on D: [{lo:.4f}, {hi:.4f}])")
print(f"true W2-bar between the spectra = {w2_exact(s0, s1)[1]:.4f}")

# The test is a real test at eight images per side: the pair group then has
# 2^8 = 256 elements, so the smallest attainable p-value is 1/256.
small = msw.test(A[:8], B[:8])
print(f"N=8: p = {small.p:.4f} over {small.group_size} group elements")
