# Tab. "Reading a detection" — per-level transport z by slice family

| | level 0 | level 1 | level 2 |
|---|---|---|---|
| noise 0.0025, per-channel DCT | −0.2 | −0.2 | −0.1 |
| noise 0.0025, opponent luminance | −0.2 | −0.2 | −0.2 |
| noise 0.0025, opponent chroma | **1.5** | 0.0 | −0.1 |
| shading 0.08, per-channel DCT | **15.2** | 11.3 | 8.5 |
| shading 0.08, opponent luminance | 14.6 | 11.0 | 8.6 |
| shading 0.08, opponent chroma | 0.3 | 0.0 | −0.1 |
| null, per-channel DCT | −0.3 | 0.1 | 0.0 |
| null, opponent luminance | −0.4 | 0.0 | −0.0 |
| null, opponent chroma | −0.8 | −0.8 | −0.8 |

`msw.test(...).per_level` (identity transport rows averaged over the slices of a level, unweighted, divided by their spread over the 16 blocks; level 0 = finest; 32-px crops have three levels), mean over 12 repetitions, N = 500, CelebA 32-px crops plus white noise of std 0.0025 or plus a unit-std 1–3-cycle field times 0.08. Power is 1.00 for both contrasts and 0.08 on the null. The colour axis localises (noise: chroma only; shading: luminance only), but the scale ordering of z does not follow the original table: shading's z falls from fine to coarse because the coarse levels have fewer, noisier positions, while its raw transport mean grows about 60× from level 0 to level 2 (CPU check, N = 400). The noise detection is carried by the precision-weighted chroma transport component (c_T_iv, z ≈ 19 at N = 400), which the unweighted per-level read-out dilutes to 1.5.

```
python run.py && python table.py      # ~3 GPU-minutes
```
