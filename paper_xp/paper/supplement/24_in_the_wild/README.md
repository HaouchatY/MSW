# Preprocessing in the wild

Deflection z at N = 16 / 64 / 256; bold: z ≥ 2.4865.

| contrast | FID | KID | CMMD | DINOv2 | C2ST | MSW |
|---|---|---|---|---|---|---|
| bicubic vs bilinear (64 → 32 px) | +1.1 / −0.2 / +0.7 | +0.5 / +1.6 / **+3.2** | **+3.1** / **+15.6** / **+41.6** | +0.5 / +1.5 / +1.0 | +0.6 / +1.1 / **+6.0** | **+5.9** / **+7.1** / **+7.0** |
| JPEG q = 95 vs q = 90 (64 px) | +0.4 / +0.2 / −0.1 | +0.6 / +2.3 / +1.8 | +1.1 / **+6.2** / **+14.3** | −0.0 / +0.9 / −0.4 | +0.1 / +0.2 / +0.4 | +0.7 / **+4.1** / **+5.3** |

Two real pipeline differences on disjoint CelebA pools (images 0–6143 against the second half), no injected corruption; a null repetition passes both pools through the same pipeline, an alternative one passes the B pool through the other. z = (mean alt − mean null)/sd null, 12 repetitions for the archived baselines and 200 null + 200 alternative repetitions for MSW's T. The exact test rejects the resize contrast with power 1.00 at every N and the JPEG contrast with power 0.13 / 0.975 / 1.00 (size 0.035–0.060 over the six null cells); the bold pattern of the MSW column is the same as in the paper. Coverage, SW and CSW (in `table.csv`) stay below the bar everywhere.

At N = 8 (resize only, `python run.py --tasks resize --N 8`) MSW rejects in 72% of 200 repetitions at null size 0.035; N = 32 and 128 were not rerun.

```
python run.py && python table.py      # ~40 GPU-minutes (run.py estimate; not logged); baselines: results/baselines.json
```
