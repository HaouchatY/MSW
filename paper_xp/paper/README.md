# The paper's experiments

One folder per figure or table. Each holds the code that reproduces it (`run.py`, then `table.py` or `figure.py`), its results (`results/`), and the figure or table itself.

| | |
|---|---|
| [main/01](main/01_slice_marginals) | Fig. 1 — what one slice sees |
| [main/02](main/02_detection_thresholds) | Table 1 — detection thresholds at N = 500 |
| [main/03](main/03_images_needed) | Table 2 — images needed |
| [main/04](main/04_answer_key) | Fig. 3 — ordering and calibration against a known answer |

Every MSW number is recomputed with the public library (`msw`, pinned in `requirements.txt`): `msw.test` for detection, `msw.distance` for ordering. Baseline columns are the values of the original runs (`results/baselines.json`); `common/baselines.py` recomputes them identically.

## Running

```
pip install -r requirements.txt
export MSW_XP_DATA=/data/msw_xp     # celeba_64_16000.pt, cifar10_train.parquet, ...
```

`common/` holds the image distributions (`draws.py`), the decision rules (`protocol.py`), MSW through the library (`ours.py`) and the baseline metrics (`baselines.py`). Draws use CUDA generators with the seeds in `draws.py`; on another GPU or torch version the images differ, the numbers within Monte-Carlo error.
