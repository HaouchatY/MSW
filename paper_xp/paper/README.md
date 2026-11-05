# The paper's experiments

One folder per figure or table. Each holds the code that reproduces it (`run.py`, then `table.py` or `figure.py`), its results (`results/`), and the figure or table itself.

| | |
|---|---|
| [main/01](main/01_slice_marginals) | Fig. 1 — what one slice sees |
| [main/02](main/02_detection_thresholds) | Table 1 — detection thresholds at N = 500 |
| [main/03](main/03_images_needed) | Table 2 — images needed |
| [main/04](main/04_answer_key) | Fig. 3 — ordering and calibration against a known answer |
| [main/05](main/05_pipeline_audit) | Table 3 — which preprocessing pipeline made these images? |
| [supplement/01](supplement/01_datasets) | The nine settings |
| [supplement/02](supplement/02_timing) | Wall-clock per evaluation |
| [supplement/03](supplement/03_pooled_slices) | Generic two-sample tests on the same pooled slices |
| [supplement/04](supplement/04_sparse_processes) | Sparse stochastic processes |
| [supplement/05](supplement/05_feature_nullspace) | Resize blindness: the feature nullspace |
| [supplement/06](supplement/06_mode_dropping) | Mode dropping: semantic and statistical modes |
| [supplement/07](supplement/07_distance_growth) | Does the distance grow with the perturbation? |
| [supplement/08](supplement/08_power) | Power against sample count |
| [supplement/09](supplement/09_leaderboard) | Ranking ten models with a known key |
| [supplement/10](supplement/10_contrast_mixtures) | Moment-matched contrast mixtures |
| [supplement/11](supplement/11_ablations) | Ablations |
| [supplement/12](supplement/12_null_floor) | Zero floor on null draws |
| [supplement/13](supplement/13_test_size) | Size of the test |
| [supplement/14](supplement/14_gaming) | Gaming the metrics |
| [supplement/15](supplement/15_gender) | CelebA gender, aligned |
| [supplement/16](supplement/16_generators) | Real generators |
| [supplement/17](supplement/17_edm2) | EDM2 training snapshots |
| [supplement/18](supplement/18_t3_preprocessing) | Preprocessing on the t3 row |
| [supplement/19](supplement/19_dtd) | Held-out textures (DTD) |
| [supplement/20](supplement/20_color_cast) | Global colour cast |
| [supplement/21](supplement/21_filter_atlas) | Filter atlas |
| [supplement/22](supplement/22_ps_machinery) | Portilla–Simoncelli inside MSW's machinery |
| [supplement/23](supplement/23_audit_large_n) | Preprocessing audit at N = 800 |
| [supplement/24](supplement/24_in_the_wild) | Preprocessing in the wild |
| [supplement/25](supplement/25_reading_a_detection) | Reading a detection |
| [supplement/26](supplement/26_implementation_check) | Baselines against the official packages |

Every MSW number is recomputed with the public library (`msw`, pinned in `requirements.txt`): `msw.test` for detection, `msw.distance` for ordering. Baseline columns are the values of the original runs (`results/baselines.json`); `common/baselines.py` recomputes them identically.

## Running

```
pip install -r requirements.txt
export MSW_XP_DATA=/data/msw_xp     # celeba_64_16000.pt, cifar10_train.parquet, ...
```

`common/` holds the image distributions (`draws.py`), the decision rules (`protocol.py`), MSW through the library (`ours.py`) and the baseline metrics (`baselines.py`). Draws use CUDA generators with the seeds in `draws.py`; on another GPU or torch version the images differ, the numbers within Monte-Carlo error.
