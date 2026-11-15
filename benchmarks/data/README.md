# Benchmark datasets

Raw files from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/), unchanged, downloaded on 2026-11-15 by `benchmarks/benchmark.py`. Refresh with `uv run python benchmarks/benchmark.py --download`. UCI datasets are licensed CC BY 4.0; cite the repository and each dataset's donors (see its `.names` file on UCI).

| Name | File | Cases | Notes |
|---|---|---|---|
| iris | `iris.data` | 150 | 4 continuous, 3 classes |
| breast-w | `breast-cancer-wisconsin.data` | 699 | ID in column 1 (dropped); `?` missing |
| glass | `glass.data` | 214 | ID in column 1 (dropped); 6 classes |
| credit-a | `crx.data` | 690 | mixed types; `?` missing |
| credit-g | `german.data` | 1000 | space-separated; mixed types |
| heart-c | `processed.cleveland.data` | 303 | class 0 → `absent`, 1–4 → `present`; `?` missing |
| hepatitis | `hepatitis.data` | 155 | class in column 1 (moved last); `?` missing |
| sonar | `sonar.all-data` | 208 | 60 continuous |
| car | `car.data` | 1728 | discrete only |
| mushroom | `agaricus-lepiota.data` | 8124 | class in column 1 (moved last); `?` missing |
| vote | `house-votes-84.data` | 435 | class in column 1 (moved last); many `?` |
| adult | `adult.data`, `adult.test` | 32561 + 16281 | train/test; `adult.test` has a junk first line and classes ending in `.` |

The benchmark rewrites each file into a temporary directory (decision last, IDs dropped) because the loader cannot yet pick the decision column or ignore columns (tickets 31, 35).

Pima diabetes (Quinlan's `diabetes`) is not here: UCI no longer hosts it.
