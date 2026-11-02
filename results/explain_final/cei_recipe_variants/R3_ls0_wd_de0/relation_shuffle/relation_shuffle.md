# Relation shuffling of the trained final models

Patient-level bootstrap, 2000 resamples, intervals withheld below 10 patients. The models were never trained on shuffled graphs: read a change as dependence on the relations, not as proof of meaningful use.

## Effect of changing the relations of the trained models

Accuracy and probability changes are after minus before, per patient (mean over visits, seeds and shuffles), patient-level 95% interval. Flip = the predicted class changes. Moved/removed = share of visits where at least one edge was replaced (shuffle) or the graph lost edges (drop). Macro-F1 change: mean ± SD over seeds.

### K = 10

**cei_mp2** (seeds [7, 1234, 2025], 634 visits, 500 patients, 3 shuffles per arm, accuracy before 0.762)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -24.59 [-28.18, -21.36] (n=500) | 43.10 [39.96, 46.41] (n=500) | -0.385 [-0.404, -0.366] (n=500) | 100.0 | -0.2625 ± 0.0727 |
| drop_informative | -15.82 [-19.02, -12.96] (n=500) | 30.18 [27.09, 33.35] (n=500) | -0.258 [-0.278, -0.239] (n=500) | 100.0 | -0.1577 ± 0.0448 |
| shuffle_all | -2.36 [-3.52, -1.23] (n=500) | 6.62 [5.29, 8.01] (n=500) | -0.035 [-0.043, -0.028] (n=500) | 100.0 | -0.0249 ± 0.0126 |
| shuffle_informative | -1.46 [-2.40, -0.52] (n=500) | 4.83 [3.65, 6.10] (n=500) | -0.023 [-0.029, -0.017] (n=500) | 100.0 | -0.0152 ± 0.0116 |
| shuffle_structural | -0.85 [-1.57, -0.16] (n=500) | 2.59 [1.83, 3.49] (n=500) | -0.012 [-0.017, -0.008] (n=500) | 99.3 | -0.0086 ± 0.0023 |

### K = 20

**cei_mp2** (seeds [7, 1234, 2025], 672 visits, 500 patients, 3 shuffles per arm, accuracy before 0.702)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -26.29 [-29.79, -22.70] (n=500) | 51.16 [48.09, 54.36] (n=500) | -0.403 [-0.422, -0.384] (n=500) | 100.0 | -0.2523 ± 0.0449 |
| drop_informative | -19.44 [-22.88, -15.80] (n=500) | 39.07 [35.82, 42.45] (n=500) | -0.290 [-0.313, -0.267] (n=500) | 100.0 | -0.1824 ± 0.0309 |
| shuffle_all | -2.26 [-3.69, -0.91] (n=500) | 10.33 [8.62, 12.04] (n=500) | -0.047 [-0.057, -0.038] (n=500) | 100.0 | -0.0278 ± 0.0144 |
| shuffle_informative | -1.58 [-2.84, -0.29] (n=500) | 7.64 [6.16, 9.18] (n=500) | -0.033 [-0.041, -0.025] (n=500) | 100.0 | -0.0234 ± 0.0066 |
| shuffle_structural | -1.12 [-1.85, -0.44] (n=500) | 4.51 [3.52, 5.67] (n=500) | -0.014 [-0.020, -0.010] (n=500) | 98.5 | -0.0091 ± 0.0070 |

### K = 30

**cei_mp2** (seeds [7, 1234, 2025], 671 visits, 500 patients, 3 shuffles per arm, accuracy before 0.652)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -27.84 [-31.07, -24.63] (n=500) | 55.24 [52.26, 58.32] (n=500) | -0.460 [-0.479, -0.442] (n=500) | 100.0 | -0.2465 ± 0.0524 |
| drop_informative | -18.30 [-21.62, -15.09] (n=500) | 39.63 [36.58, 42.82] (n=500) | -0.306 [-0.326, -0.286] (n=500) | 100.0 | -0.1771 ± 0.0315 |
| shuffle_all | -1.41 [-2.71, -0.09] (n=500) | 9.29 [7.79, 10.92] (n=500) | -0.038 [-0.046, -0.029] (n=500) | 100.0 | -0.0184 ± 0.0021 |
| shuffle_informative | -1.04 [-2.20, +0.07] (n=500) | 6.86 [5.67, 8.18] (n=500) | -0.023 [-0.030, -0.016] (n=500) | 100.0 | -0.0147 ± 0.0031 |
| shuffle_structural | -0.48 [-1.28, +0.22] (n=500) | 3.44 [2.51, 4.47] (n=500) | -0.014 [-0.019, -0.009] (n=500) | 98.6 | -0.0019 ± 0.0074 |

