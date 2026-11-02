# Relation shuffling of the trained final models

Patient-level bootstrap, 2000 resamples, intervals withheld below 10 patients. The models were never trained on shuffled graphs: read a change as dependence on the relations, not as proof of meaningful use.

## Effect of changing the relations of the trained models

Accuracy and probability changes are after minus before, per patient (mean over visits, seeds and shuffles), patient-level 95% interval. Flip = the predicted class changes. Moved/removed = share of visits where at least one edge was replaced (shuffle) or the graph lost edges (drop). Macro-F1 change: mean ± SD over seeds.

### K = 10

**cei_mp2** (seeds [7, 1234, 2025], 634 visits, 500 patients, 3 shuffles per arm, accuracy before 0.757)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -25.72 [-28.84, -22.48] (n=500) | 46.00 [42.92, 49.25] (n=500) | -0.423 [-0.437, -0.409] (n=500) | 100.0 | -0.2712 ± 0.0653 |
| drop_informative | -16.35 [-19.56, -13.33] (n=500) | 32.29 [29.25, 35.46] (n=500) | -0.286 [-0.302, -0.269] (n=500) | 100.0 | -0.1541 ± 0.0053 |
| shuffle_all | -2.30 [-3.84, -0.80] (n=500) | 8.78 [7.12, 10.53] (n=500) | -0.049 [-0.058, -0.040] (n=500) | 100.0 | -0.0256 ± 0.0139 |
| shuffle_informative | -1.44 [-2.60, -0.34] (n=500) | 5.88 [4.66, 7.20] (n=500) | -0.030 [-0.037, -0.023] (n=500) | 100.0 | -0.0172 ± 0.0103 |
| shuffle_structural | -1.21 [-2.25, -0.26] (n=500) | 3.96 [2.93, 5.14] (n=500) | -0.019 [-0.025, -0.013] (n=500) | 99.3 | -0.0139 ± 0.0158 |

### K = 20

**cei_mp2** (seeds [7, 1234, 2025], 672 visits, 500 patients, 3 shuffles per arm, accuracy before 0.701)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -36.41 [-40.45, -32.45] (n=500) | 63.86 [60.68, 67.12] (n=500) | -0.433 [-0.452, -0.415] (n=500) | 100.0 | -0.3933 ± 0.0898 |
| drop_informative | -23.48 [-27.20, -19.75] (n=500) | 46.00 [42.80, 49.39] (n=500) | -0.319 [-0.339, -0.299] (n=500) | 100.0 | -0.2427 ± 0.0279 |
| shuffle_all | -2.57 [-4.20, -1.02] (n=500) | 11.57 [9.67, 13.61] (n=500) | -0.061 [-0.072, -0.050] (n=500) | 100.0 | -0.0289 ± 0.0108 |
| shuffle_informative | -1.30 [-2.67, +0.06] (n=500) | 8.32 [6.73, 10.07] (n=500) | -0.041 [-0.050, -0.033] (n=500) | 100.0 | -0.0173 ± 0.0112 |
| shuffle_structural | -1.25 [-2.25, -0.29] (n=500) | 5.21 [3.96, 6.62] (n=500) | -0.020 [-0.027, -0.014] (n=500) | 98.5 | -0.0105 ± 0.0059 |

### K = 30

**cei_mp2** (seeds [7, 1234, 2025], 671 visits, 500 patients, 3 shuffles per arm, accuracy before 0.652)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -28.55 [-31.93, -25.17] (n=500) | 57.06 [53.92, 60.26] (n=500) | -0.461 [-0.478, -0.442] (n=500) | 100.0 | -0.2704 ± 0.0558 |
| drop_informative | -18.98 [-22.14, -15.68] (n=500) | 40.93 [37.68, 44.38] (n=500) | -0.314 [-0.332, -0.296] (n=500) | 100.0 | -0.1838 ± 0.0454 |
| shuffle_all | -1.67 [-3.07, -0.26] (n=500) | 10.81 [8.99, 12.77] (n=500) | -0.055 [-0.064, -0.046] (n=500) | 100.0 | -0.0192 ± 0.0040 |
| shuffle_informative | -1.27 [-2.47, -0.11] (n=500) | 7.56 [6.09, 9.25] (n=500) | -0.038 [-0.046, -0.030] (n=500) | 100.0 | -0.0161 ± 0.0049 |
| shuffle_structural | -0.56 [-1.53, +0.26] (n=500) | 4.42 [3.36, 5.55] (n=500) | -0.016 [-0.021, -0.012] (n=500) | 98.6 | -0.0042 ± 0.0015 |

