# Relation shuffling of the trained final models

Patient-level bootstrap, 2000 resamples, intervals withheld below 10 patients. The models were never trained on shuffled graphs: read a change as dependence on the relations, not as proof of meaningful use.

## Effect of changing the relations of the trained models

Accuracy and probability changes are after minus before, per patient (mean over visits, seeds and shuffles), patient-level 95% interval. Flip = the predicted class changes. Moved/removed = share of visits where at least one edge was replaced (shuffle) or the graph lost edges (drop). Macro-F1 change: mean ± SD over seeds.

### K = 10

**cei_mp2** (seeds [7, 1234, 2025], 634 visits, 500 patients, 3 shuffles per arm, accuracy before 0.760)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -13.63 [-16.63, -10.93] (n=500) | 28.42 [25.50, 31.62] (n=500) | -0.285 [-0.303, -0.269] (n=500) | 100.0 | -0.1316 ± 0.0260 |
| drop_informative | -9.01 [-11.82, -6.39] (n=500) | 22.16 [19.35, 25.04] (n=500) | -0.176 [-0.193, -0.160] (n=500) | 100.0 | -0.0859 ± 0.0030 |
| shuffle_all | -1.32 [-2.32, -0.37] (n=500) | 4.63 [3.62, 5.81] (n=500) | -0.025 [-0.032, -0.019] (n=500) | 100.0 | -0.0106 ± 0.0058 |
| shuffle_informative | -0.56 [-1.46, +0.29] (n=500) | 3.30 [2.37, 4.35] (n=500) | -0.016 [-0.021, -0.011] (n=500) | 100.0 | -0.0029 ± 0.0014 |
| shuffle_structural | -0.77 [-1.34, -0.23] (n=500) | 1.74 [1.13, 2.45] (n=500) | -0.009 [-0.013, -0.005] (n=500) | 99.3 | -0.0071 ± 0.0054 |

### K = 20

**cei_mp2** (seeds [7, 1234, 2025], 672 visits, 500 patients, 3 shuffles per arm, accuracy before 0.693)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -11.25 [-13.99, -8.56] (n=500) | 28.77 [25.94, 31.71] (n=500) | -0.277 [-0.292, -0.263] (n=500) | 100.0 | -0.1008 ± 0.0195 |
| drop_informative | -10.24 [-13.20, -7.23] (n=500) | 27.79 [24.62, 30.85] (n=500) | -0.187 [-0.206, -0.168] (n=500) | 100.0 | -0.0927 ± 0.0229 |
| shuffle_all | -0.11 [-1.27, +1.10] (n=500) | 6.50 [5.20, 7.85] (n=500) | -0.027 [-0.034, -0.021] (n=500) | 100.0 | -0.0017 ± 0.0059 |
| shuffle_informative | +0.43 [-0.51, +1.36] (n=500) | 4.12 [3.14, 5.14] (n=500) | -0.017 [-0.022, -0.012] (n=500) | 100.0 | +0.0014 ± 0.0114 |
| shuffle_structural | -0.31 [-1.05, +0.44] (n=500) | 3.65 [2.73, 4.67] (n=500) | -0.009 [-0.013, -0.006] (n=500) | 98.5 | -0.0010 ± 0.0024 |

### K = 30

**cei_mp2** (seeds [7, 1234, 2025], 671 visits, 500 patients, 3 shuffles per arm, accuracy before 0.666)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -14.78 [-17.64, -12.05] (n=500) | 36.11 [33.30, 39.22] (n=500) | -0.354 [-0.370, -0.339] (n=500) | 100.0 | -0.1287 ± 0.0275 |
| drop_informative | -13.69 [-16.47, -10.95] (n=500) | 30.38 [27.44, 33.32] (n=500) | -0.225 [-0.243, -0.208] (n=500) | 100.0 | -0.1194 ± 0.0179 |
| shuffle_all | -1.14 [-2.33, +0.10] (n=500) | 7.89 [6.52, 9.42] (n=500) | -0.025 [-0.031, -0.018] (n=500) | 100.0 | -0.0043 ± 0.0071 |
| shuffle_informative | -0.43 [-1.40, +0.56] (n=500) | 5.18 [4.06, 6.37] (n=500) | -0.013 [-0.016, -0.009] (n=500) | 100.0 | +0.0009 ± 0.0127 |
| shuffle_structural | -0.60 [-1.30, +0.02] (n=500) | 3.01 [2.22, 3.88] (n=500) | -0.008 [-0.012, -0.005] (n=500) | 98.6 | -0.0022 ± 0.0067 |

