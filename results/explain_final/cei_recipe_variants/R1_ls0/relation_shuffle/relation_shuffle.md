# Relation shuffling of the trained final models

Patient-level bootstrap, 2000 resamples, intervals withheld below 10 patients. The models were never trained on shuffled graphs: read a change as dependence on the relations, not as proof of meaningful use.

## Effect of changing the relations of the trained models

Accuracy and probability changes are after minus before, per patient (mean over visits, seeds and shuffles), patient-level 95% interval. Flip = the predicted class changes. Moved/removed = share of visits where at least one edge was replaced (shuffle) or the graph lost edges (drop). Macro-F1 change: mean ± SD over seeds.

### K = 10

**cei_mp2** (seeds [7, 1234, 2025], 634 visits, 500 patients, 3 shuffles per arm, accuracy before 0.761)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -12.85 [-15.73, -10.23] (n=500) | 28.25 [25.22, 31.35] (n=500) | -0.292 [-0.309, -0.276] (n=500) | 100.0 | -0.1226 ± 0.0283 |
| drop_informative | -9.10 [-11.74, -6.50] (n=500) | 21.98 [19.27, 24.70] (n=500) | -0.179 [-0.196, -0.163] (n=500) | 100.0 | -0.0896 ± 0.0209 |
| shuffle_all | -1.42 [-2.44, -0.42] (n=500) | 5.22 [4.05, 6.53] (n=500) | -0.026 [-0.033, -0.020] (n=500) | 100.0 | -0.0150 ± 0.0072 |
| shuffle_informative | -0.60 [-1.52, +0.30] (n=500) | 3.58 [2.56, 4.75] (n=500) | -0.016 [-0.022, -0.011] (n=500) | 100.0 | -0.0057 ± 0.0125 |
| shuffle_structural | -0.46 [-1.00, +0.08] (n=500) | 1.92 [1.27, 2.64] (n=500) | -0.009 [-0.013, -0.005] (n=500) | 99.3 | -0.0040 ± 0.0045 |

### K = 20

**cei_mp2** (seeds [7, 1234, 2025], 672 visits, 500 patients, 3 shuffles per arm, accuracy before 0.697)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -11.55 [-14.30, -8.69] (n=500) | 29.57 [26.72, 32.55] (n=500) | -0.287 [-0.301, -0.273] (n=500) | 100.0 | -0.1053 ± 0.0282 |
| drop_informative | -10.65 [-13.77, -7.55] (n=500) | 27.96 [24.79, 31.07] (n=500) | -0.195 [-0.215, -0.176] (n=500) | 100.0 | -0.0978 ± 0.0264 |
| shuffle_all | -0.43 [-1.76, +0.91] (n=500) | 7.59 [6.13, 9.17] (n=500) | -0.030 [-0.037, -0.024] (n=500) | 100.0 | -0.0074 ± 0.0100 |
| shuffle_informative | +0.25 [-0.67, +1.14] (n=500) | 4.45 [3.40, 5.55] (n=500) | -0.017 [-0.023, -0.013] (n=500) | 100.0 | -0.0017 ± 0.0116 |
| shuffle_structural | -0.67 [-1.57, +0.22] (n=500) | 3.75 [2.79, 4.85] (n=500) | -0.011 [-0.015, -0.008] (n=500) | 98.5 | -0.0065 ± 0.0107 |

### K = 30

**cei_mp2** (seeds [7, 1234, 2025], 671 visits, 500 patients, 3 shuffles per arm, accuracy before 0.663)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -13.68 [-16.50, -10.85] (n=500) | 34.59 [31.68, 37.68] (n=500) | -0.347 [-0.363, -0.332] (n=500) | 100.0 | -0.1325 ± 0.0378 |
| drop_informative | -14.33 [-17.30, -11.47] (n=500) | 31.19 [28.39, 34.38] (n=500) | -0.223 [-0.241, -0.206] (n=500) | 100.0 | -0.1260 ± 0.0189 |
| shuffle_all | -1.25 [-2.35, -0.16] (n=500) | 7.38 [6.06, 8.75] (n=500) | -0.025 [-0.032, -0.019] (n=500) | 100.0 | -0.0195 ± 0.0095 |
| shuffle_informative | -0.84 [-1.71, +0.05] (n=500) | 5.04 [3.95, 6.25] (n=500) | -0.014 [-0.018, -0.009] (n=500) | 100.0 | -0.0129 ± 0.0059 |
| shuffle_structural | -0.38 [-1.03, +0.22] (n=500) | 2.87 [2.11, 3.75] (n=500) | -0.009 [-0.013, -0.006] (n=500) | 98.6 | -0.0075 ± 0.0054 |

