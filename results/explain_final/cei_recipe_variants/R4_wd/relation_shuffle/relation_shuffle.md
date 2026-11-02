# Relation shuffling of the trained final models

Patient-level bootstrap, 2000 resamples, intervals withheld below 10 patients. The models were never trained on shuffled graphs: read a change as dependence on the relations, not as proof of meaningful use.

## Effect of changing the relations of the trained models

Accuracy and probability changes are after minus before, per patient (mean over visits, seeds and shuffles), patient-level 95% interval. Flip = the predicted class changes. Moved/removed = share of visits where at least one edge was replaced (shuffle) or the graph lost edges (drop). Macro-F1 change: mean ± SD over seeds.

### K = 10

**cei_mp2** (seeds [7, 1234, 2025], 634 visits, 500 patients, 3 shuffles per arm, accuracy before 0.758)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -11.76 [-14.45, -9.21] (n=500) | 26.68 [23.70, 29.66] (n=500) | -0.270 [-0.283, -0.258] (n=500) | 100.0 | -0.1125 ± 0.0239 |
| drop_informative | -6.91 [-9.37, -4.46] (n=500) | 19.94 [17.28, 22.67] (n=500) | -0.172 [-0.186, -0.159] (n=500) | 100.0 | -0.0630 ± 0.0076 |
| shuffle_all | -1.21 [-2.37, -0.12] (n=500) | 6.49 [5.18, 7.92] (n=500) | -0.032 [-0.039, -0.026] (n=500) | 100.0 | -0.0151 ± 0.0198 |
| shuffle_informative | -0.39 [-1.47, +0.60] (n=500) | 4.89 [3.67, 6.19] (n=500) | -0.022 [-0.027, -0.016] (n=500) | 100.0 | -0.0078 ± 0.0137 |
| shuffle_structural | -0.82 [-1.31, -0.33] (n=500) | 2.12 [1.42, 2.87] (n=500) | -0.010 [-0.014, -0.007] (n=500) | 99.3 | -0.0074 ± 0.0085 |

### K = 20

**cei_mp2** (seeds [7, 1234, 2025], 672 visits, 500 patients, 3 shuffles per arm, accuracy before 0.697)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -8.87 [-11.52, -6.21] (n=500) | 26.66 [23.79, 29.54] (n=500) | -0.261 [-0.274, -0.248] (n=500) | 100.0 | -0.0909 ± 0.0310 |
| drop_informative | -10.64 [-13.54, -7.60] (n=500) | 28.10 [25.05, 31.26] (n=500) | -0.178 [-0.195, -0.161] (n=500) | 100.0 | -0.0970 ± 0.0059 |
| shuffle_all | -0.22 [-1.52, +0.97] (n=500) | 7.51 [6.11, 9.06] (n=500) | -0.035 [-0.041, -0.028] (n=500) | 100.0 | -0.0078 ± 0.0031 |
| shuffle_informative | -0.38 [-1.54, +0.74] (n=500) | 5.51 [4.28, 6.87] (n=500) | -0.022 [-0.027, -0.017] (n=500) | 100.0 | -0.0093 ± 0.0067 |
| shuffle_structural | -0.15 [-0.83, +0.49] (n=500) | 3.19 [2.35, 4.16] (n=500) | -0.011 [-0.015, -0.007] (n=500) | 98.5 | +0.0012 ± 0.0040 |

### K = 30

**cei_mp2** (seeds [7, 1234, 2025], 671 visits, 500 patients, 3 shuffles per arm, accuracy before 0.653)

| arm | Δ accuracy (pp) | flip rate (%) | Δ probability of predicted class | moved / removed (%) | Δ macro-F1 |
|---|---|---|---|---|---|
| drop_all | -9.91 [-12.51, -7.33] (n=500) | 31.26 [28.44, 34.43] (n=500) | -0.323 [-0.336, -0.310] (n=500) | 100.0 | -0.1010 ± 0.0401 |
| drop_informative | -9.55 [-12.48, -6.79] (n=500) | 28.58 [25.79, 31.61] (n=500) | -0.219 [-0.235, -0.204] (n=500) | 100.0 | -0.0859 ± 0.0255 |
| shuffle_all | -1.06 [-2.46, +0.26] (n=500) | 9.58 [7.98, 11.27] (n=500) | -0.036 [-0.043, -0.029] (n=500) | 100.0 | -0.0102 ± 0.0212 |
| shuffle_informative | -0.44 [-1.60, +0.70] (n=500) | 6.53 [5.16, 7.96] (n=500) | -0.022 [-0.028, -0.017] (n=500) | 100.0 | -0.0036 ± 0.0151 |
| shuffle_structural | -0.48 [-1.26, +0.26] (n=500) | 3.44 [2.51, 4.50] (n=500) | -0.010 [-0.013, -0.007] (n=500) | 98.6 | -0.0016 ± 0.0084 |

