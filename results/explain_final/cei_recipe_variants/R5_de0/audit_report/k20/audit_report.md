# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 31 patients)

Prediction changes under randomised weights: 1.000 [1.000, 1.000] (n=31); share of noise draws that change the prediction: 0.038 [0.005, 0.081] (n=31).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.112 [-0.186, -0.035] (n=31) | 0.116 [0.075, 0.165] (n=31) | 0.974 [0.959, 0.986] (n=31) | 0.927 [0.903, 0.950] (n=31) |
| GNNExplainer | 0.462 [0.368, 0.554] (n=31) | 0.290 [0.228, 0.354] (n=31) | 0.992 [0.988, 0.995] (n=31) | 0.960 [0.935, 0.980] (n=31) |
| GradExplainer | 0.196 [0.100, 0.289] (n=31) | 0.175 [0.133, 0.226] (n=31) | 0.986 [0.982, 0.989] (n=31) | 0.914 [0.890, 0.936] (n=31) |
| IntegratedGradExplainer | -0.016 [-0.117, 0.075] (n=31) | 0.121 [0.079, 0.176] (n=31) | 0.996 [0.994, 0.997] (n=31) | 0.966 [0.949, 0.981] (n=31) |

