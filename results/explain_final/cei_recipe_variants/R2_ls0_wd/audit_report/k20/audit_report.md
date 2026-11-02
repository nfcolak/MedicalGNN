# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 31 patients)

Prediction changes under randomised weights: 1.000 [1.000, 1.000] (n=31); share of noise draws that change the prediction: 0.024 [0.000, 0.065] (n=31).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.061 [-0.140, 0.011] (n=31) | 0.136 [0.095, 0.185] (n=31) | 0.977 [0.964, 0.986] (n=31) | 0.944 [0.922, 0.965] (n=31) |
| GNNExplainer | 0.531 [0.451, 0.614] (n=31) | 0.338 [0.268, 0.418] (n=31) | 0.997 [0.995, 0.999] (n=31) | 0.978 [0.963, 0.990] (n=31) |
| GradExplainer | 0.201 [0.104, 0.301] (n=31) | 0.213 [0.155, 0.288] (n=31) | 0.982 [0.977, 0.987] (n=31) | 0.912 [0.883, 0.936] (n=31) |
| IntegratedGradExplainer | -0.002 [-0.107, 0.099] (n=31) | 0.122 [0.080, 0.177] (n=31) | 0.997 [0.995, 0.997] (n=31) | 0.951 [0.931, 0.969] (n=31) |

