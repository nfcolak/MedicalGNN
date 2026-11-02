# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 33 patients)

Prediction changes under randomised weights: 0.955 [0.904, 0.995] (n=33); share of noise draws that change the prediction: 0.022 [0.000, 0.059] (n=33).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | 0.016 [-0.071, 0.105] (n=33) | 0.165 [0.114, 0.223] (n=33) | 0.978 [0.960, 0.990] (n=33) | 0.953 [0.936, 0.970] (n=33) |
| GNNExplainer | 0.497 [0.415, 0.572] (n=33) | 0.405 [0.320, 0.504] (n=33) | 0.995 [0.991, 0.998] (n=33) | 0.984 [0.968, 0.994] (n=33) |
| GradExplainer | 0.204 [0.119, 0.293] (n=33) | 0.222 [0.163, 0.298] (n=33) | 0.987 [0.985, 0.989] (n=33) | 0.916 [0.894, 0.937] (n=33) |
| IntegratedGradExplainer | 0.015 [-0.073, 0.104] (n=33) | 0.120 [0.091, 0.151] (n=33) | 0.996 [0.995, 0.997] (n=33) | 0.963 [0.946, 0.977] (n=33) |

