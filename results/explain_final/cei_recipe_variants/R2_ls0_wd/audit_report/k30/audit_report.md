# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 32 patients)

Prediction changes under randomised weights: 0.958 [0.891, 1.000] (n=32); share of noise draws that change the prediction: 0.013 [0.000, 0.029] (n=32).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.086 [-0.157, -0.017] (n=32) | 0.134 [0.101, 0.171] (n=32) | 0.981 [0.968, 0.989] (n=32) | 0.927 [0.898, 0.954] (n=32) |
| GNNExplainer | 0.490 [0.426, 0.558] (n=32) | 0.366 [0.298, 0.445] (n=32) | 0.997 [0.994, 0.998] (n=32) | 0.974 [0.959, 0.987] (n=32) |
| GradExplainer | 0.222 [0.118, 0.323] (n=32) | 0.225 [0.163, 0.301] (n=32) | 0.981 [0.970, 0.988] (n=32) | 0.934 [0.911, 0.955] (n=32) |
| IntegratedGradExplainer | 0.023 [-0.076, 0.121] (n=32) | 0.157 [0.103, 0.216] (n=32) | 0.997 [0.996, 0.997] (n=32) | 0.959 [0.940, 0.974] (n=32) |

