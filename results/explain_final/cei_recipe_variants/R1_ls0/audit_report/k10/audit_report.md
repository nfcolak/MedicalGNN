# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 33 patients)

Prediction changes under randomised weights: 0.949 [0.899, 0.990] (n=33); share of noise draws that change the prediction: 0.017 [0.000, 0.049] (n=33).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | 0.016 [-0.065, 0.100] (n=33) | 0.171 [0.120, 0.230] (n=33) | 0.978 [0.961, 0.990] (n=33) | 0.949 [0.930, 0.966] (n=33) |
| GNNExplainer | 0.505 [0.425, 0.578] (n=33) | 0.388 [0.310, 0.476] (n=33) | 0.997 [0.994, 0.999] (n=33) | 0.984 [0.975, 0.993] (n=33) |
| GradExplainer | 0.207 [0.119, 0.297] (n=33) | 0.221 [0.159, 0.300] (n=33) | 0.989 [0.987, 0.991] (n=33) | 0.919 [0.898, 0.940] (n=33) |
| IntegratedGradExplainer | 0.023 [-0.063, 0.108] (n=33) | 0.129 [0.102, 0.158] (n=33) | 0.997 [0.996, 0.998] (n=33) | 0.968 [0.954, 0.981] (n=33) |

