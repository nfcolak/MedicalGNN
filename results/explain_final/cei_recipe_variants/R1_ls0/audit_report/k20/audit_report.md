# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 31 patients)

Prediction changes under randomised weights: 1.000 [1.000, 1.000] (n=31); share of noise draws that change the prediction: 0.035 [0.008, 0.073] (n=31).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.086 [-0.172, -0.003] (n=31) | 0.127 [0.081, 0.181] (n=31) | 0.974 [0.952, 0.988] (n=31) | 0.926 [0.899, 0.950] (n=31) |
| GNNExplainer | 0.507 [0.419, 0.596] (n=31) | 0.307 [0.239, 0.384] (n=31) | 0.992 [0.987, 0.996] (n=31) | 0.958 [0.939, 0.977] (n=31) |
| GradExplainer | 0.196 [0.100, 0.293] (n=31) | 0.206 [0.149, 0.282] (n=31) | 0.984 [0.980, 0.988] (n=31) | 0.900 [0.878, 0.924] (n=31) |
| IntegratedGradExplainer | -0.003 [-0.106, 0.098] (n=31) | 0.123 [0.081, 0.178] (n=31) | 0.997 [0.996, 0.998] (n=31) | 0.972 [0.961, 0.983] (n=31) |

