# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 32 patients)

Prediction changes under randomised weights: 0.969 [0.917, 1.000] (n=32); share of noise draws that change the prediction: 0.047 [0.016, 0.089] (n=32).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.083 [-0.151, -0.010] (n=32) | 0.140 [0.109, 0.173] (n=32) | 0.984 [0.979, 0.989] (n=32) | 0.944 [0.926, 0.961] (n=32) |
| GNNExplainer | 0.467 [0.397, 0.539] (n=32) | 0.343 [0.269, 0.427] (n=32) | 0.996 [0.993, 0.998] (n=32) | 0.979 [0.963, 0.991] (n=32) |
| GradExplainer | 0.216 [0.119, 0.315] (n=32) | 0.216 [0.158, 0.289] (n=32) | 0.987 [0.983, 0.990] (n=32) | 0.944 [0.924, 0.965] (n=32) |
| IntegratedGradExplainer | -0.036 [-0.140, 0.063] (n=32) | 0.169 [0.111, 0.238] (n=32) | 0.996 [0.996, 0.997] (n=32) | 0.970 [0.953, 0.984] (n=32) |

