# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 32 patients)

Prediction changes under randomised weights: 0.979 [0.948, 1.000] (n=32); share of noise draws that change the prediction: 0.029 [0.000, 0.062] (n=32).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.072 [-0.150, 0.007] (n=32) | 0.136 [0.103, 0.171] (n=32) | 0.978 [0.968, 0.987] (n=32) | 0.936 [0.907, 0.960] (n=32) |
| GNNExplainer | 0.492 [0.422, 0.565] (n=32) | 0.375 [0.301, 0.456] (n=32) | 0.995 [0.992, 0.997] (n=32) | 0.974 [0.962, 0.985] (n=32) |
| GradExplainer | 0.190 [0.083, 0.294] (n=32) | 0.212 [0.161, 0.268] (n=32) | 0.979 [0.967, 0.987] (n=32) | 0.919 [0.891, 0.946] (n=32) |
| IntegratedGradExplainer | 0.020 [-0.089, 0.126] (n=32) | 0.168 [0.109, 0.238] (n=32) | 0.996 [0.994, 0.997] (n=32) | 0.974 [0.962, 0.984] (n=32) |

