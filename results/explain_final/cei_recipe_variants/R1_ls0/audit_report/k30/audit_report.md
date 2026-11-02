# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 32 patients)

Prediction changes under randomised weights: 0.964 [0.911, 1.000] (n=32); share of noise draws that change the prediction: 0.039 [0.003, 0.091] (n=32).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.073 [-0.147, 0.001] (n=32) | 0.140 [0.106, 0.178] (n=32) | 0.977 [0.961, 0.989] (n=32) | 0.952 [0.927, 0.973] (n=32) |
| GNNExplainer | 0.500 [0.428, 0.574] (n=32) | 0.347 [0.283, 0.419] (n=32) | 0.996 [0.993, 0.998] (n=32) | 0.979 [0.961, 0.993] (n=32) |
| GradExplainer | 0.221 [0.117, 0.324] (n=32) | 0.242 [0.179, 0.319] (n=32) | 0.980 [0.970, 0.988] (n=32) | 0.926 [0.900, 0.949] (n=32) |
| IntegratedGradExplainer | 0.006 [-0.104, 0.110] (n=32) | 0.157 [0.104, 0.215] (n=32) | 0.997 [0.997, 0.998] (n=32) | 0.964 [0.947, 0.979] (n=32) |

