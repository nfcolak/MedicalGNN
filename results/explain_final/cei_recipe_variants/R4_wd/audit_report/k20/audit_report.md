# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 31 patients)

Prediction changes under randomised weights: 1.000 [1.000, 1.000] (n=31); share of noise draws that change the prediction: 0.051 [0.013, 0.105] (n=31).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.096 [-0.166, -0.023] (n=31) | 0.124 [0.084, 0.169] (n=31) | 0.982 [0.975, 0.987] (n=31) | 0.952 [0.931, 0.969] (n=31) |
| GNNExplainer | 0.484 [0.398, 0.572] (n=31) | 0.299 [0.219, 0.386] (n=31) | 0.992 [0.988, 0.996] (n=31) | 0.956 [0.939, 0.971] (n=31) |
| GradExplainer | 0.219 [0.124, 0.316] (n=31) | 0.207 [0.154, 0.272] (n=31) | 0.986 [0.983, 0.989] (n=31) | 0.909 [0.886, 0.931] (n=31) |
| IntegratedGradExplainer | -0.003 [-0.109, 0.097] (n=31) | 0.125 [0.084, 0.179] (n=31) | 0.997 [0.996, 0.997] (n=31) | 0.949 [0.933, 0.965] (n=31) |

