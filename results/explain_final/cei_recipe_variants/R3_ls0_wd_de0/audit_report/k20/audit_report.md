# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 31 patients)

Prediction changes under randomised weights: 0.989 [0.968, 1.000] (n=31); share of noise draws that change the prediction: 0.016 [0.000, 0.032] (n=31).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.049 [-0.123, 0.023] (n=31) | 0.117 [0.082, 0.161] (n=31) | 0.978 [0.967, 0.987] (n=31) | 0.931 [0.911, 0.951] (n=31) |
| GNNExplainer | 0.508 [0.425, 0.592] (n=31) | 0.322 [0.260, 0.392] (n=31) | 0.993 [0.988, 0.997] (n=31) | 0.963 [0.941, 0.980] (n=31) |
| GradExplainer | 0.191 [0.099, 0.288] (n=31) | 0.204 [0.149, 0.277] (n=31) | 0.980 [0.972, 0.986] (n=31) | 0.916 [0.897, 0.936] (n=31) |
| IntegratedGradExplainer | 0.005 [-0.097, 0.102] (n=31) | 0.129 [0.085, 0.184] (n=31) | 0.997 [0.997, 0.998] (n=31) | 0.953 [0.933, 0.971] (n=31) |

