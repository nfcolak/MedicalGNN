# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 33 patients)

Prediction changes under randomised weights: 0.960 [0.909, 1.000] (n=33); share of noise draws that change the prediction: 0.012 [0.000, 0.027] (n=33).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.033 [-0.118, 0.055] (n=33) | 0.164 [0.116, 0.216] (n=33) | 0.982 [0.973, 0.989] (n=33) | 0.960 [0.945, 0.976] (n=33) |
| GNNExplainer | 0.440 [0.358, 0.519] (n=33) | 0.287 [0.219, 0.365] (n=33) | 0.994 [0.991, 0.996] (n=33) | 0.960 [0.939, 0.978] (n=33) |
| GradExplainer | 0.193 [0.112, 0.278] (n=33) | 0.198 [0.146, 0.268] (n=33) | 0.982 [0.973, 0.989] (n=33) | 0.898 [0.861, 0.930] (n=33) |
| IntegratedGradExplainer | 0.013 [-0.073, 0.102] (n=33) | 0.132 [0.098, 0.171] (n=33) | 0.997 [0.996, 0.997] (n=33) | 0.959 [0.942, 0.975] (n=33) |

