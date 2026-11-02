# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 33 patients)

Prediction changes under randomised weights: 0.960 [0.909, 1.000] (n=33); share of noise draws that change the prediction: 0.029 [0.000, 0.076] (n=33).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.011 [-0.087, 0.065] (n=33) | 0.159 [0.110, 0.217] (n=33) | 0.985 [0.975, 0.992] (n=33) | 0.960 [0.943, 0.976] (n=33) |
| GNNExplainer | 0.489 [0.426, 0.551] (n=33) | 0.349 [0.282, 0.430] (n=33) | 0.995 [0.989, 0.998] (n=33) | 0.976 [0.964, 0.988] (n=33) |
| GradExplainer | 0.227 [0.150, 0.309] (n=33) | 0.198 [0.153, 0.252] (n=33) | 0.987 [0.983, 0.990] (n=33) | 0.903 [0.869, 0.931] (n=33) |
| IntegratedGradExplainer | 0.016 [-0.075, 0.108] (n=33) | 0.140 [0.105, 0.178] (n=33) | 0.995 [0.993, 0.997] (n=33) | 0.976 [0.965, 0.986] (n=33) |

