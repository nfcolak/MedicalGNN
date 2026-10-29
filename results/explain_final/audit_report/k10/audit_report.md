# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 33 patients)

Prediction changes under randomised weights: 0.960 [0.914, 1.000] (n=33); share of noise draws that change the prediction: 0.021 [0.005, 0.040] (n=33).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.030 [-0.113, 0.051] (n=33) | 0.163 [0.111, 0.223] (n=33) | 0.982 [0.970, 0.990] (n=33) | 0.959 [0.941, 0.977] (n=33) |
| GNNExplainer | 0.480 [0.419, 0.541] (n=33) | 0.313 [0.249, 0.388] (n=33) | 0.996 [0.994, 0.998] (n=33) | 0.970 [0.956, 0.982] (n=33) |
| GradExplainer | 0.209 [0.133, 0.290] (n=33) | 0.190 [0.149, 0.238] (n=33) | 0.986 [0.981, 0.990] (n=33) | 0.897 [0.863, 0.926] (n=33) |
| IntegratedGradExplainer | 0.008 [-0.072, 0.094] (n=33) | 0.130 [0.097, 0.169] (n=33) | 0.997 [0.995, 0.998] (n=33) | 0.975 [0.962, 0.986] (n=33) |

### graphcare (seeds [7, 1234, 2025], 120 audited visits, 33 patients)

Prediction changes under randomised weights: 0.929 [0.838, 1.000] (n=33); share of noise draws that change the prediction: 0.035 [0.000, 0.086] (n=33).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | 0.006 [-0.417, 0.424] (n=12) | 0.826 [0.708, 0.922] (n=33) | 0.998 [0.994, 1.000] (n=12) | 0.996 [0.990, 1.000] (n=33) |
| GNNExplainer | 0.747 [0.679, 0.809] (n=33) | 0.596 [0.507, 0.687] (n=33) | 1.000 [0.999, 1.000] (n=33) | 0.994 [0.988, 0.998] (n=33) |
| GradExplainer | 0.147 [0.069, 0.228] (n=33) | 0.206 [0.157, 0.263] (n=33) | 0.981 [0.975, 0.986] (n=33) | 0.891 [0.851, 0.926] (n=33) |
| IntegratedGradExplainer | 0.107 [0.030, 0.180] (n=33) | 0.199 [0.137, 0.272] (n=33) | 0.989 [0.981, 0.995] (n=33) | 0.922 [0.879, 0.956] (n=33) |

### gsat (seeds [7, 1234, 2025], 120 audited visits, 33 patients)

Prediction changes under randomised weights: 0.939 [0.848, 1.000] (n=33); share of noise draws that change the prediction: 0.010 [0.000, 0.025] (n=33).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | 0.062 [-0.032, 0.156] (n=33) | 0.224 [0.169, 0.285] (n=33) | 0.992 [0.987, 0.995] (n=33) | 0.941 [0.912, 0.963] (n=33) |
| GNNExplainer | 0.527 [0.460, 0.594] (n=33) | 0.325 [0.268, 0.396] (n=33) | 0.988 [0.984, 0.992] (n=33) | 0.949 [0.932, 0.965] (n=33) |
| GradExplainer | 0.095 [0.022, 0.162] (n=33) | 0.174 [0.128, 0.227] (n=33) | 0.977 [0.974, 0.981] (n=33) | 0.893 [0.869, 0.919] (n=33) |
| IntegratedGradExplainer | -0.003 [-0.082, 0.066] (n=33) | 0.162 [0.116, 0.216] (n=33) | 0.992 [0.988, 0.995] (n=33) | 0.935 [0.912, 0.955] (n=33) |

### protgnn (seeds [7, 1234, 2025], 120 audited visits, 33 patients)

Prediction changes under randomised weights: 0.955 [0.879, 1.000] (n=33); share of noise draws that change the prediction: 0.010 [0.000, 0.025] (n=33).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.006 [-0.048, 0.046] (n=31) | 0.038 [0.012, 0.072] (n=33) | 0.961 [0.939, 0.979] (n=33) | 0.942 [0.916, 0.965] (n=33) |
| GNNExplainer | 0.737 [0.689, 0.787] (n=33) | 0.453 [0.376, 0.539] (n=33) | 0.996 [0.993, 0.997] (n=33) | 0.972 [0.956, 0.985] (n=33) |
| GradExplainer | 0.228 [0.160, 0.291] (n=33) | 0.186 [0.132, 0.248] (n=33) | 0.986 [0.979, 0.992] (n=33) | 0.903 [0.866, 0.934] (n=33) |
| IntegratedGradExplainer | 0.217 [0.144, 0.291] (n=33) | 0.204 [0.135, 0.279] (n=33) | 0.990 [0.985, 0.995] (n=33) | 0.935 [0.902, 0.960] (n=33) |

