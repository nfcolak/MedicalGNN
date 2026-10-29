# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 31 patients)

Prediction changes under randomised weights: 1.000 [1.000, 1.000] (n=31); share of noise draws that change the prediction: 0.053 [0.021, 0.093] (n=31).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.086 [-0.157, -0.013] (n=31) | 0.126 [0.081, 0.176] (n=31) | 0.978 [0.968, 0.986] (n=31) | 0.944 [0.916, 0.968] (n=31) |
| GNNExplainer | 0.483 [0.401, 0.565] (n=31) | 0.299 [0.233, 0.370] (n=31) | 0.995 [0.992, 0.997] (n=31) | 0.974 [0.958, 0.987] (n=31) |
| GradExplainer | 0.206 [0.110, 0.303] (n=31) | 0.192 [0.145, 0.250] (n=31) | 0.984 [0.980, 0.988] (n=31) | 0.920 [0.899, 0.941] (n=31) |
| IntegratedGradExplainer | -0.017 [-0.121, 0.080] (n=31) | 0.128 [0.087, 0.181] (n=31) | 0.996 [0.995, 0.997] (n=31) | 0.964 [0.950, 0.978] (n=31) |

### graphcare (seeds [7, 1234, 2025], 120 audited visits, 31 patients)

Prediction changes under randomised weights: 0.961 [0.900, 1.000] (n=31); share of noise draws that change the prediction: 0.032 [0.000, 0.075] (n=31).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.070 [-0.234, 0.098] (n=14) | 0.797 [0.700, 0.886] (n=31) | 1.000 [1.000, 1.000] (n=14) | 1.000 [1.000, 1.000] (n=31) |
| GNNExplainer | 0.766 [0.714, 0.819] (n=31) | 0.518 [0.448, 0.593] (n=31) | 0.997 [0.994, 0.999] (n=31) | 0.981 [0.970, 0.990] (n=31) |
| GradExplainer | 0.147 [0.062, 0.234] (n=31) | 0.214 [0.162, 0.271] (n=31) | 0.981 [0.972, 0.988] (n=31) | 0.932 [0.909, 0.953] (n=31) |
| IntegratedGradExplainer | 0.028 [-0.037, 0.085] (n=31) | 0.180 [0.136, 0.226] (n=31) | 0.994 [0.990, 0.996] (n=31) | 0.961 [0.944, 0.977] (n=31) |

### gsat (seeds [7, 1234, 2025], 120 audited visits, 31 patients)

Prediction changes under randomised weights: 0.930 [0.860, 0.984] (n=31); share of noise draws that change the prediction: 0.022 [0.000, 0.054] (n=31).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.048 [-0.125, 0.032] (n=31) | 0.223 [0.168, 0.279] (n=31) | 0.991 [0.983, 0.995] (n=31) | 0.955 [0.928, 0.976] (n=31) |
| GNNExplainer | 0.440 [0.392, 0.489] (n=31) | 0.336 [0.287, 0.384] (n=31) | 0.980 [0.975, 0.984] (n=31) | 0.906 [0.883, 0.927] (n=31) |
| GradExplainer | 0.116 [0.029, 0.201] (n=31) | 0.226 [0.178, 0.279] (n=31) | 0.972 [0.964, 0.978] (n=31) | 0.863 [0.828, 0.896] (n=31) |
| IntegratedGradExplainer | 0.020 [-0.058, 0.095] (n=31) | 0.191 [0.150, 0.236] (n=31) | 0.992 [0.989, 0.995] (n=31) | 0.937 [0.917, 0.957] (n=31) |

### protgnn (seeds [7, 1234, 2025], 120 audited visits, 31 patients)

Prediction changes under randomised weights: 0.957 [0.882, 1.000] (n=31); share of noise draws that change the prediction: 0.027 [0.000, 0.059] (n=31).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.049 [-0.071, -0.024] (n=26) | 0.048 [0.016, 0.086] (n=31) | 0.964 [0.933, 0.987] (n=31) | 0.937 [0.901, 0.970] (n=31) |
| GNNExplainer | 0.662 [0.608, 0.717] (n=31) | 0.355 [0.285, 0.432] (n=31) | 0.992 [0.988, 0.995] (n=31) | 0.949 [0.931, 0.966] (n=31) |
| GradExplainer | 0.024 [-0.059, 0.105] (n=31) | 0.138 [0.103, 0.177] (n=31) | 0.994 [0.992, 0.995] (n=31) | 0.943 [0.926, 0.959] (n=31) |
| IntegratedGradExplainer | 0.137 [0.068, 0.203] (n=31) | 0.174 [0.130, 0.218] (n=31) | 0.996 [0.995, 0.997] (n=31) | 0.971 [0.959, 0.982] (n=31) |

