# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 32 patients)

Prediction changes under randomised weights: 0.979 [0.948, 1.000] (n=32); share of noise draws that change the prediction: 0.033 [0.002, 0.073] (n=32).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.064 [-0.139, 0.014] (n=32) | 0.152 [0.117, 0.186] (n=32) | 0.979 [0.969, 0.986] (n=32) | 0.943 [0.925, 0.962] (n=32) |
| GNNExplainer | 0.474 [0.403, 0.544] (n=32) | 0.302 [0.245, 0.366] (n=32) | 0.994 [0.990, 0.997] (n=32) | 0.948 [0.922, 0.969] (n=32) |
| GradExplainer | 0.213 [0.106, 0.315] (n=32) | 0.208 [0.152, 0.280] (n=32) | 0.982 [0.974, 0.988] (n=32) | 0.932 [0.907, 0.954] (n=32) |
| IntegratedGradExplainer | 0.016 [-0.079, 0.109] (n=32) | 0.159 [0.102, 0.221] (n=32) | 0.996 [0.995, 0.997] (n=32) | 0.975 [0.964, 0.985] (n=32) |

### graphcare (seeds [7, 1234, 2025], 120 audited visits, 32 patients)

Prediction changes under randomised weights: 1.000 [1.000, 1.000] (n=32); share of noise draws that change the prediction: 0.010 [0.000, 0.026] (n=32).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.041 [-0.279, 0.213] (n=14) | 0.829 [0.739, 0.910] (n=32) | 1.000 [1.000, 1.000] (n=14) | 1.000 [1.000, 1.000] (n=32) |
| GNNExplainer | 0.809 [0.725, 0.880] (n=32) | 0.619 [0.533, 0.701] (n=32) | 0.999 [0.997, 0.999] (n=32) | 0.993 [0.986, 0.999] (n=32) |
| GradExplainer | 0.116 [0.050, 0.179] (n=32) | 0.187 [0.142, 0.235] (n=32) | 0.981 [0.974, 0.988] (n=32) | 0.897 [0.867, 0.925] (n=32) |
| IntegratedGradExplainer | 0.069 [-0.017, 0.153] (n=32) | 0.182 [0.140, 0.227] (n=32) | 0.989 [0.982, 0.995] (n=32) | 0.946 [0.904, 0.973] (n=32) |

### gsat (seeds [7, 1234, 2025], 120 audited visits, 32 patients)

Prediction changes under randomised weights: 0.891 [0.781, 0.984] (n=32); share of noise draws that change the prediction: 0.004 [0.000, 0.011] (n=32).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | 0.019 [-0.073, 0.111] (n=32) | 0.252 [0.188, 0.323] (n=32) | 0.992 [0.989, 0.994] (n=32) | 0.968 [0.953, 0.981] (n=32) |
| GNNExplainer | 0.438 [0.382, 0.501] (n=32) | 0.291 [0.234, 0.369] (n=32) | 0.973 [0.965, 0.980] (n=32) | 0.881 [0.851, 0.907] (n=32) |
| GradExplainer | 0.137 [0.073, 0.204] (n=32) | 0.199 [0.161, 0.240] (n=32) | 0.969 [0.964, 0.973] (n=32) | 0.873 [0.845, 0.898] (n=32) |
| IntegratedGradExplainer | 0.053 [-0.002, 0.109] (n=32) | 0.180 [0.141, 0.225] (n=32) | 0.990 [0.986, 0.993] (n=32) | 0.949 [0.934, 0.965] (n=32) |

### protgnn (seeds [7, 1234, 2025], 120 audited visits, 32 patients)

Prediction changes under randomised weights: 0.958 [0.906, 1.000] (n=32); share of noise draws that change the prediction: 0.008 [0.000, 0.021] (n=32).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | 0.017 [-0.034, 0.082] (n=27) | 0.083 [0.037, 0.138] (n=32) | 0.929 [0.850, 0.979] (n=32) | 0.948 [0.914, 0.976] (n=32) |
| GNNExplainer | 0.649 [0.596, 0.698] (n=32) | 0.364 [0.290, 0.451] (n=32) | 0.993 [0.991, 0.995] (n=32) | 0.951 [0.931, 0.970] (n=32) |
| GradExplainer | 0.024 [-0.055, 0.103] (n=32) | 0.128 [0.088, 0.174] (n=32) | 0.992 [0.989, 0.995] (n=32) | 0.920 [0.897, 0.941] (n=32) |
| IntegratedGradExplainer | 0.112 [0.039, 0.178] (n=32) | 0.129 [0.097, 0.159] (n=32) | 0.996 [0.994, 0.997] (n=32) | 0.964 [0.951, 0.977] (n=32) |

