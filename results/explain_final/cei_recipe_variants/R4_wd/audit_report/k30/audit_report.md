# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 32 patients)

Prediction changes under randomised weights: 0.964 [0.917, 1.000] (n=32); share of noise draws that change the prediction: 0.036 [0.000, 0.104] (n=32).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | -0.057 [-0.131, 0.020] (n=32) | 0.151 [0.114, 0.189] (n=32) | 0.982 [0.974, 0.988] (n=31) | 0.948 [0.930, 0.965] (n=31) |
| GNNExplainer | 0.486 [0.420, 0.553] (n=32) | 0.292 [0.240, 0.349] (n=32) | 0.994 [0.992, 0.997] (n=31) | 0.962 [0.947, 0.977] (n=31) |
| GradExplainer | 0.224 [0.121, 0.326] (n=32) | 0.210 [0.157, 0.268] (n=32) | 0.982 [0.975, 0.988] (n=31) | 0.911 [0.877, 0.941] (n=31) |
| IntegratedGradExplainer | -0.005 [-0.107, 0.094] (n=32) | 0.166 [0.107, 0.228] (n=32) | 0.995 [0.992, 0.997] (n=31) | 0.971 [0.958, 0.982] (n=31) |

