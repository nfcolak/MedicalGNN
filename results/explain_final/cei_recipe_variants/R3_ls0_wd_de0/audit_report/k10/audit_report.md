# Controls on the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Weight randomisation and stability of the explanations

Weight randomisation: agreement between the real explanation and the one from a model with re-drawn weights (low is what a faithful explanation should show). Stability: agreement between the explanation and the one after small noise on the measured values, on draws where the prediction does not change (high is good).

### cei_mp2 (seeds [7, 1234, 2025], 120 audited visits, 33 patients)

Prediction changes under randomised weights: 0.965 [0.919, 1.000] (n=33); share of noise draws that change the prediction: 0.029 [0.002, 0.075] (n=33).

| source | Spearman, real vs randomised | top-k Jaccard, real vs randomised | Spearman, original vs perturbed | top-k Jaccard, original vs perturbed |
|---|---|---|---|---|
| builtin | 0.016 [-0.062, 0.096] (n=33) | 0.182 [0.125, 0.245] (n=33) | 0.979 [0.962, 0.990] (n=33) | 0.962 [0.945, 0.977] (n=33) |
| GNNExplainer | 0.513 [0.433, 0.590] (n=33) | 0.397 [0.319, 0.488] (n=33) | 0.995 [0.991, 0.998] (n=33) | 0.970 [0.943, 0.990] (n=33) |
| GradExplainer | 0.206 [0.129, 0.285] (n=33) | 0.216 [0.159, 0.287] (n=33) | 0.986 [0.982, 0.989] (n=33) | 0.913 [0.889, 0.936] (n=33) |
| IntegratedGradExplainer | 0.038 [-0.046, 0.118] (n=33) | 0.130 [0.100, 0.162] (n=33) | 0.997 [0.996, 0.998] (n=33) | 0.964 [0.951, 0.975] (n=33) |

