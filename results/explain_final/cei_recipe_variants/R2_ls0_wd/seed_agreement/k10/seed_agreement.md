# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.571 [0.558, 0.584] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.471 [0.455, 0.488] (n=500) | 0.552 [0.539, 0.565] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.446 [0.432, 0.459] (n=500) |
| GNNExplainer | 500 | 0.748 [0.731, 0.766] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.691 [0.669, 0.713] (n=500) | 0.680 [0.661, 0.702] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.871 [0.860, 0.883] (n=500) |
| GradExplainer | 500 | 0.530 [0.517, 0.543] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.421 [0.406, 0.438] (n=500) | 0.545 [0.530, 0.559] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.475 [0.461, 0.490] (n=500) |
| IntegratedGradExplainer | 500 | 0.526 [0.512, 0.540] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.416 [0.398, 0.434] (n=500) | 0.546 [0.531, 0.561] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.409 [0.393, 0.425] (n=500) |

