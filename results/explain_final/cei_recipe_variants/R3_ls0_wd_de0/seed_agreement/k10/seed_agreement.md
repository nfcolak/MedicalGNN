# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.564 [0.551, 0.577] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.463 [0.447, 0.479] (n=500) | 0.543 [0.530, 0.556] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.432 [0.418, 0.445] (n=500) |
| GNNExplainer | 500 | 0.730 [0.712, 0.747] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.668 [0.647, 0.690] (n=500) | 0.662 [0.643, 0.681] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.863 [0.851, 0.875] (n=500) |
| GradExplainer | 500 | 0.534 [0.520, 0.547] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.426 [0.409, 0.441] (n=500) | 0.539 [0.524, 0.553] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.481 [0.467, 0.495] (n=500) |
| IntegratedGradExplainer | 500 | 0.525 [0.510, 0.540] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.414 [0.396, 0.433] (n=500) | 0.528 [0.513, 0.543] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.407 [0.393, 0.421] (n=500) |

