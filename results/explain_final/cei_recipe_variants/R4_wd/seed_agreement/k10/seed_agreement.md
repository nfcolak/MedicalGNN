# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.559 [0.547, 0.572] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.457 [0.441, 0.473] (n=500) | 0.530 [0.517, 0.544] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.366 [0.354, 0.378] (n=500) |
| GNNExplainer | 500 | 0.669 [0.651, 0.686] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.594 [0.573, 0.614] (n=500) | 0.591 [0.572, 0.608] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.837 [0.826, 0.849] (n=500) |
| GradExplainer | 500 | 0.533 [0.520, 0.547] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.425 [0.408, 0.443] (n=500) | 0.517 [0.503, 0.530] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.477 [0.460, 0.494] (n=500) |
| IntegratedGradExplainer | 500 | 0.546 [0.531, 0.561] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.440 [0.423, 0.459] (n=500) | 0.525 [0.510, 0.539] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.442 [0.424, 0.460] (n=500) |

