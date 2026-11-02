# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.619 [0.606, 0.633] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.531 [0.514, 0.548] (n=500) | 0.586 [0.573, 0.600] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.441 [0.428, 0.453] (n=500) |
| GNNExplainer | 500 | 0.638 [0.622, 0.654] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.555 [0.536, 0.575] (n=500) | 0.551 [0.534, 0.568] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.811 [0.799, 0.824] (n=500) |
| GradExplainer | 500 | 0.648 [0.635, 0.661] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.566 [0.550, 0.582] (n=500) | 0.622 [0.609, 0.633] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.603 [0.589, 0.618] (n=500) |
| IntegratedGradExplainer | 500 | 0.617 [0.603, 0.631] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.528 [0.511, 0.545] (n=500) | 0.618 [0.605, 0.631] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.511 [0.494, 0.527] (n=500) |

