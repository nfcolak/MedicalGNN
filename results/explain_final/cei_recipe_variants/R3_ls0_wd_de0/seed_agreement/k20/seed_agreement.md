# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.577 [0.564, 0.590] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.479 [0.463, 0.495] (n=500) | 0.576 [0.561, 0.590] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.429 [0.417, 0.440] (n=500) |
| GNNExplainer | 500 | 0.696 [0.679, 0.713] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.627 [0.606, 0.648] (n=500) | 0.634 [0.616, 0.652] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.841 [0.828, 0.853] (n=500) |
| GradExplainer | 500 | 0.587 [0.574, 0.600] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.491 [0.476, 0.507] (n=500) | 0.584 [0.570, 0.598] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.545 [0.531, 0.559] (n=500) |
| IntegratedGradExplainer | 500 | 0.573 [0.560, 0.586] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.474 [0.458, 0.490] (n=500) | 0.581 [0.567, 0.595] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.489 [0.473, 0.505] (n=500) |

