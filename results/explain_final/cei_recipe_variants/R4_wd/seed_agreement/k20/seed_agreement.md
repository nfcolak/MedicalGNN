# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.572 [0.559, 0.585] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.473 [0.457, 0.488] (n=500) | 0.573 [0.559, 0.586] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.418 [0.405, 0.431] (n=500) |
| GNNExplainer | 500 | 0.647 [0.630, 0.664] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.567 [0.545, 0.587] (n=500) | 0.579 [0.562, 0.596] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.811 [0.797, 0.824] (n=500) |
| GradExplainer | 500 | 0.602 [0.590, 0.615] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.510 [0.495, 0.527] (n=500) | 0.604 [0.591, 0.617] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.569 [0.555, 0.584] (n=500) |
| IntegratedGradExplainer | 500 | 0.581 [0.566, 0.594] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.483 [0.465, 0.500] (n=500) | 0.592 [0.578, 0.607] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.504 [0.487, 0.520] (n=500) |

