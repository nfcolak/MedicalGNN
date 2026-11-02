# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.573 [0.560, 0.587] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.475 [0.458, 0.491] (n=500) | 0.575 [0.560, 0.590] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.430 [0.418, 0.442] (n=500) |
| GNNExplainer | 500 | 0.722 [0.704, 0.739] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.659 [0.637, 0.679] (n=500) | 0.659 [0.639, 0.678] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.867 [0.854, 0.878] (n=500) |
| GradExplainer | 500 | 0.583 [0.569, 0.597] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.486 [0.470, 0.503] (n=500) | 0.582 [0.569, 0.595] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.525 [0.510, 0.540] (n=500) |
| IntegratedGradExplainer | 500 | 0.572 [0.557, 0.586] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.472 [0.454, 0.490] (n=500) | 0.580 [0.566, 0.595] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.512 [0.497, 0.529] (n=500) |

