# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.586 [0.573, 0.599] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.490 [0.473, 0.506] (n=500) | 0.580 [0.566, 0.594] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.434 [0.423, 0.446] (n=500) |
| GNNExplainer | 500 | 0.612 [0.596, 0.627] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.523 [0.503, 0.541] (n=500) | 0.546 [0.528, 0.562] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.773 [0.758, 0.787] (n=500) |
| GradExplainer | 500 | 0.609 [0.598, 0.622] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.519 [0.506, 0.534] (n=500) | 0.604 [0.591, 0.617] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.573 [0.558, 0.588] (n=500) |
| IntegratedGradExplainer | 500 | 0.584 [0.572, 0.597] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.488 [0.472, 0.503] (n=500) | 0.597 [0.584, 0.611] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.526 [0.510, 0.541] (n=500) |

