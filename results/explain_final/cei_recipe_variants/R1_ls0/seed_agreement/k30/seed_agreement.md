# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.623 [0.609, 0.638] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.535 [0.518, 0.554] (n=500) | 0.589 [0.573, 0.603] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.474 [0.461, 0.486] (n=500) |
| GNNExplainer | 500 | 0.702 [0.684, 0.719] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.634 [0.611, 0.655] (n=500) | 0.629 [0.610, 0.649] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.844 [0.832, 0.857] (n=500) |
| GradExplainer | 500 | 0.641 [0.629, 0.653] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.558 [0.542, 0.573] (n=500) | 0.620 [0.607, 0.633] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.594 [0.580, 0.608] (n=500) |
| IntegratedGradExplainer | 500 | 0.597 [0.582, 0.611] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.503 [0.485, 0.521] (n=500) | 0.600 [0.585, 0.613] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.490 [0.473, 0.506] (n=500) |

