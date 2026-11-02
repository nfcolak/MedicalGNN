# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.581 [0.567, 0.595] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.485 [0.467, 0.501] (n=500) | 0.585 [0.571, 0.599] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.442 [0.429, 0.455] (n=500) |
| GNNExplainer | 500 | 0.717 [0.699, 0.735] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.653 [0.631, 0.674] (n=500) | 0.648 [0.629, 0.667] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.860 [0.847, 0.872] (n=500) |
| GradExplainer | 500 | 0.588 [0.575, 0.601] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.492 [0.476, 0.508] (n=500) | 0.590 [0.576, 0.604] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.538 [0.523, 0.553] (n=500) |
| IntegratedGradExplainer | 500 | 0.579 [0.565, 0.593] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.481 [0.464, 0.499] (n=500) | 0.589 [0.575, 0.604] (n=500) | 0.149 [0.141, 0.158] (n=500) | 0.525 [0.509, 0.542] (n=500) |

