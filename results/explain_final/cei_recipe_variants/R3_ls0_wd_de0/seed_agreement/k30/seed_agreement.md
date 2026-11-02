# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.616 [0.601, 0.629] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.526 [0.509, 0.543] (n=500) | 0.578 [0.563, 0.592] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.462 [0.450, 0.474] (n=500) |
| GNNExplainer | 500 | 0.685 [0.668, 0.702] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.614 [0.592, 0.634] (n=500) | 0.617 [0.597, 0.635] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.835 [0.822, 0.847] (n=500) |
| GradExplainer | 500 | 0.643 [0.630, 0.655] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.560 [0.545, 0.575] (n=500) | 0.620 [0.607, 0.632] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.597 [0.585, 0.611] (n=500) |
| IntegratedGradExplainer | 500 | 0.606 [0.592, 0.619] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.515 [0.498, 0.531] (n=500) | 0.605 [0.591, 0.619] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.491 [0.475, 0.508] (n=500) |

