# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.554 [0.542, 0.567] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.451 [0.436, 0.466] (n=500) | 0.528 [0.515, 0.540] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.370 [0.357, 0.383] (n=500) |
| GNNExplainer | 500 | 0.605 [0.589, 0.622] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.515 [0.495, 0.536] (n=500) | 0.535 [0.517, 0.552] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.781 [0.768, 0.795] (n=500) |
| GradExplainer | 500 | 0.534 [0.521, 0.548] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.426 [0.410, 0.442] (n=500) | 0.522 [0.508, 0.535] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.486 [0.469, 0.503] (n=500) |
| IntegratedGradExplainer | 500 | 0.540 [0.525, 0.555] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.432 [0.414, 0.452] (n=500) | 0.529 [0.515, 0.544] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.436 [0.419, 0.453] (n=500) |

