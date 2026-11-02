# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.619 [0.604, 0.633] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.530 [0.511, 0.547] (n=500) | 0.585 [0.570, 0.599] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.446 [0.434, 0.458] (n=500) |
| GNNExplainer | 500 | 0.672 [0.654, 0.688] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.597 [0.576, 0.617] (n=500) | 0.594 [0.576, 0.610] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.839 [0.828, 0.850] (n=500) |
| GradExplainer | 500 | 0.650 [0.638, 0.662] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.569 [0.554, 0.584] (n=500) | 0.615 [0.602, 0.627] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.629 [0.616, 0.641] (n=500) |
| IntegratedGradExplainer | 500 | 0.626 [0.613, 0.639] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.539 [0.523, 0.555] (n=500) | 0.611 [0.597, 0.624] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.536 [0.521, 0.551] (n=500) |

