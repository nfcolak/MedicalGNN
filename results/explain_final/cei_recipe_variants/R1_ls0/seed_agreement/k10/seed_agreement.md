# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.569 [0.556, 0.582] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.468 [0.453, 0.484] (n=500) | 0.548 [0.534, 0.561] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.440 [0.426, 0.454] (n=500) |
| GNNExplainer | 500 | 0.752 [0.734, 0.770] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.695 [0.674, 0.717] (n=500) | 0.688 [0.669, 0.708] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.874 [0.862, 0.886] (n=500) |
| GradExplainer | 500 | 0.541 [0.529, 0.554] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.435 [0.420, 0.451] (n=500) | 0.544 [0.530, 0.558] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.478 [0.464, 0.492] (n=500) |
| IntegratedGradExplainer | 500 | 0.531 [0.517, 0.546] (n=500) | 0.187 [0.186, 0.188] (n=500) | 0.422 [0.405, 0.440] (n=500) | 0.549 [0.534, 0.564] (n=500) | 0.151 [0.143, 0.160] (n=500) | 0.414 [0.399, 0.430] (n=500) |

