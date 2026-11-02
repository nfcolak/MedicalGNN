# Seed agreement of the explanations

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement between seeds

Same source, same visits, pairs of seeds; eligible evidence nodes only. Overlap is the share of the top nodes two seeds share; chance is what random subsets of that size would share; lift is 0 at chance and 1 for identical top sets; rank correlation is Spearman over all eligible nodes.

### cei_mp2 (seeds [7, 1234, 2025])

| source | patients | overlap at cut | chance | lift | overlap at k=5 | chance | rank correlation |
|---|---|---|---|---|---|---|---|
| builtin | 500 | 0.618 [0.604, 0.633] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.529 [0.512, 0.547] (n=500) | 0.579 [0.563, 0.593] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.466 [0.453, 0.479] (n=500) |
| GNNExplainer | 500 | 0.717 [0.699, 0.734] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.653 [0.631, 0.673] (n=500) | 0.648 [0.629, 0.667] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.854 [0.842, 0.866] (n=500) |
| GradExplainer | 500 | 0.640 [0.628, 0.652] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.556 [0.541, 0.571] (n=500) | 0.610 [0.596, 0.622] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.576 [0.563, 0.591] (n=500) |
| IntegratedGradExplainer | 500 | 0.589 [0.574, 0.603] (n=500) | 0.187 [0.185, 0.188] (n=500) | 0.493 [0.475, 0.510] (n=500) | 0.583 [0.568, 0.596] (n=500) | 0.150 [0.141, 0.158] (n=500) | 0.467 [0.451, 0.483] (n=500) |

