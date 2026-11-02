# Explanation consistency across similar patients

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement of explanations across similar patients

Similar = different patients, same diagnosis group, at least 5 shared evidence items; each visit's 5 most similar visits by Jaccard of their item sets. Baselines: up to 5 random same-group visits that are not neighbours, and up to 5 random visits of other groups. Agreement is over the items both visits contain.

Visits per diagnosis group (patients in brackets): cardiac 42 (37); infection 192 (169); injury_pain 147 (131); gastrointestinal 81 (66); metabolic_renal_heme 76 (60); neuro_psych_substance 67 (57)

| pair type | pairs | mean Jaccard of the item sets |
|---|---|---|
| similar | 3025 | 0.520 |
| same_group | 3025 | 0.257 |
| other_group | 3025 | 0.259 |

### cei_mp2 (seeds [7, 1234, 2025])

| source | pair type | patients | overlap at cut | chance | lift | rank correlation |
|---|---|---|---|---|---|---|
| builtin | similar | 458 | 0.532 [0.518, 0.546] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.427 [0.410, 0.443] (n=458) | 0.430 [0.414, 0.447] (n=458) |
| builtin | same_group | 458 | 0.450 [0.437, 0.462] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.331 [0.315, 0.346] (n=458) | 0.338 [0.323, 0.353] (n=458) |
| builtin | other_group | 458 | 0.338 [0.327, 0.349] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.197 [0.183, 0.210] (n=458) | 0.188 [0.175, 0.201] (n=458) |
| GNNExplainer | similar | 458 | 0.260 [0.251, 0.269] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.093 [0.082, 0.104] (n=458) | 0.072 [0.062, 0.081] (n=458) |
| GNNExplainer | same_group | 458 | 0.233 [0.224, 0.242] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.067 [0.056, 0.078] (n=458) | 0.068 [0.056, 0.078] (n=458) |
| GNNExplainer | other_group | 458 | 0.216 [0.207, 0.224] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.047 [0.037, 0.058] (n=458) | 0.046 [0.036, 0.058] (n=458) |
| GradExplainer | similar | 458 | 0.578 [0.567, 0.589] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.483 [0.469, 0.496] (n=458) | 0.595 [0.582, 0.607] (n=458) |
| GradExplainer | same_group | 458 | 0.501 [0.490, 0.513] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.393 [0.380, 0.407] (n=458) | 0.482 [0.471, 0.494] (n=458) |
| GradExplainer | other_group | 458 | 0.437 [0.427, 0.446] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.316 [0.305, 0.328] (n=458) | 0.410 [0.400, 0.419] (n=458) |
| IntegratedGradExplainer | similar | 458 | 0.571 [0.559, 0.583] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.474 [0.459, 0.488] (n=458) | 0.599 [0.585, 0.612] (n=458) |
| IntegratedGradExplainer | same_group | 458 | 0.495 [0.484, 0.507] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.386 [0.372, 0.401] (n=458) | 0.491 [0.478, 0.504] (n=458) |
| IntegratedGradExplainer | other_group | 458 | 0.416 [0.407, 0.425] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.291 [0.279, 0.302] (n=458) | 0.405 [0.395, 0.416] (n=458) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.096 [0.080, 0.112] (n=458) | 0.092 [0.076, 0.108] (n=458) | 0.230 [0.212, 0.249] (n=458) | 0.242 [0.224, 0.260] (n=458) |
| GNNExplainer | 0.026 [0.014, 0.039] (n=458) | 0.004 [-0.009, 0.016] (n=458) | 0.046 [0.032, 0.060] (n=458) | 0.025 [0.012, 0.038] (n=458) |
| GradExplainer | 0.090 [0.076, 0.103] (n=458) | 0.113 [0.101, 0.124] (n=458) | 0.167 [0.151, 0.183] (n=458) | 0.185 [0.172, 0.198] (n=458) |
| IntegratedGradExplainer | 0.088 [0.072, 0.102] (n=458) | 0.107 [0.094, 0.121] (n=458) | 0.183 [0.166, 0.199] (n=458) | 0.193 [0.178, 0.207] (n=458) |

