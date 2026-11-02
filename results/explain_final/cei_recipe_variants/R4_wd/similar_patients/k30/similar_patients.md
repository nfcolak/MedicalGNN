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
| builtin | similar | 458 | 0.535 [0.523, 0.549] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.431 [0.415, 0.447] (n=458) | 0.408 [0.393, 0.424] (n=458) |
| builtin | same_group | 458 | 0.441 [0.427, 0.453] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.321 [0.304, 0.335] (n=458) | 0.323 [0.309, 0.337] (n=458) |
| builtin | other_group | 458 | 0.348 [0.337, 0.358] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.209 [0.196, 0.221] (n=458) | 0.199 [0.187, 0.211] (n=458) |
| GNNExplainer | similar | 458 | 0.276 [0.267, 0.283] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.112 [0.102, 0.122] (n=458) | 0.082 [0.072, 0.091] (n=458) |
| GNNExplainer | same_group | 458 | 0.243 [0.234, 0.251] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.079 [0.069, 0.090] (n=458) | 0.075 [0.063, 0.085] (n=458) |
| GNNExplainer | other_group | 458 | 0.223 [0.216, 0.230] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.056 [0.048, 0.065] (n=458) | 0.057 [0.047, 0.068] (n=458) |
| GradExplainer | similar | 458 | 0.565 [0.553, 0.576] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.466 [0.452, 0.481] (n=458) | 0.596 [0.581, 0.609] (n=458) |
| GradExplainer | same_group | 458 | 0.485 [0.474, 0.496] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.374 [0.360, 0.387] (n=458) | 0.477 [0.464, 0.489] (n=458) |
| GradExplainer | other_group | 458 | 0.409 [0.400, 0.419] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.283 [0.271, 0.295] (n=458) | 0.397 [0.385, 0.408] (n=458) |
| IntegratedGradExplainer | similar | 458 | 0.563 [0.551, 0.575] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.464 [0.449, 0.479] (n=458) | 0.583 [0.569, 0.598] (n=458) |
| IntegratedGradExplainer | same_group | 458 | 0.486 [0.475, 0.497] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.375 [0.362, 0.389] (n=458) | 0.482 [0.470, 0.495] (n=458) |
| IntegratedGradExplainer | other_group | 458 | 0.398 [0.388, 0.407] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.269 [0.257, 0.280] (n=458) | 0.379 [0.369, 0.389] (n=458) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.110 [0.094, 0.127] (n=458) | 0.085 [0.071, 0.099] (n=458) | 0.222 [0.204, 0.241] (n=458) | 0.209 [0.192, 0.226] (n=458) |
| GNNExplainer | 0.033 [0.022, 0.045] (n=458) | 0.007 [-0.005, 0.020] (n=458) | 0.056 [0.044, 0.068] (n=458) | 0.025 [0.012, 0.038] (n=458) |
| GradExplainer | 0.093 [0.077, 0.107] (n=458) | 0.119 [0.106, 0.132] (n=458) | 0.183 [0.167, 0.200] (n=458) | 0.199 [0.184, 0.213] (n=458) |
| IntegratedGradExplainer | 0.089 [0.074, 0.103] (n=458) | 0.101 [0.086, 0.115] (n=458) | 0.195 [0.178, 0.212] (n=458) | 0.204 [0.188, 0.221] (n=458) |

