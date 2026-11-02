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
| builtin | similar | 458 | 0.542 [0.529, 0.554] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.438 [0.423, 0.454] (n=458) | 0.418 [0.404, 0.433] (n=458) |
| builtin | same_group | 458 | 0.457 [0.444, 0.470] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.340 [0.324, 0.355] (n=458) | 0.329 [0.314, 0.343] (n=458) |
| builtin | other_group | 458 | 0.354 [0.344, 0.364] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.217 [0.204, 0.229] (n=458) | 0.203 [0.191, 0.214] (n=458) |
| GNNExplainer | similar | 458 | 0.271 [0.262, 0.279] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.107 [0.097, 0.116] (n=458) | 0.083 [0.073, 0.093] (n=458) |
| GNNExplainer | same_group | 458 | 0.247 [0.238, 0.256] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.085 [0.073, 0.095] (n=458) | 0.079 [0.068, 0.090] (n=458) |
| GNNExplainer | other_group | 458 | 0.225 [0.217, 0.233] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.059 [0.049, 0.068] (n=458) | 0.060 [0.049, 0.071] (n=458) |
| GradExplainer | similar | 458 | 0.598 [0.586, 0.610] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.507 [0.492, 0.522] (n=458) | 0.649 [0.638, 0.660] (n=458) |
| GradExplainer | same_group | 458 | 0.511 [0.500, 0.523] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.405 [0.391, 0.419] (n=458) | 0.547 [0.536, 0.558] (n=458) |
| GradExplainer | other_group | 458 | 0.441 [0.431, 0.451] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.322 [0.310, 0.334] (n=458) | 0.468 [0.457, 0.478] (n=458) |
| IntegratedGradExplainer | similar | 458 | 0.570 [0.557, 0.582] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.472 [0.457, 0.487] (n=458) | 0.604 [0.590, 0.618] (n=458) |
| IntegratedGradExplainer | same_group | 458 | 0.489 [0.478, 0.501] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.379 [0.365, 0.393] (n=458) | 0.490 [0.478, 0.503] (n=458) |
| IntegratedGradExplainer | other_group | 458 | 0.395 [0.385, 0.406] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.266 [0.254, 0.279] (n=458) | 0.388 [0.377, 0.400] (n=458) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.098 [0.082, 0.114] (n=458) | 0.089 [0.074, 0.104] (n=458) | 0.222 [0.204, 0.241] (n=458) | 0.215 [0.197, 0.231] (n=458) |
| GNNExplainer | 0.022 [0.010, 0.034] (n=458) | 0.004 [-0.008, 0.017] (n=458) | 0.048 [0.035, 0.061] (n=458) | 0.023 [0.010, 0.037] (n=458) |
| GradExplainer | 0.102 [0.087, 0.119] (n=458) | 0.102 [0.091, 0.113] (n=458) | 0.185 [0.167, 0.202] (n=458) | 0.181 [0.169, 0.193] (n=458) |
| IntegratedGradExplainer | 0.094 [0.078, 0.109] (n=458) | 0.114 [0.099, 0.128] (n=458) | 0.207 [0.188, 0.225] (n=458) | 0.216 [0.199, 0.231] (n=458) |

