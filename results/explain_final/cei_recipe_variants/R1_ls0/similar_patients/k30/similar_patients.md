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
| builtin | similar | 458 | 0.529 [0.515, 0.543] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.423 [0.406, 0.440] (n=458) | 0.417 [0.400, 0.434] (n=458) |
| builtin | same_group | 458 | 0.435 [0.422, 0.447] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.313 [0.298, 0.328] (n=458) | 0.319 [0.305, 0.334] (n=458) |
| builtin | other_group | 458 | 0.323 [0.313, 0.334] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.179 [0.166, 0.191] (n=458) | 0.174 [0.161, 0.187] (n=458) |
| GNNExplainer | similar | 458 | 0.258 [0.250, 0.266] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.091 [0.081, 0.100] (n=458) | 0.071 [0.062, 0.081] (n=458) |
| GNNExplainer | same_group | 458 | 0.236 [0.226, 0.244] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.070 [0.059, 0.081] (n=458) | 0.068 [0.057, 0.079] (n=458) |
| GNNExplainer | other_group | 458 | 0.220 [0.212, 0.228] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.053 [0.043, 0.063] (n=458) | 0.047 [0.037, 0.058] (n=458) |
| GradExplainer | similar | 458 | 0.583 [0.572, 0.593] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.488 [0.475, 0.501] (n=458) | 0.602 [0.589, 0.614] (n=458) |
| GradExplainer | same_group | 458 | 0.501 [0.490, 0.512] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.393 [0.379, 0.407] (n=458) | 0.488 [0.477, 0.499] (n=458) |
| GradExplainer | other_group | 458 | 0.427 [0.417, 0.436] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.304 [0.293, 0.316] (n=458) | 0.410 [0.400, 0.420] (n=458) |
| IntegratedGradExplainer | similar | 458 | 0.586 [0.574, 0.598] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.492 [0.477, 0.507] (n=458) | 0.610 [0.597, 0.624] (n=458) |
| IntegratedGradExplainer | same_group | 458 | 0.499 [0.487, 0.511] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.391 [0.377, 0.405] (n=458) | 0.509 [0.496, 0.521] (n=458) |
| IntegratedGradExplainer | other_group | 458 | 0.420 [0.411, 0.429] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.296 [0.285, 0.307] (n=458) | 0.422 [0.411, 0.431] (n=458) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.110 [0.093, 0.126] (n=458) | 0.098 [0.082, 0.114] (n=458) | 0.244 [0.225, 0.263] (n=458) | 0.243 [0.225, 0.261] (n=458) |
| GNNExplainer | 0.021 [0.009, 0.034] (n=458) | 0.003 [-0.009, 0.016] (n=458) | 0.038 [0.025, 0.052] (n=458) | 0.024 [0.011, 0.037] (n=458) |
| GradExplainer | 0.096 [0.082, 0.110] (n=458) | 0.114 [0.102, 0.126] (n=458) | 0.184 [0.169, 0.200] (n=458) | 0.192 [0.179, 0.205] (n=458) |
| IntegratedGradExplainer | 0.102 [0.086, 0.117] (n=458) | 0.101 [0.088, 0.114] (n=458) | 0.196 [0.179, 0.213] (n=458) | 0.188 [0.173, 0.203] (n=458) |

