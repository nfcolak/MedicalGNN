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
| builtin | similar | 458 | 0.524 [0.511, 0.537] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.417 [0.401, 0.433] (n=458) | 0.418 [0.403, 0.435] (n=458) |
| builtin | same_group | 458 | 0.436 [0.423, 0.448] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.315 [0.299, 0.329] (n=458) | 0.320 [0.306, 0.334] (n=458) |
| builtin | other_group | 458 | 0.332 [0.321, 0.342] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.190 [0.176, 0.202] (n=458) | 0.185 [0.173, 0.196] (n=458) |
| GNNExplainer | similar | 458 | 0.260 [0.251, 0.269] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.094 [0.083, 0.104] (n=458) | 0.077 [0.067, 0.087] (n=458) |
| GNNExplainer | same_group | 458 | 0.233 [0.225, 0.242] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.068 [0.057, 0.077] (n=458) | 0.073 [0.062, 0.084] (n=458) |
| GNNExplainer | other_group | 458 | 0.216 [0.207, 0.224] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.048 [0.038, 0.058] (n=458) | 0.055 [0.044, 0.066] (n=458) |
| GradExplainer | similar | 458 | 0.576 [0.566, 0.586] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.480 [0.467, 0.493] (n=458) | 0.598 [0.586, 0.609] (n=458) |
| GradExplainer | same_group | 458 | 0.501 [0.490, 0.513] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.393 [0.380, 0.407] (n=458) | 0.497 [0.486, 0.508] (n=458) |
| GradExplainer | other_group | 458 | 0.424 [0.414, 0.433] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.301 [0.289, 0.312] (n=458) | 0.422 [0.411, 0.431] (n=458) |
| IntegratedGradExplainer | similar | 458 | 0.562 [0.551, 0.574] (n=458) | 0.184 [0.183, 0.185] (n=458) | 0.464 [0.450, 0.478] (n=458) | 0.600 [0.587, 0.613] (n=458) |
| IntegratedGradExplainer | same_group | 458 | 0.475 [0.464, 0.486] (n=458) | 0.178 [0.177, 0.179] (n=458) | 0.361 [0.348, 0.376] (n=458) | 0.494 [0.482, 0.506] (n=458) |
| IntegratedGradExplainer | other_group | 458 | 0.394 [0.385, 0.403] (n=458) | 0.177 [0.176, 0.177] (n=458) | 0.265 [0.254, 0.276] (n=458) | 0.408 [0.398, 0.419] (n=458) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.103 [0.085, 0.120] (n=458) | 0.099 [0.082, 0.114] (n=458) | 0.228 [0.210, 0.246] (n=458) | 0.234 [0.216, 0.251] (n=458) |
| GNNExplainer | 0.026 [0.014, 0.039] (n=458) | 0.004 [-0.009, 0.017] (n=458) | 0.046 [0.033, 0.059] (n=458) | 0.023 [0.009, 0.036] (n=458) |
| GradExplainer | 0.087 [0.073, 0.103] (n=458) | 0.101 [0.089, 0.113] (n=458) | 0.180 [0.165, 0.195] (n=458) | 0.176 [0.164, 0.189] (n=458) |
| IntegratedGradExplainer | 0.102 [0.087, 0.117] (n=458) | 0.106 [0.093, 0.119] (n=458) | 0.199 [0.182, 0.216] (n=458) | 0.192 [0.178, 0.207] (n=458) |

