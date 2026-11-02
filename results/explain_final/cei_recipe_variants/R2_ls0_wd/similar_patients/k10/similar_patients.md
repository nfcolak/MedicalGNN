# Explanation consistency across similar patients

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement of explanations across similar patients

Similar = different patients, same diagnosis group, at least 5 shared evidence items; each visit's 5 most similar visits by Jaccard of their item sets. Baselines: up to 5 random same-group visits that are not neighbours, and up to 5 random visits of other groups. Agreement is over the items both visits contain.

Visits per diagnosis group (patients in brackets): cardiac 25 (21); infection 229 (199); injury_pain 219 (195); gastrointestinal 81 (68); metabolic_renal_heme 28 (23); neuro_psych_substance 52 (51)

| pair type | pairs | mean Jaccard of the item sets |
|---|---|---|
| similar | 3170 | 0.550 |
| same_group | 3170 | 0.266 |
| other_group | 3170 | 0.263 |

### cei_mp2 (seeds [7, 1234, 2025])

| source | pair type | patients | overlap at cut | chance | lift | rank correlation |
|---|---|---|---|---|---|---|
| builtin | similar | 500 | 0.572 [0.559, 0.585] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.475 [0.459, 0.491] (n=500) | 0.472 [0.454, 0.491] (n=500) |
| builtin | same_group | 500 | 0.487 [0.473, 0.500] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.375 [0.358, 0.391] (n=500) | 0.371 [0.353, 0.390] (n=500) |
| builtin | other_group | 500 | 0.297 [0.287, 0.307] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.146 [0.134, 0.158] (n=500) | 0.094 [0.081, 0.106] (n=500) |
| GNNExplainer | similar | 500 | 0.232 [0.226, 0.239] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.059 [0.051, 0.067] (n=500) | 0.041 [0.033, 0.050] (n=500) |
| GNNExplainer | same_group | 500 | 0.224 [0.216, 0.232] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.055 [0.045, 0.064] (n=500) | 0.044 [0.034, 0.053] (n=500) |
| GNNExplainer | other_group | 500 | 0.196 [0.189, 0.204] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.023 [0.014, 0.033] (n=500) | 0.027 [0.017, 0.037] (n=500) |
| GradExplainer | similar | 500 | 0.589 [0.579, 0.599] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.496 [0.484, 0.508] (n=500) | 0.655 [0.645, 0.664] (n=500) |
| GradExplainer | same_group | 500 | 0.499 [0.489, 0.510] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.390 [0.378, 0.403] (n=500) | 0.551 [0.541, 0.563] (n=500) |
| GradExplainer | other_group | 500 | 0.388 [0.380, 0.397] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.257 [0.246, 0.267] (n=500) | 0.419 [0.409, 0.429] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.639 [0.628, 0.650] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.557 [0.543, 0.570] (n=500) | 0.716 [0.706, 0.725] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.546 [0.534, 0.559] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.447 [0.432, 0.462] (n=500) | 0.610 [0.597, 0.623] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.420 [0.411, 0.429] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.295 [0.284, 0.305] (n=500) | 0.469 [0.458, 0.480] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.100 [0.084, 0.116] (n=500) | 0.101 [0.085, 0.116] (n=500) | 0.329 [0.308, 0.351] (n=500) | 0.378 [0.354, 0.406] (n=500) |
| GNNExplainer | 0.004 [-0.007, 0.015] (n=500) | -0.002 [-0.014, 0.009] (n=500) | 0.036 [0.023, 0.048] (n=500) | 0.015 [0.001, 0.027] (n=500) |
| GradExplainer | 0.106 [0.092, 0.120] (n=500) | 0.103 [0.093, 0.114] (n=500) | 0.239 [0.224, 0.255] (n=500) | 0.236 [0.223, 0.249] (n=500) |
| IntegratedGradExplainer | 0.109 [0.095, 0.124] (n=500) | 0.106 [0.094, 0.117] (n=500) | 0.262 [0.246, 0.278] (n=500) | 0.246 [0.233, 0.261] (n=500) |

