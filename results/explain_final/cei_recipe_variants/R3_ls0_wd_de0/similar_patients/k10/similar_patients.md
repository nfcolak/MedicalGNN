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
| builtin | similar | 500 | 0.557 [0.544, 0.572] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.457 [0.441, 0.474] (n=500) | 0.469 [0.451, 0.488] (n=500) |
| builtin | same_group | 500 | 0.476 [0.462, 0.489] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.361 [0.345, 0.377] (n=500) | 0.371 [0.353, 0.390] (n=500) |
| builtin | other_group | 500 | 0.293 [0.283, 0.302] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.140 [0.129, 0.152] (n=500) | 0.107 [0.095, 0.120] (n=500) |
| GNNExplainer | similar | 500 | 0.242 [0.234, 0.249] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.071 [0.061, 0.079] (n=500) | 0.051 [0.042, 0.060] (n=500) |
| GNNExplainer | same_group | 500 | 0.232 [0.224, 0.240] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.064 [0.055, 0.074] (n=500) | 0.054 [0.044, 0.064] (n=500) |
| GNNExplainer | other_group | 500 | 0.205 [0.197, 0.213] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.033 [0.024, 0.044] (n=500) | 0.041 [0.030, 0.052] (n=500) |
| GradExplainer | similar | 500 | 0.588 [0.576, 0.599] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.494 [0.480, 0.508] (n=500) | 0.661 [0.651, 0.670] (n=500) |
| GradExplainer | same_group | 500 | 0.490 [0.479, 0.501] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.379 [0.365, 0.392] (n=500) | 0.548 [0.536, 0.560] (n=500) |
| GradExplainer | other_group | 500 | 0.379 [0.371, 0.388] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.245 [0.235, 0.256] (n=500) | 0.427 [0.417, 0.438] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.655 [0.644, 0.667] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.576 [0.563, 0.591] (n=500) | 0.717 [0.707, 0.727] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.561 [0.548, 0.573] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.465 [0.450, 0.480] (n=500) | 0.610 [0.597, 0.623] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.426 [0.417, 0.436] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.303 [0.292, 0.314] (n=500) | 0.473 [0.462, 0.483] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.096 [0.080, 0.111] (n=500) | 0.098 [0.082, 0.113] (n=500) | 0.317 [0.295, 0.339] (n=500) | 0.362 [0.338, 0.386] (n=500) |
| GNNExplainer | 0.006 [-0.005, 0.017] (n=500) | -0.003 [-0.015, 0.009] (n=500) | 0.037 [0.024, 0.049] (n=500) | 0.010 [-0.005, 0.023] (n=500) |
| GradExplainer | 0.116 [0.101, 0.131] (n=500) | 0.113 [0.102, 0.124] (n=500) | 0.249 [0.233, 0.265] (n=500) | 0.233 [0.220, 0.247] (n=500) |
| IntegratedGradExplainer | 0.111 [0.096, 0.126] (n=500) | 0.107 [0.094, 0.119] (n=500) | 0.274 [0.258, 0.291] (n=500) | 0.244 [0.230, 0.258] (n=500) |

