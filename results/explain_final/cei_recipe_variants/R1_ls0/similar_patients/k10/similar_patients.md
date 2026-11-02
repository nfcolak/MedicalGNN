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
| builtin | similar | 500 | 0.570 [0.557, 0.583] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.472 [0.456, 0.489] (n=500) | 0.476 [0.458, 0.494] (n=500) |
| builtin | same_group | 500 | 0.487 [0.473, 0.501] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.375 [0.358, 0.392] (n=500) | 0.378 [0.361, 0.398] (n=500) |
| builtin | other_group | 500 | 0.300 [0.290, 0.309] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.149 [0.137, 0.160] (n=500) | 0.104 [0.091, 0.116] (n=500) |
| GNNExplainer | similar | 500 | 0.229 [0.223, 0.236] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.055 [0.047, 0.064] (n=500) | 0.041 [0.033, 0.050] (n=500) |
| GNNExplainer | same_group | 500 | 0.224 [0.216, 0.232] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.055 [0.045, 0.064] (n=500) | 0.043 [0.033, 0.052] (n=500) |
| GNNExplainer | other_group | 500 | 0.196 [0.188, 0.203] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.022 [0.013, 0.031] (n=500) | 0.028 [0.017, 0.039] (n=500) |
| GradExplainer | similar | 500 | 0.590 [0.580, 0.601] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.498 [0.485, 0.511] (n=500) | 0.658 [0.649, 0.668] (n=500) |
| GradExplainer | same_group | 500 | 0.492 [0.482, 0.503] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.382 [0.369, 0.395] (n=500) | 0.558 [0.547, 0.569] (n=500) |
| GradExplainer | other_group | 500 | 0.391 [0.382, 0.400] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.259 [0.249, 0.270] (n=500) | 0.431 [0.421, 0.441] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.634 [0.623, 0.645] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.551 [0.538, 0.565] (n=500) | 0.721 [0.711, 0.731] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.545 [0.533, 0.557] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.446 [0.431, 0.460] (n=500) | 0.614 [0.601, 0.627] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.423 [0.414, 0.433] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.299 [0.287, 0.311] (n=500) | 0.477 [0.466, 0.488] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.098 [0.081, 0.113] (n=500) | 0.097 [0.082, 0.112] (n=500) | 0.324 [0.302, 0.345] (n=500) | 0.372 [0.348, 0.398] (n=500) |
| GNNExplainer | 0.000 [-0.011, 0.012] (n=500) | -0.001 [-0.013, 0.010] (n=500) | 0.033 [0.021, 0.045] (n=500) | 0.014 [-0.000, 0.026] (n=500) |
| GradExplainer | 0.116 [0.101, 0.131] (n=500) | 0.100 [0.091, 0.110] (n=500) | 0.239 [0.223, 0.255] (n=500) | 0.227 [0.215, 0.241] (n=500) |
| IntegratedGradExplainer | 0.105 [0.091, 0.119] (n=500) | 0.107 [0.095, 0.118] (n=500) | 0.252 [0.236, 0.269] (n=500) | 0.243 [0.230, 0.257] (n=500) |

