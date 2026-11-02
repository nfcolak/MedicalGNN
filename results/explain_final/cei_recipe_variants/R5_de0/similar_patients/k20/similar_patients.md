# Explanation consistency across similar patients

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## Agreement of explanations across similar patients

Similar = different patients, same diagnosis group, at least 5 shared evidence items; each visit's 5 most similar visits by Jaccard of their item sets. Baselines: up to 5 random same-group visits that are not neighbours, and up to 5 random visits of other groups. Agreement is over the items both visits contain.

Visits per diagnosis group (patients in brackets): cardiac 51 (44); infection 213 (184); injury_pain 162 (146); gastrointestinal 90 (74); metabolic_renal_heme 85 (68); neuro_psych_substance 71 (61)

| pair type | pairs | mean Jaccard of the item sets |
|---|---|---|
| similar | 3360 | 0.522 |
| same_group | 3360 | 0.255 |
| other_group | 3360 | 0.256 |

### cei_mp2 (seeds [7, 1234, 2025])

| source | pair type | patients | overlap at cut | chance | lift | rank correlation |
|---|---|---|---|---|---|---|
| builtin | similar | 500 | 0.499 [0.486, 0.511] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.385 [0.370, 0.401] (n=500) | 0.393 [0.379, 0.408] (n=500) |
| builtin | same_group | 500 | 0.405 [0.395, 0.417] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.276 [0.263, 0.290] (n=500) | 0.286 [0.272, 0.299] (n=500) |
| builtin | other_group | 500 | 0.306 [0.297, 0.315] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.157 [0.146, 0.167] (n=500) | 0.148 [0.137, 0.159] (n=500) |
| GNNExplainer | similar | 500 | 0.282 [0.275, 0.289] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.120 [0.112, 0.129] (n=500) | 0.097 [0.089, 0.105] (n=500) |
| GNNExplainer | same_group | 500 | 0.250 [0.242, 0.259] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.088 [0.078, 0.098] (n=500) | 0.079 [0.069, 0.089] (n=500) |
| GNNExplainer | other_group | 500 | 0.223 [0.216, 0.230] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.056 [0.047, 0.064] (n=500) | 0.059 [0.050, 0.070] (n=500) |
| GradExplainer | similar | 500 | 0.531 [0.521, 0.542] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.425 [0.413, 0.438] (n=500) | 0.586 [0.573, 0.599] (n=500) |
| GradExplainer | same_group | 500 | 0.433 [0.423, 0.444] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.310 [0.298, 0.323] (n=500) | 0.464 [0.453, 0.474] (n=500) |
| GradExplainer | other_group | 500 | 0.370 [0.362, 0.378] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.235 [0.225, 0.244] (n=500) | 0.393 [0.382, 0.402] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.535 [0.523, 0.547] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.429 [0.415, 0.444] (n=500) | 0.553 [0.537, 0.568] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.430 [0.420, 0.441] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.307 [0.294, 0.319] (n=500) | 0.399 [0.386, 0.411] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.351 [0.343, 0.359] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.211 [0.201, 0.220] (n=500) | 0.311 [0.300, 0.322] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.109 [0.093, 0.125] (n=500) | 0.107 [0.092, 0.122] (n=500) | 0.229 [0.210, 0.246] (n=500) | 0.245 [0.228, 0.262] (n=500) |
| GNNExplainer | 0.032 [0.021, 0.044] (n=500) | 0.018 [0.008, 0.030] (n=500) | 0.065 [0.053, 0.077] (n=500) | 0.038 [0.025, 0.050] (n=500) |
| GradExplainer | 0.115 [0.101, 0.129] (n=500) | 0.123 [0.110, 0.135] (n=500) | 0.191 [0.177, 0.206] (n=500) | 0.194 [0.181, 0.206] (n=500) |
| IntegratedGradExplainer | 0.123 [0.107, 0.139] (n=500) | 0.154 [0.139, 0.169] (n=500) | 0.218 [0.202, 0.236] (n=500) | 0.242 [0.226, 0.258] (n=500) |

