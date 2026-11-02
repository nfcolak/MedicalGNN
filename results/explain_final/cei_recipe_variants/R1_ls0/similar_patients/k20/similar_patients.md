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
| builtin | similar | 500 | 0.499 [0.486, 0.513] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.386 [0.370, 0.402] (n=500) | 0.407 [0.390, 0.422] (n=500) |
| builtin | same_group | 500 | 0.403 [0.392, 0.414] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.274 [0.260, 0.288] (n=500) | 0.292 [0.278, 0.306] (n=500) |
| builtin | other_group | 500 | 0.289 [0.280, 0.299] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.136 [0.125, 0.148] (n=500) | 0.141 [0.128, 0.152] (n=500) |
| GNNExplainer | similar | 500 | 0.253 [0.246, 0.260] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.085 [0.076, 0.094] (n=500) | 0.068 [0.059, 0.077] (n=500) |
| GNNExplainer | same_group | 500 | 0.234 [0.226, 0.242] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.068 [0.058, 0.078] (n=500) | 0.066 [0.055, 0.076] (n=500) |
| GNNExplainer | other_group | 500 | 0.213 [0.205, 0.221] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.043 [0.035, 0.053] (n=500) | 0.049 [0.038, 0.060] (n=500) |
| GradExplainer | similar | 500 | 0.535 [0.525, 0.545] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.430 [0.418, 0.443] (n=500) | 0.585 [0.572, 0.598] (n=500) |
| GradExplainer | same_group | 500 | 0.433 [0.422, 0.443] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.310 [0.297, 0.323] (n=500) | 0.450 [0.439, 0.462] (n=500) |
| GradExplainer | other_group | 500 | 0.354 [0.345, 0.362] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.215 [0.205, 0.225] (n=500) | 0.363 [0.353, 0.373] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.564 [0.552, 0.576] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.466 [0.450, 0.480] (n=500) | 0.617 [0.601, 0.631] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.469 [0.458, 0.480] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.353 [0.340, 0.367] (n=500) | 0.493 [0.480, 0.505] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.376 [0.366, 0.386] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.242 [0.229, 0.254] (n=500) | 0.395 [0.384, 0.407] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.112 [0.096, 0.128] (n=500) | 0.115 [0.098, 0.131] (n=500) | 0.250 [0.231, 0.267] (n=500) | 0.266 [0.246, 0.285] (n=500) |
| GNNExplainer | 0.017 [0.005, 0.029] (n=500) | 0.002 [-0.009, 0.015] (n=500) | 0.041 [0.028, 0.055] (n=500) | 0.019 [0.005, 0.032] (n=500) |
| GradExplainer | 0.121 [0.107, 0.135] (n=500) | 0.135 [0.123, 0.148] (n=500) | 0.215 [0.201, 0.230] (n=500) | 0.222 [0.209, 0.236] (n=500) |
| IntegratedGradExplainer | 0.112 [0.097, 0.127] (n=500) | 0.124 [0.110, 0.137] (n=500) | 0.224 [0.207, 0.241] (n=500) | 0.221 [0.206, 0.237] (n=500) |

