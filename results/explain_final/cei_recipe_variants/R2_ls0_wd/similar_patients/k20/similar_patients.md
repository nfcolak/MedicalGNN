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
| builtin | similar | 500 | 0.501 [0.488, 0.514] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.388 [0.373, 0.404] (n=500) | 0.409 [0.393, 0.424] (n=500) |
| builtin | same_group | 500 | 0.400 [0.389, 0.413] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.270 [0.257, 0.285] (n=500) | 0.290 [0.276, 0.304] (n=500) |
| builtin | other_group | 500 | 0.286 [0.277, 0.295] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.132 [0.121, 0.143] (n=500) | 0.136 [0.124, 0.147] (n=500) |
| GNNExplainer | similar | 500 | 0.252 [0.245, 0.259] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.083 [0.074, 0.092] (n=500) | 0.066 [0.058, 0.075] (n=500) |
| GNNExplainer | same_group | 500 | 0.235 [0.226, 0.244] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.069 [0.058, 0.080] (n=500) | 0.064 [0.054, 0.074] (n=500) |
| GNNExplainer | other_group | 500 | 0.219 [0.211, 0.228] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.051 [0.041, 0.061] (n=500) | 0.053 [0.043, 0.064] (n=500) |
| GradExplainer | similar | 500 | 0.537 [0.526, 0.547] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.432 [0.420, 0.445] (n=500) | 0.584 [0.572, 0.597] (n=500) |
| GradExplainer | same_group | 500 | 0.442 [0.431, 0.453] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.321 [0.307, 0.334] (n=500) | 0.467 [0.456, 0.478] (n=500) |
| GradExplainer | other_group | 500 | 0.361 [0.353, 0.370] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.224 [0.214, 0.234] (n=500) | 0.379 [0.369, 0.388] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.572 [0.559, 0.584] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.474 [0.459, 0.489] (n=500) | 0.607 [0.592, 0.621] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.476 [0.464, 0.487] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.361 [0.347, 0.376] (n=500) | 0.488 [0.475, 0.501] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.373 [0.364, 0.382] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.238 [0.226, 0.249] (n=500) | 0.380 [0.369, 0.392] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.118 [0.102, 0.134] (n=500) | 0.119 [0.102, 0.134] (n=500) | 0.256 [0.237, 0.274] (n=500) | 0.273 [0.255, 0.291] (n=500) |
| GNNExplainer | 0.015 [0.004, 0.026] (n=500) | 0.002 [-0.010, 0.015] (n=500) | 0.032 [0.019, 0.046] (n=500) | 0.013 [-0.001, 0.026] (n=500) |
| GradExplainer | 0.111 [0.097, 0.126] (n=500) | 0.117 [0.105, 0.129] (n=500) | 0.208 [0.194, 0.224] (n=500) | 0.206 [0.193, 0.218] (n=500) |
| IntegratedGradExplainer | 0.113 [0.097, 0.129] (n=500) | 0.118 [0.105, 0.132] (n=500) | 0.237 [0.220, 0.255] (n=500) | 0.226 [0.211, 0.242] (n=500) |

