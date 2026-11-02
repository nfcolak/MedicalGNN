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
| builtin | similar | 500 | 0.496 [0.483, 0.508] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.382 [0.366, 0.397] (n=500) | 0.390 [0.375, 0.405] (n=500) |
| builtin | same_group | 500 | 0.399 [0.389, 0.410] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.269 [0.256, 0.282] (n=500) | 0.283 [0.270, 0.296] (n=500) |
| builtin | other_group | 500 | 0.299 [0.290, 0.307] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.148 [0.137, 0.158] (n=500) | 0.147 [0.136, 0.158] (n=500) |
| GNNExplainer | similar | 500 | 0.271 [0.264, 0.279] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.107 [0.098, 0.116] (n=500) | 0.081 [0.073, 0.090] (n=500) |
| GNNExplainer | same_group | 500 | 0.245 [0.237, 0.253] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.081 [0.072, 0.091] (n=500) | 0.069 [0.060, 0.079] (n=500) |
| GNNExplainer | other_group | 500 | 0.219 [0.212, 0.226] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.051 [0.042, 0.060] (n=500) | 0.056 [0.046, 0.066] (n=500) |
| GradExplainer | similar | 500 | 0.524 [0.514, 0.533] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.416 [0.404, 0.428] (n=500) | 0.589 [0.576, 0.601] (n=500) |
| GradExplainer | same_group | 500 | 0.431 [0.421, 0.442] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.308 [0.295, 0.321] (n=500) | 0.462 [0.452, 0.473] (n=500) |
| GradExplainer | other_group | 500 | 0.365 [0.356, 0.373] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.228 [0.218, 0.238] (n=500) | 0.395 [0.385, 0.405] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.535 [0.523, 0.548] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.429 [0.415, 0.445] (n=500) | 0.551 [0.536, 0.566] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.430 [0.419, 0.441] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.306 [0.293, 0.319] (n=500) | 0.405 [0.392, 0.418] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.343 [0.334, 0.351] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.201 [0.190, 0.210] (n=500) | 0.310 [0.298, 0.322] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.113 [0.097, 0.129] (n=500) | 0.107 [0.092, 0.122] (n=500) | 0.234 [0.215, 0.251] (n=500) | 0.243 [0.225, 0.261] (n=500) |
| GNNExplainer | 0.026 [0.015, 0.037] (n=500) | 0.012 [0.001, 0.024] (n=500) | 0.056 [0.044, 0.069] (n=500) | 0.026 [0.012, 0.038] (n=500) |
| GradExplainer | 0.108 [0.093, 0.122] (n=500) | 0.126 [0.113, 0.139] (n=500) | 0.188 [0.174, 0.203] (n=500) | 0.194 [0.181, 0.206] (n=500) |
| IntegratedGradExplainer | 0.124 [0.108, 0.139] (n=500) | 0.146 [0.131, 0.161] (n=500) | 0.229 [0.211, 0.245] (n=500) | 0.241 [0.224, 0.258] (n=500) |

