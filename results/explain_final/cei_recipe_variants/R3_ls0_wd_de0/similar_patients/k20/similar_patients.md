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
| builtin | similar | 500 | 0.502 [0.490, 0.515] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.390 [0.374, 0.405] (n=500) | 0.400 [0.385, 0.415] (n=500) |
| builtin | same_group | 500 | 0.406 [0.394, 0.418] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.278 [0.263, 0.291] (n=500) | 0.287 [0.273, 0.301] (n=500) |
| builtin | other_group | 500 | 0.292 [0.282, 0.301] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.139 [0.128, 0.150] (n=500) | 0.137 [0.126, 0.149] (n=500) |
| GNNExplainer | similar | 500 | 0.267 [0.259, 0.274] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.101 [0.092, 0.110] (n=500) | 0.081 [0.072, 0.091] (n=500) |
| GNNExplainer | same_group | 500 | 0.244 [0.235, 0.253] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.080 [0.069, 0.091] (n=500) | 0.079 [0.067, 0.089] (n=500) |
| GNNExplainer | other_group | 500 | 0.221 [0.213, 0.229] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.053 [0.044, 0.063] (n=500) | 0.061 [0.050, 0.073] (n=500) |
| GradExplainer | similar | 500 | 0.538 [0.528, 0.548] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.434 [0.422, 0.446] (n=500) | 0.589 [0.577, 0.600] (n=500) |
| GradExplainer | same_group | 500 | 0.442 [0.432, 0.453] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.322 [0.309, 0.334] (n=500) | 0.472 [0.461, 0.481] (n=500) |
| GradExplainer | other_group | 500 | 0.367 [0.358, 0.375] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.230 [0.220, 0.241] (n=500) | 0.398 [0.389, 0.408] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.566 [0.554, 0.579] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.468 [0.453, 0.483] (n=500) | 0.573 [0.559, 0.587] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.466 [0.455, 0.477] (n=500) | 0.178 [0.177, 0.179] (n=500) | 0.350 [0.336, 0.364] (n=500) | 0.440 [0.427, 0.452] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.374 [0.364, 0.382] (n=500) | 0.177 [0.176, 0.178] (n=500) | 0.239 [0.227, 0.249] (n=500) | 0.341 [0.330, 0.352] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.112 [0.096, 0.128] (n=500) | 0.113 [0.098, 0.128] (n=500) | 0.250 [0.232, 0.268] (n=500) | 0.263 [0.245, 0.281] (n=500) |
| GNNExplainer | 0.021 [0.010, 0.033] (n=500) | 0.003 [-0.009, 0.015] (n=500) | 0.048 [0.036, 0.061] (n=500) | 0.020 [0.007, 0.033] (n=500) |
| GradExplainer | 0.112 [0.097, 0.127] (n=500) | 0.117 [0.106, 0.129] (n=500) | 0.203 [0.189, 0.218] (n=500) | 0.191 [0.179, 0.202] (n=500) |
| IntegratedGradExplainer | 0.118 [0.103, 0.133] (n=500) | 0.133 [0.119, 0.147] (n=500) | 0.229 [0.212, 0.246] (n=500) | 0.232 [0.217, 0.247] (n=500) |

