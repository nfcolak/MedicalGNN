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
| builtin | similar | 500 | 0.545 [0.533, 0.559] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.442 [0.427, 0.459] (n=500) | 0.443 [0.425, 0.461] (n=500) |
| builtin | same_group | 500 | 0.464 [0.452, 0.477] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.348 [0.333, 0.363] (n=500) | 0.343 [0.326, 0.361] (n=500) |
| builtin | other_group | 500 | 0.292 [0.283, 0.302] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.139 [0.128, 0.151] (n=500) | 0.108 [0.096, 0.120] (n=500) |
| GNNExplainer | similar | 500 | 0.253 [0.246, 0.260] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.084 [0.076, 0.093] (n=500) | 0.057 [0.048, 0.066] (n=500) |
| GNNExplainer | same_group | 500 | 0.240 [0.232, 0.248] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.074 [0.064, 0.084] (n=500) | 0.054 [0.044, 0.064] (n=500) |
| GNNExplainer | other_group | 500 | 0.208 [0.201, 0.216] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.037 [0.028, 0.046] (n=500) | 0.035 [0.025, 0.045] (n=500) |
| GradExplainer | similar | 500 | 0.600 [0.590, 0.610] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.510 [0.497, 0.522] (n=500) | 0.666 [0.656, 0.676] (n=500) |
| GradExplainer | same_group | 500 | 0.498 [0.487, 0.509] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.389 [0.375, 0.403] (n=500) | 0.559 [0.548, 0.570] (n=500) |
| GradExplainer | other_group | 500 | 0.389 [0.380, 0.398] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.257 [0.246, 0.268] (n=500) | 0.447 [0.437, 0.458] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.634 [0.623, 0.646] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.552 [0.538, 0.566] (n=500) | 0.695 [0.683, 0.707] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.537 [0.525, 0.548] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.436 [0.421, 0.450] (n=500) | 0.582 [0.568, 0.596] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.395 [0.385, 0.406] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.264 [0.252, 0.277] (n=500) | 0.421 [0.409, 0.433] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.095 [0.080, 0.111] (n=500) | 0.100 [0.084, 0.115] (n=500) | 0.303 [0.282, 0.324] (n=500) | 0.335 [0.312, 0.358] (n=500) |
| GNNExplainer | 0.010 [-0.000, 0.021] (n=500) | 0.003 [-0.009, 0.013] (n=500) | 0.047 [0.035, 0.058] (n=500) | 0.022 [0.009, 0.035] (n=500) |
| GradExplainer | 0.120 [0.105, 0.136] (n=500) | 0.107 [0.096, 0.117] (n=500) | 0.252 [0.237, 0.268] (n=500) | 0.219 [0.206, 0.231] (n=500) |
| IntegratedGradExplainer | 0.116 [0.101, 0.132] (n=500) | 0.113 [0.100, 0.125] (n=500) | 0.287 [0.268, 0.306] (n=500) | 0.274 [0.257, 0.291] (n=500) |

