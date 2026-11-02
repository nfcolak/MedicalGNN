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
| builtin | similar | 500 | 0.538 [0.526, 0.551] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.434 [0.419, 0.449] (n=500) | 0.444 [0.428, 0.460] (n=500) |
| builtin | same_group | 500 | 0.471 [0.459, 0.483] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.355 [0.341, 0.370] (n=500) | 0.354 [0.339, 0.370] (n=500) |
| builtin | other_group | 500 | 0.315 [0.305, 0.324] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.167 [0.155, 0.178] (n=500) | 0.154 [0.142, 0.165] (n=500) |
| GNNExplainer | similar | 500 | 0.264 [0.256, 0.271] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.098 [0.088, 0.106] (n=500) | 0.072 [0.063, 0.080] (n=500) |
| GNNExplainer | same_group | 500 | 0.253 [0.244, 0.260] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.090 [0.080, 0.099] (n=500) | 0.069 [0.060, 0.078] (n=500) |
| GNNExplainer | other_group | 500 | 0.214 [0.207, 0.222] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.045 [0.036, 0.054] (n=500) | 0.045 [0.036, 0.054] (n=500) |
| GradExplainer | similar | 500 | 0.586 [0.576, 0.597] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.493 [0.480, 0.506] (n=500) | 0.657 [0.646, 0.668] (n=500) |
| GradExplainer | same_group | 500 | 0.479 [0.468, 0.491] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.366 [0.353, 0.380] (n=500) | 0.539 [0.527, 0.551] (n=500) |
| GradExplainer | other_group | 500 | 0.381 [0.373, 0.390] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.248 [0.238, 0.258] (n=500) | 0.434 [0.423, 0.445] (n=500) |
| IntegratedGradExplainer | similar | 500 | 0.609 [0.597, 0.621] (n=500) | 0.184 [0.183, 0.185] (n=500) | 0.521 [0.506, 0.536] (n=500) | 0.662 [0.649, 0.674] (n=500) |
| IntegratedGradExplainer | same_group | 500 | 0.506 [0.493, 0.518] (n=500) | 0.179 [0.178, 0.180] (n=500) | 0.398 [0.382, 0.413] (n=500) | 0.535 [0.520, 0.551] (n=500) |
| IntegratedGradExplainer | other_group | 500 | 0.362 [0.352, 0.371] (n=500) | 0.178 [0.177, 0.178] (n=500) | 0.223 [0.212, 0.235] (n=500) | 0.369 [0.356, 0.383] (n=500) |

Similar minus baseline, on the patients that have both (positive: explanations agree more for similar patients):

| source | lift vs same-group | rank corr. vs same-group | lift vs other-group | rank corr. vs other-group |
|---|---|---|---|---|
| builtin | 0.079 [0.064, 0.095] (n=500) | 0.090 [0.074, 0.105] (n=500) | 0.267 [0.248, 0.287] (n=500) | 0.290 [0.270, 0.310] (n=500) |
| GNNExplainer | 0.008 [-0.003, 0.018] (n=500) | 0.002 [-0.009, 0.013] (n=500) | 0.053 [0.041, 0.064] (n=500) | 0.027 [0.015, 0.039] (n=500) |
| GradExplainer | 0.127 [0.112, 0.141] (n=500) | 0.118 [0.107, 0.129] (n=500) | 0.245 [0.228, 0.262] (n=500) | 0.223 [0.210, 0.236] (n=500) |
| IntegratedGradExplainer | 0.123 [0.108, 0.140] (n=500) | 0.126 [0.113, 0.140] (n=500) | 0.298 [0.280, 0.316] (n=500) | 0.292 [0.275, 0.310] (n=500) |

