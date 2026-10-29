# Fairness checks on the explanation results

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## 1. How many nodes the main fidelity cut removes

Main cut = top 20% of the visit's eligible evidence nodes, capped at the explanation's own support. `k` is the number actually deleted; `eligible` the visit's eligible nodes; `capped` the share of visits where the support, not the 20%, set k.

**top30/cei_mp2**

- builtin: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)

**top30/graphcare**

- builtin: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)

**top30/gsat**

- builtin: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)

**top30/protgnn**

- builtin: k = 2.3, eligible = 48.4, capped by support in 94% of patients (n=499)
- GNNExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)

## 2. Matched evidence budgets

Every source scored at the same fixed k. Cells: probability drop with the top-k deleted (fidelity+, higher = more faithful) / with only the top-k kept (fidelity-, lower = more faithful); then the difference to random deletion of the same size on the same visits (+ column: higher is better; - column: lower is better). A source with fewer than k eligible nodes of nonzero importance is `beyond support` at k and is not padded.

### top30/cei_mp2

- builtin, k=1: fidelity+ +0.376 [+0.355, +0.396] (n=500), vs random +0.363 [+0.343, +0.383] (n=500); fidelity- +0.041 [+0.025, +0.057] (n=500), vs random -0.482 [-0.504, -0.460] (n=500)
- builtin, k=2: fidelity+ +0.465 [+0.444, +0.485] (n=500), vs random +0.439 [+0.419, +0.458] (n=500); fidelity- -0.012 [-0.023, +0.001] (n=500), vs random -0.510 [-0.529, -0.487] (n=500)
- builtin, k=3: fidelity+ +0.500 [+0.478, +0.520] (n=500), vs random +0.463 [+0.443, +0.482] (n=500); fidelity- -0.018 [-0.027, -0.007] (n=500), vs random -0.494 [-0.513, -0.474] (n=500)
- builtin, k=5: fidelity+ +0.517 [+0.495, +0.537] (n=500), vs random +0.456 [+0.436, +0.474] (n=500); fidelity- -0.024 [-0.032, -0.014] (n=500), vs random -0.458 [-0.477, -0.439] (n=500)
- builtin, k=10: fidelity+ +0.534 [+0.514, +0.555] (n=492), vs random +0.411 [+0.392, +0.429] (n=492); fidelity- -0.026 [-0.032, -0.019] (n=492), vs random -0.375 [-0.393, -0.358] (n=492)
- GNNExplainer, k=1: fidelity+ +0.196 [+0.178, +0.214] (n=500), vs random +0.183 [+0.165, +0.201] (n=500); fidelity- +0.266 [+0.243, +0.289] (n=500), vs random -0.257 [-0.281, -0.235] (n=500)
- GNNExplainer, k=2: fidelity+ +0.296 [+0.275, +0.317] (n=500), vs random +0.270 [+0.250, +0.291] (n=500); fidelity- +0.144 [+0.123, +0.165] (n=500), vs random -0.354 [-0.379, -0.329] (n=500)
- GNNExplainer, k=3: fidelity+ +0.353 [+0.332, +0.374] (n=500), vs random +0.316 [+0.296, +0.338] (n=500); fidelity- +0.082 [+0.064, +0.100] (n=500), vs random -0.394 [-0.418, -0.370] (n=500)
- GNNExplainer, k=5: fidelity+ +0.421 [+0.399, +0.442] (n=500), vs random +0.360 [+0.340, +0.380] (n=500); fidelity- +0.019 [+0.005, +0.035] (n=500), vs random -0.415 [-0.437, -0.392] (n=500)
- GNNExplainer, k=10: fidelity+ +0.484 [+0.465, +0.504] (n=492), vs random +0.361 [+0.342, +0.379] (n=492); fidelity- -0.023 [-0.034, -0.013] (n=492), vs random -0.373 [-0.392, -0.354] (n=492)
- GradExplainer, k=1: fidelity+ +0.284 [+0.262, +0.304] (n=500), vs random +0.271 [+0.250, +0.291] (n=500); fidelity- +0.151 [+0.133, +0.170] (n=500), vs random -0.372 [-0.394, -0.348] (n=500)
- GradExplainer, k=2: fidelity+ +0.405 [+0.384, +0.426] (n=500), vs random +0.380 [+0.360, +0.399] (n=500); fidelity- +0.051 [+0.039, +0.065] (n=500), vs random -0.447 [-0.468, -0.424] (n=500)
- GradExplainer, k=3: fidelity+ +0.461 [+0.439, +0.481] (n=500), vs random +0.424 [+0.404, +0.443] (n=500); fidelity- +0.031 [+0.021, +0.042] (n=500), vs random -0.445 [-0.465, -0.425] (n=500)
- GradExplainer, k=5: fidelity+ +0.496 [+0.474, +0.517] (n=500), vs random +0.435 [+0.415, +0.455] (n=500); fidelity- +0.030 [+0.021, +0.040] (n=500), vs random -0.404 [-0.423, -0.385] (n=500)
- GradExplainer, k=10: fidelity+ +0.520 [+0.499, +0.541] (n=492), vs random +0.397 [+0.378, +0.415] (n=492); fidelity- +0.020 [+0.013, +0.027] (n=492), vs random -0.330 [-0.348, -0.312] (n=492)
- IntegratedGradExplainer, k=1: fidelity+ +0.352 [+0.330, +0.374] (n=500), vs random +0.340 [+0.318, +0.360] (n=500); fidelity- +0.064 [+0.048, +0.081] (n=500), vs random -0.459 [-0.482, -0.435] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.445 [+0.425, +0.465] (n=500), vs random +0.420 [+0.400, +0.439] (n=500); fidelity- +0.025 [+0.014, +0.037] (n=500), vs random -0.473 [-0.493, -0.451] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.487 [+0.465, +0.506] (n=500), vs random +0.450 [+0.430, +0.468] (n=500); fidelity- +0.012 [+0.003, +0.023] (n=500), vs random -0.464 [-0.483, -0.444] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.507 [+0.484, +0.528] (n=500), vs random +0.446 [+0.426, +0.465] (n=500); fidelity- +0.014 [+0.005, +0.024] (n=500), vs random -0.420 [-0.440, -0.400] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.520 [+0.499, +0.540] (n=492), vs random +0.397 [+0.378, +0.414] (n=492); fidelity- +0.019 [+0.011, +0.027] (n=492), vs random -0.331 [-0.348, -0.312] (n=492)

### top30/graphcare

- builtin, k=1: fidelity+ +0.009 [+0.007, +0.011] (n=500), vs random -0.004 [-0.007, -0.001] (n=500); fidelity- +0.503 [+0.477, +0.529] (n=500), vs random +0.001 [-0.005, +0.007] (n=500)
- builtin, k=2: fidelity+ +0.023 [+0.019, +0.028] (n=500), vs random -0.001 [-0.006, +0.003] (n=500); fidelity- +0.478 [+0.454, +0.502] (n=500), vs random +0.003 [-0.005, +0.010] (n=500)
- builtin, k=3: fidelity+ +0.034 [+0.028, +0.041] (n=500), vs random -0.001 [-0.006, +0.004] (n=500); fidelity- +0.449 [+0.426, +0.473] (n=500), vs random +0.000 [-0.007, +0.008] (n=500)
- builtin, k=5: fidelity+ +0.061 [+0.051, +0.072] (n=500), vs random +0.003 [-0.003, +0.010] (n=500); fidelity- +0.401 [+0.380, +0.423] (n=500), vs random +0.000 [-0.009, +0.009] (n=500)
- builtin, k=10: fidelity+ +0.120 [+0.106, +0.136] (n=492), vs random +0.005 [-0.004, +0.013] (n=492); fidelity- +0.295 [+0.276, +0.315] (n=492), vs random -0.001 [-0.012, +0.010] (n=492)
- GNNExplainer, k=1: fidelity+ +0.220 [+0.199, +0.244] (n=500), vs random +0.208 [+0.186, +0.231] (n=500); fidelity- +0.182 [+0.150, +0.213] (n=500), vs random -0.320 [-0.350, -0.290] (n=500)
- GNNExplainer, k=2: fidelity+ +0.275 [+0.253, +0.298] (n=500), vs random +0.250 [+0.228, +0.274] (n=500); fidelity- +0.119 [+0.087, +0.150] (n=500), vs random -0.356 [-0.387, -0.326] (n=500)
- GNNExplainer, k=3: fidelity+ +0.295 [+0.273, +0.319] (n=500), vs random +0.260 [+0.238, +0.284] (n=500); fidelity- +0.089 [+0.058, +0.118] (n=500), vs random -0.361 [-0.392, -0.330] (n=500)
- GNNExplainer, k=5: fidelity+ +0.333 [+0.311, +0.357] (n=500), vs random +0.275 [+0.253, +0.297] (n=500); fidelity- +0.032 [+0.006, +0.058] (n=500), vs random -0.369 [-0.398, -0.339] (n=500)
- GNNExplainer, k=10: fidelity+ +0.377 [+0.354, +0.403] (n=492), vs random +0.261 [+0.241, +0.284] (n=492); fidelity- -0.025 [-0.047, -0.003] (n=492), vs random -0.322 [-0.349, -0.295] (n=492)
- GradExplainer, k=1: fidelity+ +0.181 [+0.161, +0.203] (n=500), vs random +0.168 [+0.149, +0.189] (n=500); fidelity- +0.204 [+0.177, +0.231] (n=500), vs random -0.298 [-0.328, -0.266] (n=500)
- GradExplainer, k=2: fidelity+ +0.293 [+0.268, +0.319] (n=500), vs random +0.269 [+0.246, +0.293] (n=500); fidelity- +0.087 [+0.062, +0.113] (n=500), vs random -0.389 [-0.420, -0.358] (n=500)
- GradExplainer, k=3: fidelity+ +0.380 [+0.354, +0.406] (n=500), vs random +0.345 [+0.322, +0.369] (n=500); fidelity- +0.009 [-0.016, +0.032] (n=500), vs random -0.440 [-0.471, -0.411] (n=500)
- GradExplainer, k=5: fidelity+ +0.467 [+0.440, +0.494] (n=500), vs random +0.409 [+0.386, +0.432] (n=500); fidelity- -0.077 [-0.096, -0.059] (n=500), vs random -0.477 [-0.503, -0.452] (n=500)
- GradExplainer, k=10: fidelity+ +0.529 [+0.504, +0.554] (n=486), vs random +0.418 [+0.397, +0.440] (n=486); fidelity- -0.140 [-0.155, -0.125] (n=486), vs random -0.440 [-0.462, -0.417] (n=486); reached by 97% of patients
- IntegratedGradExplainer, k=1: fidelity+ +0.365 [+0.340, +0.390] (n=500), vs random +0.352 [+0.328, +0.377] (n=500); fidelity- -0.049 [-0.073, -0.023] (n=500), vs random -0.551 [-0.584, -0.518] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.457 [+0.433, +0.482] (n=500), vs random +0.432 [+0.409, +0.455] (n=500); fidelity- -0.101 [-0.122, -0.078] (n=500), vs random -0.577 [-0.606, -0.547] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.501 [+0.476, +0.526] (n=500), vs random +0.466 [+0.442, +0.489] (n=500); fidelity- -0.116 [-0.136, -0.095] (n=500), vs random -0.565 [-0.593, -0.537] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.523 [+0.498, +0.548] (n=500), vs random +0.465 [+0.442, +0.487] (n=500); fidelity- -0.134 [-0.152, -0.116] (n=500), vs random -0.534 [-0.559, -0.510] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.543 [+0.518, +0.568] (n=492), vs random +0.427 [+0.407, +0.448] (n=492); fidelity- -0.157 [-0.171, -0.141] (n=492), vs random -0.453 [-0.475, -0.431] (n=492)

### top30/gsat

- builtin, k=1: fidelity+ +0.239 [+0.219, +0.259] (n=500), vs random +0.225 [+0.206, +0.244] (n=500); fidelity- +0.201 [+0.182, +0.221] (n=500), vs random -0.362 [-0.387, -0.338] (n=500)
- builtin, k=2: fidelity+ +0.375 [+0.353, +0.398] (n=500), vs random +0.348 [+0.327, +0.369] (n=500); fidelity- +0.053 [+0.034, +0.072] (n=500), vs random -0.474 [-0.500, -0.446] (n=500)
- builtin, k=3: fidelity+ +0.456 [+0.432, +0.479] (n=500), vs random +0.418 [+0.395, +0.439] (n=500); fidelity- -0.009 [-0.026, +0.010] (n=500), vs random -0.507 [-0.533, -0.480] (n=500)
- builtin, k=5: fidelity+ +0.511 [+0.487, +0.533] (n=500), vs random +0.447 [+0.425, +0.467] (n=500); fidelity- -0.046 [-0.062, -0.029] (n=500), vs random -0.493 [-0.517, -0.467] (n=500)
- builtin, k=10: fidelity+ +0.542 [+0.519, +0.564] (n=492), vs random +0.413 [+0.392, +0.433] (n=492); fidelity- -0.054 [-0.066, -0.040] (n=492), vs random -0.402 [-0.424, -0.380] (n=492)
- GNNExplainer, k=1: fidelity+ +0.107 [+0.093, +0.122] (n=500), vs random +0.094 [+0.080, +0.108] (n=500); fidelity- +0.395 [+0.370, +0.420] (n=500), vs random -0.168 [-0.190, -0.147] (n=500)
- GNNExplainer, k=2: fidelity+ +0.180 [+0.161, +0.199] (n=500), vs random +0.153 [+0.136, +0.171] (n=500); fidelity- +0.283 [+0.259, +0.308] (n=500), vs random -0.244 [-0.267, -0.222] (n=500)
- GNNExplainer, k=3: fidelity+ +0.240 [+0.220, +0.260] (n=500), vs random +0.202 [+0.184, +0.220] (n=500); fidelity- +0.204 [+0.182, +0.228] (n=500), vs random -0.294 [-0.316, -0.271] (n=500)
- GNNExplainer, k=5: fidelity+ +0.339 [+0.317, +0.361] (n=500), vs random +0.275 [+0.256, +0.295] (n=500); fidelity- +0.098 [+0.077, +0.120] (n=500), vs random -0.349 [-0.372, -0.327] (n=500)
- GNNExplainer, k=10: fidelity+ +0.456 [+0.433, +0.481] (n=492), vs random +0.328 [+0.308, +0.347] (n=492); fidelity- -0.015 [-0.033, +0.004] (n=492), vs random -0.363 [-0.383, -0.343] (n=492)
- GradExplainer, k=1: fidelity+ +0.283 [+0.261, +0.307] (n=500), vs random +0.270 [+0.248, +0.292] (n=500); fidelity- +0.098 [+0.073, +0.124] (n=500), vs random -0.465 [-0.495, -0.434] (n=500)
- GradExplainer, k=2: fidelity+ +0.410 [+0.387, +0.434] (n=500), vs random +0.383 [+0.361, +0.405] (n=500); fidelity- -0.006 [-0.027, +0.017] (n=500), vs random -0.533 [-0.561, -0.503] (n=500)
- GradExplainer, k=3: fidelity+ +0.473 [+0.449, +0.496] (n=500), vs random +0.434 [+0.412, +0.456] (n=500); fidelity- -0.043 [-0.061, -0.023] (n=500), vs random -0.542 [-0.568, -0.515] (n=500)
- GradExplainer, k=5: fidelity+ +0.525 [+0.500, +0.548] (n=500), vs random +0.461 [+0.439, +0.482] (n=500); fidelity- -0.063 [-0.079, -0.046] (n=500), vs random -0.510 [-0.535, -0.483] (n=500)
- GradExplainer, k=10: fidelity+ +0.554 [+0.531, +0.578] (n=492), vs random +0.425 [+0.405, +0.446] (n=492); fidelity- -0.065 [-0.078, -0.051] (n=492), vs random -0.413 [-0.435, -0.390] (n=492)
- IntegratedGradExplainer, k=1: fidelity+ +0.368 [+0.344, +0.391] (n=500), vs random +0.354 [+0.331, +0.377] (n=500); fidelity- -0.005 [-0.027, +0.018] (n=500), vs random -0.568 [-0.596, -0.537] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.464 [+0.441, +0.488] (n=500), vs random +0.437 [+0.415, +0.459] (n=500); fidelity- -0.054 [-0.075, -0.033] (n=500), vs random -0.581 [-0.608, -0.553] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.508 [+0.484, +0.531] (n=500), vs random +0.469 [+0.447, +0.491] (n=500); fidelity- -0.073 [-0.091, -0.056] (n=500), vs random -0.572 [-0.597, -0.546] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.542 [+0.518, +0.565] (n=500), vs random +0.478 [+0.456, +0.498] (n=500); fidelity- -0.080 [-0.096, -0.064] (n=500), vs random -0.527 [-0.550, -0.502] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.563 [+0.541, +0.587] (n=492), vs random +0.435 [+0.414, +0.456] (n=492); fidelity- -0.080 [-0.093, -0.067] (n=492), vs random -0.428 [-0.450, -0.405] (n=492)

### top30/protgnn

- builtin, k=1: fidelity+ +0.007 [+0.005, +0.009] (n=499), vs random -0.009 [-0.012, -0.007] (n=499); fidelity- +0.635 [+0.613, +0.657] (n=499), vs random -0.002 [-0.002, -0.001] (n=499)
- builtin, k=2: fidelity+ +0.015 [+0.010, +0.020] (n=461), vs random -0.017 [-0.021, -0.013] (n=461); fidelity- +0.617 [+0.594, +0.639] (n=461), vs random -0.002 [-0.003, -0.001] (n=461); reached by 65% of patients
- builtin, k=3: fidelity+ +0.013 [+0.007, +0.018] (n=406), vs random -0.026 [-0.032, -0.021] (n=406); fidelity- +0.601 [+0.576, +0.627] (n=406), vs random -0.002 [-0.005, +0.000] (n=406); reached by 43% of patients
- builtin, k=5: fidelity+ -0.001 [-0.012, +0.011] (n=149), vs random -0.042 [-0.054, -0.030] (n=149); fidelity- +0.604 [+0.563, +0.649] (n=149), vs random +0.001 [-0.010, +0.011] (n=149); reached by 10% of patients
- builtin, k=10: beyond support on every visit
- GNNExplainer, k=1: fidelity+ +0.205 [+0.185, +0.225] (n=500), vs random +0.189 [+0.169, +0.209] (n=500); fidelity- +0.644 [+0.623, +0.664] (n=500), vs random +0.003 [+0.003, +0.004] (n=500)
- GNNExplainer, k=2: fidelity+ +0.284 [+0.262, +0.305] (n=500), vs random +0.250 [+0.229, +0.271] (n=500); fidelity- +0.633 [+0.611, +0.654] (n=500), vs random +0.006 [+0.004, +0.008] (n=500)
- GNNExplainer, k=3: fidelity+ +0.336 [+0.315, +0.357] (n=500), vs random +0.287 [+0.266, +0.307] (n=500); fidelity- +0.615 [+0.594, +0.637] (n=500), vs random +0.004 [+0.001, +0.008] (n=500)
- GNNExplainer, k=5: fidelity+ +0.403 [+0.381, +0.426] (n=500), vs random +0.322 [+0.301, +0.343] (n=500); fidelity- +0.549 [+0.528, +0.571] (n=500), vs random -0.022 [-0.029, -0.014] (n=500)
- GNNExplainer, k=10: fidelity+ +0.487 [+0.465, +0.509] (n=492), vs random +0.336 [+0.315, +0.357] (n=492); fidelity- +0.301 [+0.280, +0.321] (n=492), vs random -0.140 [-0.157, -0.124] (n=492)
- GradExplainer, k=1: fidelity+ +0.163 [+0.146, +0.179] (n=500), vs random +0.146 [+0.130, +0.163] (n=500); fidelity- +0.643 [+0.622, +0.664] (n=500), vs random +0.003 [+0.002, +0.004] (n=500)
- GradExplainer, k=2: fidelity+ +0.282 [+0.261, +0.305] (n=500), vs random +0.249 [+0.228, +0.270] (n=500); fidelity- +0.636 [+0.615, +0.657] (n=500), vs random +0.009 [+0.007, +0.011] (n=500)
- GradExplainer, k=3: fidelity+ +0.358 [+0.335, +0.381] (n=500), vs random +0.309 [+0.287, +0.331] (n=500); fidelity- +0.624 [+0.603, +0.645] (n=500), vs random +0.013 [+0.010, +0.016] (n=500)
- GradExplainer, k=5: fidelity+ +0.445 [+0.422, +0.469] (n=500), vs random +0.364 [+0.342, +0.386] (n=500); fidelity- +0.577 [+0.556, +0.599] (n=500), vs random +0.006 [+0.000, +0.013] (n=500)
- GradExplainer, k=10: fidelity+ +0.540 [+0.516, +0.562] (n=492), vs random +0.388 [+0.367, +0.410] (n=492); fidelity- +0.377 [+0.356, +0.397] (n=492), vs random -0.064 [-0.079, -0.049] (n=492)
- IntegratedGradExplainer, k=1: fidelity+ +0.321 [+0.297, +0.345] (n=500), vs random +0.305 [+0.281, +0.328] (n=500); fidelity- +0.646 [+0.625, +0.667] (n=500), vs random +0.006 [+0.005, +0.007] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.430 [+0.404, +0.454] (n=500), vs random +0.396 [+0.372, +0.419] (n=500); fidelity- +0.640 [+0.619, +0.661] (n=500), vs random +0.013 [+0.011, +0.016] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.483 [+0.459, +0.506] (n=500), vs random +0.434 [+0.410, +0.457] (n=500); fidelity- +0.630 [+0.608, +0.651] (n=500), vs random +0.019 [+0.015, +0.022] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.534 [+0.510, +0.558] (n=500), vs random +0.453 [+0.431, +0.476] (n=500); fidelity- +0.587 [+0.565, +0.608] (n=500), vs random +0.016 [+0.009, +0.023] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.565 [+0.542, +0.588] (n=492), vs random +0.414 [+0.391, +0.436] (n=492); fidelity- +0.382 [+0.361, +0.402] (n=492), vs random -0.059 [-0.075, -0.044] (n=492)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

### top30/protgnn

visits with a check: 2013; with at least one removable node in the subgraph: 1609

- subgraph size: 4.3 nodes, of which removable 1.8; the visit has 48 eligible nodes
- node kinds in the subgraphs: measurement 42%, analyte 31%, knowledge 16%, visit 9%, vital 2%, diagnosis 0%

Change from the untouched visit (positive = the quantity went down). Deleting the subgraph's eligible nodes vs deleting the same number of random eligible nodes; the last column is the paired difference (subgraph - random).

| quantity | subgraph deleted | random deleted | difference |
|---|---|---|---|
| prototype similarity (activation) | +0.056 [+0.050, +0.063] (n=499) | +0.053 [+0.048, +0.059] (n=499) | +0.003 [-0.002, +0.007] (n=499) |
| prototype's share of the class logit | +0.067 [+0.058, +0.075] (n=499) | +0.063 [+0.056, +0.070] (n=499) | +0.004 [-0.002, +0.009] (n=499) |
| class probability | +0.014 [+0.010, +0.018] (n=499) | +0.031 [+0.028, +0.035] (n=499) | -0.017 [-0.021, -0.013] (n=499) |

Keeping only the subgraph's eligible nodes (all other eligible nodes deleted) vs keeping the same number of random ones: values after the intervention, as a fraction of the untouched visit's value (1 = fully preserved).

| quantity | subgraph kept | random kept | difference |
|---|---|---|---|
| prototype similarity (activation) | +0.261 [+0.251, +0.270] (n=499) | +0.245 [+0.237, +0.252] (n=499) | +0.016 [+0.012, +0.021] (n=499) |
| class probability | +0.116 [+0.108, +0.126] (n=499) | +0.115 [+0.107, +0.123] (n=499) | +0.002 [-0.002, +0.005] (n=499) |

On the untouched visits the explained prototype contributes 2.43 to the class logit, whose value is -36.64 (means over patients; the logit is the sum over all prototypes of the classifier weight times the activation).

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top30/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.523 [+0.501, +0.543] (n=500)
- GNNExplainer: +0.432 [+0.410, +0.453] (n=500)
- GradExplainer: +0.502 [+0.480, +0.522] (n=500)
- IntegratedGradExplainer: +0.510 [+0.489, +0.530] (n=500)
- GNNExplainer - GradExplainer: -0.070 [-0.084, -0.056] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.078 [-0.093, -0.063] (n=500)
- GNNExplainer - builtin: -0.091 [-0.107, -0.076] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.008 [-0.013, -0.004] (n=500)
- GradExplainer - builtin: -0.021 [-0.027, -0.015] (n=500)
- IntegratedGradExplainer - builtin: -0.013 [-0.017, -0.009] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.017 [-0.024, -0.009] (n=500)
- GNNExplainer: +0.024 [+0.008, +0.041] (n=500)
- GradExplainer: +0.034 [+0.026, +0.042] (n=500)
- IntegratedGradExplainer: +0.027 [+0.019, +0.035] (n=500)
- GNNExplainer - GradExplainer: -0.009 [-0.027, +0.008] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.002 [-0.020, +0.015] (n=500)
- GNNExplainer - builtin: +0.041 [+0.025, +0.057] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.007 [+0.002, +0.012] (n=500)
- GradExplainer - builtin: +0.050 [+0.043, +0.057] (n=500)
- IntegratedGradExplainer - builtin: +0.043 [+0.036, +0.050] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.275 [+0.264, +0.285] (n=500)
- GNNExplainer: +0.218 [+0.207, +0.228] (n=500)
- GradExplainer: +0.258 [+0.247, +0.268] (n=500)
- IntegratedGradExplainer: +0.261 [+0.249, +0.271] (n=500)
- GNNExplainer - GradExplainer: -0.040 [-0.048, -0.032] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.043 [-0.051, -0.035] (n=500)
- GNNExplainer - builtin: -0.057 [-0.065, -0.049] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.003 [-0.005, -0.001] (n=500)
- GradExplainer - builtin: -0.017 [-0.020, -0.015] (n=500)
- IntegratedGradExplainer - builtin: -0.014 [-0.017, -0.012] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.226 [-0.234, -0.217] (n=500)
- GNNExplainer: -0.200 [-0.210, -0.190] (n=500)
- GradExplainer: -0.197 [-0.206, -0.187] (n=500)
- IntegratedGradExplainer: -0.198 [-0.207, -0.188] (n=500)
- GNNExplainer - GradExplainer: -0.003 [-0.011, +0.004] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.003 [-0.010, +0.005] (n=500)
- GNNExplainer - builtin: +0.026 [+0.018, +0.033] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.001 [-0.001, +0.003] (n=500)
- GradExplainer - builtin: +0.029 [+0.026, +0.033] (n=500)
- IntegratedGradExplainer - builtin: +0.028 [+0.024, +0.032] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.718 [+0.704, +0.730] (n=500)
- GNNExplainer: +0.102 [+0.101, +0.102] (n=500)
- GradExplainer: +0.385 [+0.377, +0.394] (n=500)
- IntegratedGradExplainer: +0.347 [+0.340, +0.353] (n=500)
- GNNExplainer - GradExplainer: -0.284 [-0.292, -0.275] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.245 [-0.251, -0.239] (n=500)
- GNNExplainer - builtin: -0.616 [-0.628, -0.603] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.039 [+0.033, +0.043] (n=500)
- GradExplainer - builtin: -0.332 [-0.344, -0.320] (n=500)
- IntegratedGradExplainer - builtin: -0.371 [-0.381, -0.360] (n=500)

### top30/graphcare

**fidelity+** (higher = more faithful)

- builtin: +0.063 [+0.057, +0.070] (n=500)
- GNNExplainer: +0.322 [+0.300, +0.346] (n=500)
- GradExplainer: +0.496 [+0.470, +0.520] (n=500)
- IntegratedGradExplainer: +0.531 [+0.506, +0.555] (n=500)
- GNNExplainer - GradExplainer: -0.174 [-0.194, -0.153] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.209 [-0.231, -0.187] (n=500)
- GNNExplainer - builtin: +0.259 [+0.236, +0.282] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.035 [-0.043, -0.028] (n=500)
- GradExplainer - builtin: +0.432 [+0.410, +0.454] (n=500)
- IntegratedGradExplainer - builtin: +0.468 [+0.445, +0.490] (n=500)

**fidelity-** (lower = more faithful)

- builtin: +0.369 [+0.346, +0.391] (n=500)
- GNNExplainer: +0.051 [+0.022, +0.081] (n=500)
- GradExplainer: -0.118 [-0.136, -0.099] (n=500)
- IntegratedGradExplainer: -0.150 [-0.167, -0.132] (n=500)
- GNNExplainer - GradExplainer: +0.168 [+0.141, +0.197] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.201 [+0.172, +0.228] (n=500)
- GNNExplainer - builtin: -0.318 [-0.347, -0.289] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.032 [+0.022, +0.042] (n=500)
- GradExplainer - builtin: -0.486 [-0.509, -0.462] (n=500)
- IntegratedGradExplainer - builtin: -0.518 [-0.542, -0.494] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.014 [+0.009, +0.019] (n=500)
- GNNExplainer: +0.172 [+0.159, +0.186] (n=500)
- GradExplainer: +0.341 [+0.325, +0.357] (n=500)
- IntegratedGradExplainer: +0.327 [+0.312, +0.341] (n=500)
- GNNExplainer - GradExplainer: -0.169 [-0.185, -0.153] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.155 [-0.170, -0.140] (n=500)
- GNNExplainer - builtin: +0.158 [+0.145, +0.173] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.014 [+0.010, +0.018] (n=500)
- GradExplainer - builtin: +0.327 [+0.311, +0.343] (n=500)
- IntegratedGradExplainer - builtin: +0.313 [+0.298, +0.328] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.010 [-0.014, -0.005] (n=500)
- GNNExplainer: -0.181 [-0.195, -0.167] (n=500)
- GradExplainer: -0.334 [-0.349, -0.318] (n=500)
- IntegratedGradExplainer: -0.310 [-0.322, -0.298] (n=500)
- GNNExplainer - GradExplainer: +0.152 [+0.136, +0.168] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.129 [+0.116, +0.142] (n=500)
- GNNExplainer - builtin: -0.172 [-0.187, -0.157] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.023 [-0.029, -0.018] (n=500)
- GradExplainer - builtin: -0.324 [-0.340, -0.308] (n=500)
- IntegratedGradExplainer - builtin: -0.301 [-0.313, -0.288] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.159 [+0.151, +0.169] (n=500)
- GNNExplainer: +0.099 [+0.099, +0.100] (n=500)
- GradExplainer: +0.725 [+0.719, +0.732] (n=500)
- IntegratedGradExplainer: +0.635 [+0.628, +0.643] (n=500)
- GNNExplainer - GradExplainer: -0.626 [-0.632, -0.620] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.536 [-0.544, -0.528] (n=500)
- GNNExplainer - builtin: -0.060 [-0.069, -0.052] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.090 [+0.083, +0.097] (n=500)
- GradExplainer - builtin: +0.566 [+0.557, +0.575] (n=500)
- IntegratedGradExplainer - builtin: +0.476 [+0.464, +0.489] (n=500)

### top30/gsat

**fidelity+** (higher = more faithful)

- builtin: +0.529 [+0.506, +0.551] (n=500)
- GNNExplainer: +0.387 [+0.365, +0.409] (n=500)
- GradExplainer: +0.533 [+0.509, +0.556] (n=500)
- IntegratedGradExplainer: +0.549 [+0.525, +0.571] (n=500)
- GNNExplainer - GradExplainer: -0.146 [-0.166, -0.128] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.161 [-0.182, -0.143] (n=500)
- GNNExplainer - builtin: -0.142 [-0.162, -0.124] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.015 [-0.021, -0.010] (n=500)
- GradExplainer - builtin: +0.004 [-0.004, +0.012] (n=500)
- IntegratedGradExplainer - builtin: +0.019 [+0.012, +0.027] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.052 [-0.067, -0.037] (n=500)
- GNNExplainer: +0.043 [+0.021, +0.066] (n=500)
- GradExplainer: -0.057 [-0.071, -0.041] (n=500)
- IntegratedGradExplainer: -0.074 [-0.089, -0.059] (n=500)
- GNNExplainer - GradExplainer: +0.100 [+0.077, +0.123] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.117 [+0.095, +0.142] (n=500)
- GNNExplainer - builtin: +0.096 [+0.072, +0.119] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.018 [+0.009, +0.026] (n=500)
- GradExplainer - builtin: -0.004 [-0.015, +0.006] (n=500)
- IntegratedGradExplainer - builtin: -0.022 [-0.033, -0.012] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.284 [+0.271, +0.296] (n=500)
- GNNExplainer: +0.245 [+0.234, +0.256] (n=500)
- GradExplainer: +0.295 [+0.282, +0.308] (n=500)
- IntegratedGradExplainer: +0.304 [+0.291, +0.317] (n=500)
- GNNExplainer - GradExplainer: -0.050 [-0.059, -0.042] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.059 [-0.067, -0.051] (n=500)
- GNNExplainer - builtin: -0.039 [-0.048, -0.031] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.009 [-0.011, -0.006] (n=500)
- GradExplainer - builtin: +0.011 [+0.007, +0.016] (n=500)
- IntegratedGradExplainer - builtin: +0.020 [+0.015, +0.024] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.252 [-0.263, -0.241] (n=500)
- GNNExplainer: -0.245 [-0.255, -0.234] (n=500)
- GradExplainer: -0.256 [-0.267, -0.245] (n=500)
- IntegratedGradExplainer: -0.264 [-0.275, -0.253] (n=500)
- GNNExplainer - GradExplainer: +0.011 [+0.002, +0.021] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.020 [+0.010, +0.030] (n=500)
- GNNExplainer - builtin: +0.007 [-0.003, +0.017] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.008 [+0.006, +0.011] (n=500)
- GradExplainer - builtin: -0.004 [-0.009, +0.000] (n=500)
- IntegratedGradExplainer - builtin: -0.013 [-0.017, -0.008] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.100 [+0.099, +0.102] (n=500)
- GNNExplainer: +0.112 [+0.111, +0.113] (n=500)
- GradExplainer: +0.479 [+0.470, +0.487] (n=500)
- IntegratedGradExplainer: +0.396 [+0.389, +0.404] (n=500)
- GNNExplainer - GradExplainer: -0.366 [-0.374, -0.357] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.284 [-0.292, -0.276] (n=500)
- GNNExplainer - builtin: +0.012 [+0.011, +0.013] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.082 [+0.076, +0.088] (n=500)
- GradExplainer - builtin: +0.378 [+0.370, +0.386] (n=500)
- IntegratedGradExplainer - builtin: +0.296 [+0.288, +0.304] (n=500)

### top30/protgnn

**fidelity+** (higher = more faithful)

- builtin: +0.012 [+0.009, +0.016] (n=499)
- GNNExplainer: +0.432 [+0.410, +0.454] (n=500)
- GradExplainer: +0.499 [+0.475, +0.522] (n=500)
- IntegratedGradExplainer: +0.543 [+0.519, +0.567] (n=500)
- GNNExplainer - GradExplainer: -0.067 [-0.084, -0.050] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.111 [-0.128, -0.094] (n=500)
- GNNExplainer - builtin: +0.420 [+0.396, +0.444] (n=499)
- GradExplainer - IntegratedGradExplainer: -0.044 [-0.055, -0.034] (n=500)
- GradExplainer - builtin: +0.487 [+0.462, +0.511] (n=499)
- IntegratedGradExplainer - builtin: +0.530 [+0.505, +0.554] (n=499)

**fidelity-** (lower = more faithful)

- builtin: +0.619 [+0.597, +0.640] (n=499)
- GNNExplainer: +0.405 [+0.382, +0.431] (n=500)
- GradExplainer: +0.457 [+0.436, +0.481] (n=500)
- IntegratedGradExplainer: +0.463 [+0.442, +0.488] (n=500)
- GNNExplainer - GradExplainer: -0.053 [-0.063, -0.043] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.059 [-0.068, -0.050] (n=500)
- GNNExplainer - builtin: -0.215 [-0.233, -0.197] (n=499)
- GradExplainer - IntegratedGradExplainer: -0.006 [-0.011, -0.002] (n=500)
- GradExplainer - builtin: -0.162 [-0.178, -0.146] (n=499)
- IntegratedGradExplainer - builtin: -0.156 [-0.172, -0.139] (n=499)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: -0.009 [-0.011, -0.007] (n=499)
- GNNExplainer: +0.201 [+0.189, +0.211] (n=500)
- GradExplainer: +0.230 [+0.218, +0.242] (n=500)
- IntegratedGradExplainer: +0.248 [+0.235, +0.259] (n=500)
- GNNExplainer - GradExplainer: -0.030 [-0.038, -0.022] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.047 [-0.056, -0.039] (n=500)
- GNNExplainer - builtin: +0.210 [+0.198, +0.221] (n=499)
- GradExplainer - IntegratedGradExplainer: -0.017 [-0.021, -0.014] (n=500)
- GradExplainer - builtin: +0.240 [+0.227, +0.252] (n=499)
- IntegratedGradExplainer - builtin: +0.257 [+0.244, +0.270] (n=499)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.001 [-0.002, -0.001] (n=499)
- GNNExplainer: -0.122 [-0.130, -0.113] (n=500)
- GradExplainer: -0.080 [-0.088, -0.071] (n=500)
- IntegratedGradExplainer: -0.079 [-0.088, -0.070] (n=500)
- GNNExplainer - GradExplainer: -0.042 [-0.047, -0.037] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.042 [-0.047, -0.037] (n=500)
- GNNExplainer - builtin: -0.120 [-0.129, -0.112] (n=499)
- GradExplainer - IntegratedGradExplainer: -0.000 [-0.002, +0.002] (n=500)
- GradExplainer - builtin: -0.079 [-0.088, -0.070] (n=499)
- IntegratedGradExplainer - builtin: -0.079 [-0.087, -0.069] (n=499)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.936 [+0.933, +0.939] (n=499)
- GNNExplainer: +0.104 [+0.103, +0.105] (n=500)
- GradExplainer: +0.549 [+0.542, +0.556] (n=500)
- IntegratedGradExplainer: +0.496 [+0.490, +0.502] (n=500)
- GNNExplainer - GradExplainer: -0.445 [-0.452, -0.438] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.392 [-0.398, -0.385] (n=500)
- GNNExplainer - builtin: -0.832 [-0.835, -0.829] (n=499)
- GradExplainer - IntegratedGradExplainer: +0.053 [+0.048, +0.058] (n=500)
- GradExplainer - builtin: -0.387 [-0.393, -0.381] (n=499)
- IntegratedGradExplainer - builtin: -0.440 [-0.446, -0.434] (n=499)

### Same explainer across methods (paired over patients)

- fidelity+, GNNExplainer, top30/cei_mp2 - top30/graphcare: +0.110 [+0.090, +0.129] (n=500)
- fidelity+, GNNExplainer, top30/cei_mp2 - top30/gsat: +0.045 [+0.026, +0.065] (n=500)
- fidelity+, GNNExplainer, top30/cei_mp2 - top30/protgnn: -0.000 [-0.019, +0.019] (n=500)
- fidelity+, GNNExplainer, top30/graphcare - top30/gsat: -0.065 [-0.086, -0.043] (n=500)
- fidelity+, GNNExplainer, top30/graphcare - top30/protgnn: -0.110 [-0.130, -0.088] (n=500)
- fidelity+, GNNExplainer, top30/gsat - top30/protgnn: -0.045 [-0.067, -0.024] (n=500)
- fidelity+, GradExplainer, top30/cei_mp2 - top30/graphcare: +0.006 [-0.008, +0.020] (n=500)
- fidelity+, GradExplainer, top30/cei_mp2 - top30/gsat: -0.031 [-0.044, -0.019] (n=500)
- fidelity+, GradExplainer, top30/cei_mp2 - top30/protgnn: +0.003 [-0.014, +0.020] (n=500)
- fidelity+, GradExplainer, top30/graphcare - top30/gsat: -0.038 [-0.050, -0.025] (n=500)
- fidelity+, GradExplainer, top30/graphcare - top30/protgnn: -0.003 [-0.025, +0.018] (n=500)
- fidelity+, GradExplainer, top30/gsat - top30/protgnn: +0.034 [+0.013, +0.054] (n=500)
- fidelity+, IntegratedGradExplainer, top30/cei_mp2 - top30/graphcare: -0.021 [-0.035, -0.008] (n=500)
- fidelity+, IntegratedGradExplainer, top30/cei_mp2 - top30/gsat: -0.038 [-0.050, -0.027] (n=500)
- fidelity+, IntegratedGradExplainer, top30/cei_mp2 - top30/protgnn: -0.033 [-0.047, -0.020] (n=500)
- fidelity+, IntegratedGradExplainer, top30/graphcare - top30/gsat: -0.017 [-0.029, -0.007] (n=500)
- fidelity+, IntegratedGradExplainer, top30/graphcare - top30/protgnn: -0.012 [-0.031, +0.007] (n=500)
- fidelity+, IntegratedGradExplainer, top30/gsat - top30/protgnn: +0.005 [-0.012, +0.022] (n=500)
- fidelity+, builtin, top30/cei_mp2 - top30/graphcare: +0.460 [+0.439, +0.479] (n=500)
- fidelity+, builtin, top30/cei_mp2 - top30/gsat: -0.006 [-0.019, +0.006] (n=500)
- fidelity+, builtin, top30/cei_mp2 - top30/protgnn: +0.510 [+0.488, +0.530] (n=499)
- fidelity+, builtin, top30/graphcare - top30/gsat: -0.466 [-0.486, -0.445] (n=500)
- fidelity+, builtin, top30/graphcare - top30/protgnn: +0.051 [+0.044, +0.058] (n=499)
- fidelity+, builtin, top30/gsat - top30/protgnn: +0.516 [+0.492, +0.539] (n=499)
- fidelity-, GNNExplainer, top30/cei_mp2 - top30/graphcare: -0.026 [-0.053, +0.000] (n=500)
- fidelity-, GNNExplainer, top30/cei_mp2 - top30/gsat: -0.019 [-0.043, +0.005] (n=500)
- fidelity-, GNNExplainer, top30/cei_mp2 - top30/protgnn: -0.380 [-0.405, -0.356] (n=500)
- fidelity-, GNNExplainer, top30/graphcare - top30/gsat: +0.008 [-0.023, +0.036] (n=500)
- fidelity-, GNNExplainer, top30/graphcare - top30/protgnn: -0.354 [-0.385, -0.323] (n=500)
- fidelity-, GNNExplainer, top30/gsat - top30/protgnn: -0.362 [-0.388, -0.335] (n=500)
- fidelity-, GradExplainer, top30/cei_mp2 - top30/graphcare: +0.151 [+0.132, +0.170] (n=500)
- fidelity-, GradExplainer, top30/cei_mp2 - top30/gsat: +0.090 [+0.074, +0.107] (n=500)
- fidelity-, GradExplainer, top30/cei_mp2 - top30/protgnn: -0.424 [-0.447, -0.400] (n=500)
- fidelity-, GradExplainer, top30/graphcare - top30/gsat: -0.061 [-0.081, -0.041] (n=500)
- fidelity-, GradExplainer, top30/graphcare - top30/protgnn: -0.575 [-0.600, -0.549] (n=500)
- fidelity-, GradExplainer, top30/gsat - top30/protgnn: -0.514 [-0.540, -0.487] (n=500)
- fidelity-, IntegratedGradExplainer, top30/cei_mp2 - top30/graphcare: +0.176 [+0.158, +0.196] (n=500)
- fidelity-, IntegratedGradExplainer, top30/cei_mp2 - top30/gsat: +0.101 [+0.085, +0.117] (n=500)
- fidelity-, IntegratedGradExplainer, top30/cei_mp2 - top30/protgnn: -0.437 [-0.460, -0.412] (n=500)
- fidelity-, IntegratedGradExplainer, top30/graphcare - top30/gsat: -0.076 [-0.093, -0.057] (n=500)
- fidelity-, IntegratedGradExplainer, top30/graphcare - top30/protgnn: -0.613 [-0.638, -0.588] (n=500)
- fidelity-, IntegratedGradExplainer, top30/gsat - top30/protgnn: -0.538 [-0.565, -0.509] (n=500)
- fidelity-, builtin, top30/cei_mp2 - top30/graphcare: -0.385 [-0.408, -0.363] (n=500)
- fidelity-, builtin, top30/cei_mp2 - top30/gsat: +0.036 [+0.020, +0.052] (n=500)
- fidelity-, builtin, top30/cei_mp2 - top30/protgnn: -0.635 [-0.658, -0.612] (n=499)
- fidelity-, builtin, top30/graphcare - top30/gsat: +0.421 [+0.394, +0.448] (n=500)
- fidelity-, builtin, top30/graphcare - top30/protgnn: -0.250 [-0.272, -0.229] (n=499)
- fidelity-, builtin, top30/gsat - top30/protgnn: -0.671 [-0.700, -0.641] (n=499)
- deletion vs random, GNNExplainer, top30/cei_mp2 - top30/graphcare: +0.045 [+0.033, +0.057] (n=500)
- deletion vs random, GNNExplainer, top30/cei_mp2 - top30/gsat: -0.027 [-0.037, -0.017] (n=500)
- deletion vs random, GNNExplainer, top30/cei_mp2 - top30/protgnn: +0.017 [+0.008, +0.027] (n=500)
- deletion vs random, GNNExplainer, top30/graphcare - top30/gsat: -0.073 [-0.086, -0.059] (n=500)
- deletion vs random, GNNExplainer, top30/graphcare - top30/protgnn: -0.029 [-0.041, -0.016] (n=500)
- deletion vs random, GNNExplainer, top30/gsat - top30/protgnn: +0.044 [+0.032, +0.056] (n=500)
- deletion vs random, GradExplainer, top30/cei_mp2 - top30/graphcare: -0.083 [-0.093, -0.074] (n=500)
- deletion vs random, GradExplainer, top30/cei_mp2 - top30/gsat: -0.038 [-0.044, -0.031] (n=500)
- deletion vs random, GradExplainer, top30/cei_mp2 - top30/protgnn: +0.027 [+0.019, +0.036] (n=500)
- deletion vs random, GradExplainer, top30/graphcare - top30/gsat: +0.046 [+0.038, +0.054] (n=500)
- deletion vs random, GradExplainer, top30/graphcare - top30/protgnn: +0.111 [+0.097, +0.124] (n=500)
- deletion vs random, GradExplainer, top30/gsat - top30/protgnn: +0.065 [+0.054, +0.076] (n=500)
- deletion vs random, IntegratedGradExplainer, top30/cei_mp2 - top30/graphcare: -0.066 [-0.074, -0.059] (n=500)
- deletion vs random, IntegratedGradExplainer, top30/cei_mp2 - top30/gsat: -0.043 [-0.049, -0.037] (n=500)
- deletion vs random, IntegratedGradExplainer, top30/cei_mp2 - top30/protgnn: +0.013 [+0.004, +0.021] (n=500)
- deletion vs random, IntegratedGradExplainer, top30/graphcare - top30/gsat: +0.023 [+0.017, +0.029] (n=500)
- deletion vs random, IntegratedGradExplainer, top30/graphcare - top30/protgnn: +0.079 [+0.068, +0.090] (n=500)
- deletion vs random, IntegratedGradExplainer, top30/gsat - top30/protgnn: +0.056 [+0.046, +0.066] (n=500)
- deletion vs random, builtin, top30/cei_mp2 - top30/graphcare: +0.261 [+0.250, +0.272] (n=500)
- deletion vs random, builtin, top30/cei_mp2 - top30/gsat: -0.009 [-0.016, -0.003] (n=500)
- deletion vs random, builtin, top30/cei_mp2 - top30/protgnn: +0.284 [+0.272, +0.295] (n=499)
- deletion vs random, builtin, top30/graphcare - top30/gsat: -0.270 [-0.283, -0.257] (n=500)
- deletion vs random, builtin, top30/graphcare - top30/protgnn: +0.023 [+0.019, +0.028] (n=499)
- deletion vs random, builtin, top30/gsat - top30/protgnn: +0.293 [+0.279, +0.306] (n=499)
- insertion vs random, GNNExplainer, top30/cei_mp2 - top30/graphcare: -0.019 [-0.031, -0.008] (n=500)
- insertion vs random, GNNExplainer, top30/cei_mp2 - top30/gsat: +0.044 [+0.033, +0.054] (n=500)
- insertion vs random, GNNExplainer, top30/cei_mp2 - top30/protgnn: -0.079 [-0.088, -0.069] (n=500)
- insertion vs random, GNNExplainer, top30/graphcare - top30/gsat: +0.063 [+0.049, +0.077] (n=500)
- insertion vs random, GNNExplainer, top30/graphcare - top30/protgnn: -0.060 [-0.073, -0.047] (n=500)
- insertion vs random, GNNExplainer, top30/gsat - top30/protgnn: -0.123 [-0.134, -0.112] (n=500)
- insertion vs random, GradExplainer, top30/cei_mp2 - top30/graphcare: +0.137 [+0.125, +0.149] (n=500)
- insertion vs random, GradExplainer, top30/cei_mp2 - top30/gsat: +0.059 [+0.051, +0.067] (n=500)
- insertion vs random, GradExplainer, top30/cei_mp2 - top30/protgnn: -0.117 [-0.128, -0.108] (n=500)
- insertion vs random, GradExplainer, top30/graphcare - top30/gsat: -0.078 [-0.090, -0.066] (n=500)
- insertion vs random, GradExplainer, top30/graphcare - top30/protgnn: -0.254 [-0.269, -0.238] (n=500)
- insertion vs random, GradExplainer, top30/gsat - top30/protgnn: -0.176 [-0.188, -0.165] (n=500)
- insertion vs random, IntegratedGradExplainer, top30/cei_mp2 - top30/graphcare: +0.112 [+0.102, +0.122] (n=500)
- insertion vs random, IntegratedGradExplainer, top30/cei_mp2 - top30/gsat: +0.067 [+0.059, +0.075] (n=500)
- insertion vs random, IntegratedGradExplainer, top30/cei_mp2 - top30/protgnn: -0.118 [-0.129, -0.109] (n=500)
- insertion vs random, IntegratedGradExplainer, top30/graphcare - top30/gsat: -0.046 [-0.055, -0.037] (n=500)
- insertion vs random, IntegratedGradExplainer, top30/graphcare - top30/protgnn: -0.231 [-0.243, -0.218] (n=500)
- insertion vs random, IntegratedGradExplainer, top30/gsat - top30/protgnn: -0.185 [-0.197, -0.174] (n=500)
- insertion vs random, builtin, top30/cei_mp2 - top30/graphcare: -0.216 [-0.225, -0.207] (n=500)
- insertion vs random, builtin, top30/cei_mp2 - top30/gsat: +0.026 [+0.018, +0.034] (n=500)
- insertion vs random, builtin, top30/cei_mp2 - top30/protgnn: -0.225 [-0.233, -0.216] (n=499)
- insertion vs random, builtin, top30/graphcare - top30/gsat: +0.242 [+0.231, +0.254] (n=500)
- insertion vs random, builtin, top30/graphcare - top30/protgnn: -0.008 [-0.013, -0.004] (n=499)
- insertion vs random, builtin, top30/gsat - top30/protgnn: -0.251 [-0.262, -0.240] (n=499)
- sparsity, GNNExplainer, top30/cei_mp2 - top30/graphcare: +0.002 [+0.002, +0.003] (n=500)
- sparsity, GNNExplainer, top30/cei_mp2 - top30/gsat: -0.011 [-0.011, -0.010] (n=500)
- sparsity, GNNExplainer, top30/cei_mp2 - top30/protgnn: -0.002 [-0.003, -0.002] (n=500)
- sparsity, GNNExplainer, top30/graphcare - top30/gsat: -0.013 [-0.014, -0.012] (n=500)
- sparsity, GNNExplainer, top30/graphcare - top30/protgnn: -0.005 [-0.005, -0.004] (n=500)
- sparsity, GNNExplainer, top30/gsat - top30/protgnn: +0.008 [+0.008, +0.009] (n=500)
- sparsity, GradExplainer, top30/cei_mp2 - top30/graphcare: -0.340 [-0.349, -0.332] (n=500)
- sparsity, GradExplainer, top30/cei_mp2 - top30/gsat: -0.093 [-0.100, -0.086] (n=500)
- sparsity, GradExplainer, top30/cei_mp2 - top30/protgnn: -0.164 [-0.169, -0.158] (n=500)
- sparsity, GradExplainer, top30/graphcare - top30/gsat: +0.247 [+0.237, +0.257] (n=500)
- sparsity, GradExplainer, top30/graphcare - top30/protgnn: +0.177 [+0.170, +0.183] (n=500)
- sparsity, GradExplainer, top30/gsat - top30/protgnn: -0.070 [-0.077, -0.063] (n=500)
- sparsity, IntegratedGradExplainer, top30/cei_mp2 - top30/graphcare: -0.289 [-0.298, -0.280] (n=500)
- sparsity, IntegratedGradExplainer, top30/cei_mp2 - top30/gsat: -0.050 [-0.057, -0.042] (n=500)
- sparsity, IntegratedGradExplainer, top30/cei_mp2 - top30/protgnn: -0.149 [-0.155, -0.143] (n=500)
- sparsity, IntegratedGradExplainer, top30/graphcare - top30/gsat: +0.239 [+0.229, +0.250] (n=500)
- sparsity, IntegratedGradExplainer, top30/graphcare - top30/protgnn: +0.140 [+0.130, +0.149] (n=500)
- sparsity, IntegratedGradExplainer, top30/gsat - top30/protgnn: -0.100 [-0.107, -0.092] (n=500)
- sparsity, builtin, top30/cei_mp2 - top30/graphcare: +0.558 [+0.544, +0.571] (n=500)
- sparsity, builtin, top30/cei_mp2 - top30/gsat: +0.617 [+0.605, +0.629] (n=500)
- sparsity, builtin, top30/cei_mp2 - top30/protgnn: -0.218 [-0.231, -0.206] (n=499)
- sparsity, builtin, top30/graphcare - top30/gsat: +0.059 [+0.051, +0.068] (n=500)
- sparsity, builtin, top30/graphcare - top30/protgnn: -0.776 [-0.784, -0.768] (n=499)
- sparsity, builtin, top30/gsat - top30/protgnn: -0.835 [-0.838, -0.832] (n=499)

