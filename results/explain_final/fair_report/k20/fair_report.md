# Fairness checks on the explanation results

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## 1. How many nodes the main fidelity cut removes

Main cut = top 20% of the visit's eligible evidence nodes, capped at the explanation's own support. `k` is the number actually deleted; `eligible` the visit's eligible nodes; `capped` the share of visits where the support, not the 20%, set k.

**top20/cei_mp2**

- builtin: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)

**top20/graphcare**

- builtin: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)

**top20/gsat**

- builtin: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)

**top20/protgnn**

- builtin: k = 2.5, eligible = 48.6, capped by support in 92% of patients (n=500)
- GNNExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)

## 2. Matched evidence budgets

Every source scored at the same fixed k. Cells: probability drop with the top-k deleted (fidelity+, higher = more faithful) / with only the top-k kept (fidelity-, lower = more faithful); then the difference to random deletion of the same size on the same visits (+ column: higher is better; - column: lower is better). A source with fewer than k eligible nodes of nonzero importance is `beyond support` at k and is not padded.

### top20/cei_mp2

- builtin, k=1: fidelity+ +0.371 [+0.351, +0.391] (n=500), vs random +0.359 [+0.339, +0.378] (n=500); fidelity- +0.032 [+0.017, +0.047] (n=500), vs random -0.501 [-0.521, -0.478] (n=500)
- builtin, k=2: fidelity+ +0.457 [+0.436, +0.478] (n=500), vs random +0.433 [+0.414, +0.452] (n=500); fidelity- -0.005 [-0.016, +0.007] (n=500), vs random -0.510 [-0.531, -0.489] (n=500)
- builtin, k=3: fidelity+ +0.484 [+0.465, +0.504] (n=500), vs random +0.448 [+0.430, +0.467] (n=500); fidelity- -0.008 [-0.017, +0.002] (n=500), vs random -0.489 [-0.508, -0.470] (n=500)
- builtin, k=5: fidelity+ +0.505 [+0.485, +0.525] (n=500), vs random +0.445 [+0.427, +0.464] (n=500); fidelity- -0.012 [-0.021, -0.004] (n=500), vs random -0.450 [-0.469, -0.432] (n=500)
- builtin, k=10: fidelity+ +0.529 [+0.508, +0.550] (n=494), vs random +0.410 [+0.391, +0.428] (n=494); fidelity- -0.016 [-0.023, -0.009] (n=494), vs random -0.365 [-0.383, -0.348] (n=494)
- GNNExplainer, k=1: fidelity+ +0.194 [+0.177, +0.211] (n=500), vs random +0.182 [+0.165, +0.199] (n=500); fidelity- +0.257 [+0.236, +0.280] (n=500), vs random -0.276 [-0.297, -0.254] (n=500)
- GNNExplainer, k=2: fidelity+ +0.300 [+0.280, +0.319] (n=500), vs random +0.276 [+0.256, +0.295] (n=500); fidelity- +0.131 [+0.111, +0.151] (n=500), vs random -0.375 [-0.398, -0.353] (n=500)
- GNNExplainer, k=3: fidelity+ +0.365 [+0.345, +0.386] (n=500), vs random +0.329 [+0.310, +0.350] (n=500); fidelity- +0.061 [+0.044, +0.078] (n=500), vs random -0.421 [-0.443, -0.398] (n=500)
- GNNExplainer, k=5: fidelity+ +0.439 [+0.419, +0.460] (n=500), vs random +0.379 [+0.360, +0.398] (n=500); fidelity- -0.001 [-0.014, +0.013] (n=500), vs random -0.438 [-0.459, -0.417] (n=500)
- GNNExplainer, k=10: fidelity+ +0.493 [+0.473, +0.513] (n=494), vs random +0.373 [+0.355, +0.390] (n=494); fidelity- -0.030 [-0.039, -0.020] (n=494), vs random -0.379 [-0.397, -0.361] (n=494)
- GradExplainer, k=1: fidelity+ +0.270 [+0.249, +0.292] (n=500), vs random +0.258 [+0.238, +0.279] (n=500); fidelity- +0.168 [+0.147, +0.187] (n=500), vs random -0.365 [-0.388, -0.343] (n=500)
- GradExplainer, k=2: fidelity+ +0.380 [+0.359, +0.401] (n=500), vs random +0.355 [+0.335, +0.375] (n=500); fidelity- +0.074 [+0.058, +0.088] (n=500), vs random -0.432 [-0.453, -0.411] (n=500)
- GradExplainer, k=3: fidelity+ +0.433 [+0.413, +0.455] (n=500), vs random +0.398 [+0.378, +0.417] (n=500); fidelity- +0.049 [+0.037, +0.060] (n=500), vs random -0.433 [-0.453, -0.413] (n=500)
- GradExplainer, k=5: fidelity+ +0.478 [+0.456, +0.500] (n=500), vs random +0.418 [+0.398, +0.438] (n=500); fidelity- +0.044 [+0.034, +0.054] (n=500), vs random -0.394 [-0.414, -0.375] (n=500)
- GradExplainer, k=10: fidelity+ +0.504 [+0.482, +0.526] (n=494), vs random +0.384 [+0.364, +0.403] (n=494); fidelity- +0.034 [+0.026, +0.042] (n=494), vs random -0.315 [-0.333, -0.298] (n=494)
- IntegratedGradExplainer, k=1: fidelity+ +0.333 [+0.312, +0.354] (n=500), vs random +0.320 [+0.300, +0.342] (n=500); fidelity- +0.086 [+0.070, +0.103] (n=500), vs random -0.447 [-0.469, -0.424] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.433 [+0.412, +0.454] (n=500), vs random +0.408 [+0.388, +0.428] (n=500); fidelity- +0.029 [+0.016, +0.041] (n=500), vs random -0.477 [-0.498, -0.456] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.464 [+0.443, +0.486] (n=500), vs random +0.429 [+0.408, +0.448] (n=500); fidelity- +0.026 [+0.016, +0.037] (n=500), vs random -0.455 [-0.476, -0.434] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.490 [+0.469, +0.512] (n=500), vs random +0.431 [+0.411, +0.450] (n=500); fidelity- +0.032 [+0.023, +0.042] (n=500), vs random -0.406 [-0.426, -0.387] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.508 [+0.486, +0.530] (n=494), vs random +0.388 [+0.369, +0.407] (n=494); fidelity- +0.031 [+0.024, +0.039] (n=494), vs random -0.318 [-0.336, -0.301] (n=494)

### top20/graphcare

- builtin, k=1: fidelity+ +0.008 [+0.006, +0.011] (n=500), vs random -0.004 [-0.006, -0.001] (n=500); fidelity- +0.520 [+0.494, +0.546] (n=500), vs random +0.000 [-0.006, +0.006] (n=500)
- builtin, k=2: fidelity+ +0.024 [+0.019, +0.029] (n=500), vs random +0.001 [-0.004, +0.005] (n=500); fidelity- +0.491 [+0.465, +0.515] (n=500), vs random +0.006 [-0.002, +0.013] (n=500)
- builtin, k=3: fidelity+ +0.031 [+0.026, +0.036] (n=500), vs random -0.003 [-0.007, +0.001] (n=500); fidelity- +0.459 [+0.436, +0.481] (n=500), vs random +0.004 [-0.004, +0.011] (n=500)
- builtin, k=5: fidelity+ +0.054 [+0.046, +0.064] (n=500), vs random -0.003 [-0.008, +0.003] (n=500); fidelity- +0.413 [+0.392, +0.434] (n=500), vs random +0.008 [-0.001, +0.017] (n=500)
- builtin, k=10: fidelity+ +0.109 [+0.095, +0.124] (n=494), vs random -0.006 [-0.014, +0.002] (n=494); fidelity- +0.310 [+0.291, +0.331] (n=494), vs random +0.007 [-0.003, +0.017] (n=494)
- GNNExplainer, k=1: fidelity+ +0.209 [+0.190, +0.230] (n=500), vs random +0.198 [+0.178, +0.218] (n=500); fidelity- +0.214 [+0.183, +0.246] (n=500), vs random -0.305 [-0.335, -0.277] (n=500)
- GNNExplainer, k=2: fidelity+ +0.259 [+0.238, +0.281] (n=500), vs random +0.236 [+0.215, +0.257] (n=500); fidelity- +0.153 [+0.123, +0.184] (n=500), vs random -0.332 [-0.362, -0.303] (n=500)
- GNNExplainer, k=3: fidelity+ +0.287 [+0.266, +0.309] (n=500), vs random +0.253 [+0.233, +0.274] (n=500); fidelity- +0.104 [+0.074, +0.134] (n=500), vs random -0.351 [-0.380, -0.323] (n=500)
- GNNExplainer, k=5: fidelity+ +0.322 [+0.301, +0.344] (n=500), vs random +0.265 [+0.245, +0.286] (n=500); fidelity- +0.055 [+0.027, +0.083] (n=500), vs random -0.350 [-0.378, -0.322] (n=500)
- GNNExplainer, k=10: fidelity+ +0.378 [+0.355, +0.402] (n=494), vs random +0.263 [+0.242, +0.283] (n=494); fidelity- -0.017 [-0.040, +0.008] (n=494), vs random -0.320 [-0.345, -0.293] (n=494)
- GradExplainer, k=1: fidelity+ +0.206 [+0.186, +0.228] (n=500), vs random +0.194 [+0.175, +0.216] (n=500); fidelity- +0.170 [+0.143, +0.197] (n=500), vs random -0.349 [-0.380, -0.320] (n=500)
- GradExplainer, k=2: fidelity+ +0.334 [+0.310, +0.360] (n=500), vs random +0.311 [+0.288, +0.335] (n=500); fidelity- +0.033 [+0.008, +0.057] (n=500), vs random -0.452 [-0.483, -0.421] (n=500)
- GradExplainer, k=3: fidelity+ +0.408 [+0.383, +0.434] (n=500), vs random +0.374 [+0.351, +0.398] (n=500); fidelity- -0.023 [-0.043, -0.003] (n=500), vs random -0.478 [-0.508, -0.450] (n=500)
- GradExplainer, k=5: fidelity+ +0.489 [+0.463, +0.514] (n=500), vs random +0.432 [+0.410, +0.454] (n=500); fidelity- -0.089 [-0.104, -0.072] (n=500), vs random -0.494 [-0.519, -0.470] (n=500)
- GradExplainer, k=10: fidelity+ +0.541 [+0.517, +0.565] (n=493), vs random +0.427 [+0.406, +0.447] (n=493); fidelity- -0.124 [-0.138, -0.109] (n=493), vs random -0.429 [-0.450, -0.406] (n=493); reached by 96% of patients
- IntegratedGradExplainer, k=1: fidelity+ +0.372 [+0.348, +0.396] (n=500), vs random +0.360 [+0.337, +0.384] (n=500); fidelity- -0.077 [-0.099, -0.053] (n=500), vs random -0.596 [-0.626, -0.564] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.472 [+0.449, +0.497] (n=500), vs random +0.449 [+0.428, +0.472] (n=500); fidelity- -0.113 [-0.132, -0.093] (n=500), vs random -0.597 [-0.626, -0.569] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.512 [+0.489, +0.537] (n=500), vs random +0.479 [+0.457, +0.501] (n=500); fidelity- -0.125 [-0.143, -0.109] (n=500), vs random -0.580 [-0.606, -0.553] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.537 [+0.514, +0.562] (n=500), vs random +0.480 [+0.459, +0.502] (n=500); fidelity- -0.130 [-0.146, -0.114] (n=500), vs random -0.535 [-0.560, -0.511] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.550 [+0.526, +0.574] (n=494), vs random +0.435 [+0.414, +0.455] (n=494); fidelity- -0.134 [-0.148, -0.118] (n=494), vs random -0.437 [-0.457, -0.416] (n=494)

### top20/gsat

- builtin, k=1: fidelity+ +0.244 [+0.224, +0.265] (n=500), vs random +0.231 [+0.211, +0.251] (n=500); fidelity- +0.197 [+0.179, +0.216] (n=500), vs random -0.385 [-0.412, -0.359] (n=500)
- builtin, k=2: fidelity+ +0.381 [+0.359, +0.404] (n=500), vs random +0.355 [+0.334, +0.376] (n=500); fidelity- +0.057 [+0.040, +0.075] (n=500), vs random -0.486 [-0.513, -0.459] (n=500)
- builtin, k=3: fidelity+ +0.454 [+0.432, +0.477] (n=500), vs random +0.416 [+0.395, +0.437] (n=500); fidelity- +0.002 [-0.016, +0.019] (n=500), vs random -0.509 [-0.535, -0.482] (n=500)
- builtin, k=5: fidelity+ +0.501 [+0.479, +0.524] (n=500), vs random +0.436 [+0.416, +0.456] (n=500); fidelity- -0.034 [-0.050, -0.018] (n=500), vs random -0.491 [-0.517, -0.465] (n=500)
- builtin, k=10: fidelity+ +0.535 [+0.513, +0.558] (n=494), vs random +0.407 [+0.387, +0.427] (n=494); fidelity- -0.055 [-0.067, -0.042] (n=494), vs random -0.408 [-0.429, -0.385] (n=494)
- GNNExplainer, k=1: fidelity+ +0.109 [+0.095, +0.125] (n=500), vs random +0.096 [+0.082, +0.111] (n=500); fidelity- +0.392 [+0.366, +0.417] (n=500), vs random -0.191 [-0.214, -0.169] (n=500)
- GNNExplainer, k=2: fidelity+ +0.178 [+0.160, +0.197] (n=500), vs random +0.151 [+0.135, +0.169] (n=500); fidelity- +0.292 [+0.267, +0.317] (n=500), vs random -0.251 [-0.275, -0.229] (n=500)
- GNNExplainer, k=3: fidelity+ +0.239 [+0.218, +0.261] (n=500), vs random +0.200 [+0.182, +0.220] (n=500); fidelity- +0.216 [+0.191, +0.239] (n=500), vs random -0.295 [-0.319, -0.273] (n=500)
- GNNExplainer, k=5: fidelity+ +0.330 [+0.309, +0.354] (n=500), vs random +0.265 [+0.246, +0.286] (n=500); fidelity- +0.118 [+0.097, +0.140] (n=500), vs random -0.339 [-0.361, -0.318] (n=500)
- GNNExplainer, k=10: fidelity+ +0.454 [+0.432, +0.478] (n=494), vs random +0.327 [+0.307, +0.347] (n=494); fidelity- -0.001 [-0.019, +0.017] (n=494), vs random -0.354 [-0.374, -0.335] (n=494)
- GradExplainer, k=1: fidelity+ +0.294 [+0.272, +0.317] (n=500), vs random +0.280 [+0.259, +0.304] (n=500); fidelity- +0.083 [+0.058, +0.107] (n=500), vs random -0.500 [-0.529, -0.468] (n=500)
- GradExplainer, k=2: fidelity+ +0.418 [+0.396, +0.441] (n=500), vs random +0.391 [+0.370, +0.413] (n=500); fidelity- -0.010 [-0.028, +0.009] (n=500), vs random -0.553 [-0.580, -0.526] (n=500)
- GradExplainer, k=3: fidelity+ +0.482 [+0.460, +0.507] (n=500), vs random +0.444 [+0.423, +0.466] (n=500); fidelity- -0.046 [-0.064, -0.029] (n=500), vs random -0.557 [-0.583, -0.533] (n=500)
- GradExplainer, k=5: fidelity+ +0.531 [+0.509, +0.554] (n=500), vs random +0.466 [+0.444, +0.487] (n=500); fidelity- -0.056 [-0.072, -0.041] (n=500), vs random -0.513 [-0.539, -0.489] (n=500)
- GradExplainer, k=10: fidelity+ +0.560 [+0.538, +0.583] (n=494), vs random +0.433 [+0.412, +0.453] (n=494); fidelity- -0.061 [-0.074, -0.048] (n=494), vs random -0.415 [-0.435, -0.392] (n=494)
- IntegratedGradExplainer, k=1: fidelity+ +0.367 [+0.343, +0.391] (n=500), vs random +0.353 [+0.331, +0.377] (n=500); fidelity- -0.016 [-0.037, +0.005] (n=500), vs random -0.599 [-0.629, -0.567] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.475 [+0.453, +0.498] (n=500), vs random +0.449 [+0.428, +0.470] (n=500); fidelity- -0.066 [-0.085, -0.048] (n=500), vs random -0.609 [-0.636, -0.583] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.521 [+0.498, +0.543] (n=500), vs random +0.482 [+0.461, +0.503] (n=500); fidelity- -0.076 [-0.093, -0.060] (n=500), vs random -0.586 [-0.612, -0.561] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.547 [+0.525, +0.570] (n=500), vs random +0.482 [+0.461, +0.502] (n=500); fidelity- -0.074 [-0.090, -0.058] (n=500), vs random -0.531 [-0.556, -0.508] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.569 [+0.547, +0.592] (n=494), vs random +0.442 [+0.421, +0.462] (n=494); fidelity- -0.070 [-0.082, -0.057] (n=494), vs random -0.423 [-0.443, -0.401] (n=494)

### top20/protgnn

- builtin, k=1: fidelity+ +0.006 [+0.005, +0.008] (n=500), vs random -0.009 [-0.011, -0.007] (n=500); fidelity- +0.677 [+0.655, +0.697] (n=500), vs random -0.001 [-0.002, -0.001] (n=500)
- builtin, k=2: fidelity+ +0.013 [+0.010, +0.017] (n=490), vs random -0.018 [-0.021, -0.015] (n=490); fidelity- +0.665 [+0.644, +0.686] (n=490), vs random -0.000 [-0.002, +0.001] (n=490); reached by 75% of patients
- builtin, k=3: fidelity+ +0.014 [+0.010, +0.020] (n=426), vs random -0.026 [-0.031, -0.020] (n=426); fidelity- +0.662 [+0.639, +0.685] (n=426), vs random +0.003 [-0.001, +0.006] (n=426); reached by 41% of patients
- builtin, k=5: fidelity+ +0.016 [+0.003, +0.033] (n=141), vs random -0.039 [-0.056, -0.020] (n=141); fidelity- +0.627 [+0.585, +0.669] (n=141), vs random +0.018 [+0.002, +0.032] (n=141); reached by 10% of patients
- builtin, k=10: beyond support on every visit
- GNNExplainer, k=1: fidelity+ +0.217 [+0.197, +0.237] (n=500), vs random +0.202 [+0.182, +0.222] (n=500); fidelity- +0.683 [+0.662, +0.701] (n=500), vs random +0.003 [+0.002, +0.005] (n=500)
- GNNExplainer, k=2: fidelity+ +0.302 [+0.280, +0.323] (n=500), vs random +0.269 [+0.247, +0.289] (n=500); fidelity- +0.670 [+0.649, +0.689] (n=500), vs random +0.006 [+0.003, +0.008] (n=500)
- GNNExplainer, k=3: fidelity+ +0.347 [+0.326, +0.369] (n=500), vs random +0.299 [+0.277, +0.320] (n=500); fidelity- +0.647 [+0.625, +0.666] (n=500), vs random +0.000 [-0.004, +0.005] (n=500)
- GNNExplainer, k=5: fidelity+ +0.416 [+0.394, +0.439] (n=500), vs random +0.334 [+0.312, +0.355] (n=500); fidelity- +0.566 [+0.545, +0.586] (n=500), vs random -0.035 [-0.043, -0.027] (n=500)
- GNNExplainer, k=10: fidelity+ +0.501 [+0.480, +0.522] (n=494), vs random +0.342 [+0.322, +0.362] (n=494); fidelity- +0.288 [+0.270, +0.308] (n=494), vs random -0.171 [-0.187, -0.155] (n=494)
- GradExplainer, k=1: fidelity+ +0.173 [+0.154, +0.190] (n=500), vs random +0.157 [+0.139, +0.174] (n=500); fidelity- +0.682 [+0.662, +0.701] (n=500), vs random +0.003 [+0.002, +0.005] (n=500)
- GradExplainer, k=2: fidelity+ +0.280 [+0.259, +0.302] (n=500), vs random +0.247 [+0.227, +0.268] (n=500); fidelity- +0.672 [+0.652, +0.691] (n=500), vs random +0.008 [+0.005, +0.011] (n=500)
- GradExplainer, k=3: fidelity+ +0.371 [+0.348, +0.393] (n=500), vs random +0.322 [+0.300, +0.343] (n=500); fidelity- +0.658 [+0.638, +0.678] (n=500), vs random +0.012 [+0.008, +0.016] (n=500)
- GradExplainer, k=5: fidelity+ +0.473 [+0.450, +0.496] (n=500), vs random +0.391 [+0.368, +0.413] (n=500); fidelity- +0.607 [+0.586, +0.626] (n=500), vs random +0.005 [-0.002, +0.012] (n=500)
- GradExplainer, k=10: fidelity+ +0.567 [+0.544, +0.591] (n=494), vs random +0.409 [+0.387, +0.431] (n=494); fidelity- +0.389 [+0.370, +0.408] (n=494), vs random -0.070 [-0.084, -0.055] (n=494)
- IntegratedGradExplainer, k=1: fidelity+ +0.348 [+0.324, +0.372] (n=500), vs random +0.333 [+0.309, +0.357] (n=500); fidelity- +0.685 [+0.665, +0.704] (n=500), vs random +0.006 [+0.004, +0.007] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.466 [+0.442, +0.490] (n=500), vs random +0.433 [+0.410, +0.457] (n=500); fidelity- +0.678 [+0.657, +0.697] (n=500), vs random +0.014 [+0.011, +0.017] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.523 [+0.499, +0.546] (n=500), vs random +0.474 [+0.451, +0.496] (n=500); fidelity- +0.664 [+0.643, +0.684] (n=500), vs random +0.018 [+0.013, +0.022] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.573 [+0.551, +0.595] (n=500), vs random +0.491 [+0.469, +0.512] (n=500); fidelity- +0.610 [+0.590, +0.630] (n=500), vs random +0.009 [+0.001, +0.017] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.610 [+0.588, +0.633] (n=494), vs random +0.451 [+0.429, +0.473] (n=494); fidelity- +0.381 [+0.362, +0.400] (n=494), vs random -0.078 [-0.093, -0.063] (n=494)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

### top20/protgnn

visits with a check: 2016; with at least one removable node in the subgraph: 1784

- subgraph size: 4.6 nodes, of which removable 2.2; the visit has 48 eligible nodes
- node kinds in the subgraphs: measurement 45%, analyte 29%, knowledge 13%, visit 9%, vital 3%, diagnosis 0%

Change from the untouched visit (positive = the quantity went down). Deleting the subgraph's eligible nodes vs deleting the same number of random eligible nodes; the last column is the paired difference (subgraph - random).

| quantity | subgraph deleted | random deleted | difference |
|---|---|---|---|
| prototype similarity (activation) | +0.075 [+0.067, +0.082] (n=500) | +0.063 [+0.056, +0.071] (n=500) | +0.011 [+0.007, +0.016] (n=500) |
| prototype's share of the class logit | +0.089 [+0.079, +0.098] (n=500) | +0.076 [+0.066, +0.085] (n=500) | +0.013 [+0.008, +0.019] (n=500) |
| class probability | +0.014 [+0.011, +0.017] (n=500) | +0.035 [+0.032, +0.039] (n=500) | -0.021 [-0.025, -0.018] (n=500) |

Keeping only the subgraph's eligible nodes (all other eligible nodes deleted) vs keeping the same number of random ones: values after the intervention, as a fraction of the untouched visit's value (1 = fully preserved).

| quantity | subgraph kept | random kept | difference |
|---|---|---|---|
| prototype similarity (activation) | +0.284 [+0.275, +0.293] (n=500) | +0.267 [+0.260, +0.273] (n=500) | +0.017 [+0.013, +0.022] (n=500) |
| class probability | +0.132 [+0.122, +0.142] (n=500) | +0.135 [+0.127, +0.143] (n=500) | -0.003 [-0.008, +0.003] (n=500) |

On the untouched visits the explained prototype contributes 2.56 to the class logit, whose value is -26.97 (means over patients; the logit is the sum over all prototypes of the classifier weight times the activation).

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top20/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.514 [+0.495, +0.534] (n=500)
- GNNExplainer: +0.441 [+0.420, +0.461] (n=500)
- GradExplainer: +0.481 [+0.460, +0.503] (n=500)
- IntegratedGradExplainer: +0.495 [+0.474, +0.515] (n=500)
- GNNExplainer - GradExplainer: -0.040 [-0.055, -0.028] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.054 [-0.068, -0.041] (n=500)
- GNNExplainer - builtin: -0.073 [-0.087, -0.060] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.013 [-0.019, -0.008] (n=500)
- GradExplainer - builtin: -0.032 [-0.040, -0.026] (n=500)
- IntegratedGradExplainer - builtin: -0.019 [-0.024, -0.014] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.008 [-0.015, -0.001] (n=500)
- GNNExplainer: +0.015 [-0.002, +0.031] (n=500)
- GradExplainer: +0.049 [+0.041, +0.058] (n=500)
- IntegratedGradExplainer: +0.038 [+0.030, +0.046] (n=500)
- GNNExplainer - GradExplainer: -0.035 [-0.050, -0.018] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.023 [-0.039, -0.006] (n=500)
- GNNExplainer - builtin: +0.023 [+0.007, +0.039] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.012 [+0.006, +0.017] (n=500)
- GradExplainer - builtin: +0.058 [+0.050, +0.066] (n=500)
- IntegratedGradExplainer - builtin: +0.046 [+0.039, +0.053] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.274 [+0.264, +0.284] (n=500)
- GNNExplainer: +0.230 [+0.220, +0.240] (n=500)
- GradExplainer: +0.252 [+0.240, +0.263] (n=500)
- IntegratedGradExplainer: +0.257 [+0.245, +0.267] (n=500)
- GNNExplainer - GradExplainer: -0.022 [-0.029, -0.016] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.027 [-0.034, -0.021] (n=500)
- GNNExplainer - builtin: -0.044 [-0.051, -0.037] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.005 [-0.007, -0.003] (n=500)
- GradExplainer - builtin: -0.022 [-0.025, -0.018] (n=500)
- IntegratedGradExplainer - builtin: -0.017 [-0.021, -0.013] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.218 [-0.227, -0.209] (n=500)
- GNNExplainer: -0.207 [-0.216, -0.198] (n=500)
- GradExplainer: -0.189 [-0.199, -0.180] (n=500)
- IntegratedGradExplainer: -0.191 [-0.201, -0.182] (n=500)
- GNNExplainer - GradExplainer: -0.018 [-0.024, -0.011] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.016 [-0.022, -0.009] (n=500)
- GNNExplainer - builtin: +0.011 [+0.004, +0.018] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.002 [+0.000, +0.004] (n=500)
- GradExplainer - builtin: +0.029 [+0.025, +0.033] (n=500)
- IntegratedGradExplainer - builtin: +0.027 [+0.023, +0.031] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.742 [+0.730, +0.753] (n=500)
- GNNExplainer: +0.101 [+0.101, +0.102] (n=500)
- GradExplainer: +0.382 [+0.374, +0.390] (n=500)
- IntegratedGradExplainer: +0.326 [+0.319, +0.332] (n=500)
- GNNExplainer - GradExplainer: -0.281 [-0.289, -0.273] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.224 [-0.230, -0.218] (n=500)
- GNNExplainer - builtin: -0.641 [-0.652, -0.629] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.057 [+0.052, +0.061] (n=500)
- GradExplainer - builtin: -0.360 [-0.372, -0.348] (n=500)
- IntegratedGradExplainer - builtin: -0.416 [-0.427, -0.406] (n=500)

### top20/graphcare

**fidelity+** (higher = more faithful)

- builtin: +0.052 [+0.045, +0.058] (n=500)
- GNNExplainer: +0.320 [+0.298, +0.342] (n=500)
- GradExplainer: +0.511 [+0.488, +0.534] (n=500)
- IntegratedGradExplainer: +0.542 [+0.519, +0.566] (n=500)
- GNNExplainer - GradExplainer: -0.191 [-0.212, -0.168] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.221 [-0.244, -0.197] (n=500)
- GNNExplainer - builtin: +0.269 [+0.248, +0.291] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.031 [-0.039, -0.023] (n=500)
- GradExplainer - builtin: +0.459 [+0.437, +0.481] (n=500)
- IntegratedGradExplainer - builtin: +0.490 [+0.468, +0.512] (n=500)

**fidelity-** (lower = more faithful)

- builtin: +0.394 [+0.373, +0.416] (n=500)
- GNNExplainer: +0.058 [+0.028, +0.088] (n=500)
- GradExplainer: -0.109 [-0.125, -0.093] (n=500)
- IntegratedGradExplainer: -0.130 [-0.144, -0.115] (n=500)
- GNNExplainer - GradExplainer: +0.167 [+0.139, +0.193] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.188 [+0.160, +0.215] (n=500)
- GNNExplainer - builtin: -0.337 [-0.364, -0.309] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.021 [+0.012, +0.030] (n=500)
- GradExplainer - builtin: -0.503 [-0.526, -0.479] (n=500)
- IntegratedGradExplainer - builtin: -0.524 [-0.548, -0.501] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: -0.010 [-0.014, -0.006] (n=500)
- GNNExplainer: +0.173 [+0.159, +0.186] (n=500)
- GradExplainer: +0.342 [+0.327, +0.357] (n=500)
- IntegratedGradExplainer: +0.312 [+0.299, +0.325] (n=500)
- GNNExplainer - GradExplainer: -0.169 [-0.186, -0.154] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.139 [-0.155, -0.125] (n=500)
- GNNExplainer - builtin: +0.183 [+0.169, +0.197] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.030 [+0.025, +0.035] (n=500)
- GradExplainer - builtin: +0.352 [+0.337, +0.367] (n=500)
- IntegratedGradExplainer - builtin: +0.322 [+0.309, +0.335] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: +0.013 [+0.009, +0.017] (n=500)
- GNNExplainer: -0.179 [-0.193, -0.166] (n=500)
- GradExplainer: -0.319 [-0.334, -0.305] (n=500)
- IntegratedGradExplainer: -0.270 [-0.281, -0.260] (n=500)
- GNNExplainer - GradExplainer: +0.140 [+0.125, +0.155] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.091 [+0.079, +0.104] (n=500)
- GNNExplainer - builtin: -0.192 [-0.207, -0.177] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.049 [-0.056, -0.042] (n=500)
- GradExplainer - builtin: -0.332 [-0.347, -0.317] (n=500)
- IntegratedGradExplainer - builtin: -0.284 [-0.295, -0.272] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.216 [+0.201, +0.231] (n=500)
- GNNExplainer: +0.100 [+0.100, +0.101] (n=500)
- GradExplainer: +0.737 [+0.731, +0.743] (n=500)
- IntegratedGradExplainer: +0.546 [+0.538, +0.554] (n=500)
- GNNExplainer - GradExplainer: -0.637 [-0.642, -0.630] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.446 [-0.454, -0.438] (n=500)
- GNNExplainer - builtin: -0.115 [-0.130, -0.101] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.191 [+0.180, +0.202] (n=500)
- GradExplainer - builtin: +0.521 [+0.508, +0.534] (n=500)
- IntegratedGradExplainer - builtin: +0.330 [+0.311, +0.349] (n=500)

### top20/gsat

**fidelity+** (higher = more faithful)

- builtin: +0.521 [+0.499, +0.544] (n=500)
- GNNExplainer: +0.389 [+0.367, +0.411] (n=500)
- GradExplainer: +0.539 [+0.518, +0.562] (n=500)
- IntegratedGradExplainer: +0.558 [+0.537, +0.581] (n=500)
- GNNExplainer - GradExplainer: -0.151 [-0.172, -0.132] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.170 [-0.190, -0.151] (n=500)
- GNNExplainer - builtin: -0.132 [-0.153, -0.114] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.019 [-0.025, -0.013] (n=500)
- GradExplainer - builtin: +0.019 [+0.010, +0.028] (n=500)
- IntegratedGradExplainer - builtin: +0.037 [+0.030, +0.045] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.049 [-0.063, -0.034] (n=500)
- GNNExplainer: +0.052 [+0.030, +0.074] (n=500)
- GradExplainer: -0.051 [-0.065, -0.036] (n=500)
- IntegratedGradExplainer: -0.067 [-0.081, -0.052] (n=500)
- GNNExplainer - GradExplainer: +0.103 [+0.079, +0.128] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.119 [+0.095, +0.142] (n=500)
- GNNExplainer - builtin: +0.101 [+0.077, +0.124] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.016 [+0.008, +0.024] (n=500)
- GradExplainer - builtin: -0.002 [-0.015, +0.010] (n=500)
- IntegratedGradExplainer - builtin: -0.018 [-0.030, -0.008] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.290 [+0.277, +0.302] (n=500)
- GNNExplainer: +0.256 [+0.246, +0.267] (n=500)
- GradExplainer: +0.300 [+0.287, +0.313] (n=500)
- IntegratedGradExplainer: +0.309 [+0.296, +0.321] (n=500)
- GNNExplainer - GradExplainer: -0.044 [-0.052, -0.035] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.052 [-0.061, -0.044] (n=500)
- GNNExplainer - builtin: -0.033 [-0.042, -0.024] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.009 [-0.012, -0.006] (n=500)
- GradExplainer - builtin: +0.010 [+0.005, +0.016] (n=500)
- IntegratedGradExplainer - builtin: +0.019 [+0.014, +0.025] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.254 [-0.265, -0.243] (n=500)
- GNNExplainer: -0.242 [-0.252, -0.232] (n=500)
- GradExplainer: -0.254 [-0.265, -0.243] (n=500)
- IntegratedGradExplainer: -0.264 [-0.274, -0.253] (n=500)
- GNNExplainer - GradExplainer: +0.011 [+0.001, +0.022] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.021 [+0.011, +0.032] (n=500)
- GNNExplainer - builtin: +0.012 [+0.001, +0.022] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.010 [+0.007, +0.013] (n=500)
- GradExplainer - builtin: +0.000 [-0.004, +0.005] (n=500)
- IntegratedGradExplainer - builtin: -0.010 [-0.014, -0.005] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.097 [+0.096, +0.098] (n=500)
- GNNExplainer: +0.112 [+0.111, +0.113] (n=500)
- GradExplainer: +0.451 [+0.443, +0.460] (n=500)
- IntegratedGradExplainer: +0.396 [+0.389, +0.404] (n=500)
- GNNExplainer - GradExplainer: -0.339 [-0.348, -0.331] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.285 [-0.292, -0.277] (n=500)
- GNNExplainer - builtin: +0.015 [+0.014, +0.016] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.055 [+0.050, +0.060] (n=500)
- GradExplainer - builtin: +0.354 [+0.346, +0.363] (n=500)
- IntegratedGradExplainer - builtin: +0.300 [+0.292, +0.307] (n=500)

### top20/protgnn

**fidelity+** (higher = more faithful)

- builtin: +0.013 [+0.010, +0.015] (n=500)
- GNNExplainer: +0.440 [+0.417, +0.462] (n=500)
- GradExplainer: +0.517 [+0.493, +0.541] (n=500)
- IntegratedGradExplainer: +0.585 [+0.561, +0.607] (n=500)
- GNNExplainer - GradExplainer: -0.077 [-0.097, -0.058] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.145 [-0.163, -0.127] (n=500)
- GNNExplainer - builtin: +0.427 [+0.404, +0.451] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.068 [-0.079, -0.056] (n=500)
- GradExplainer - builtin: +0.505 [+0.480, +0.528] (n=500)
- IntegratedGradExplainer - builtin: +0.572 [+0.548, +0.596] (n=500)

**fidelity-** (lower = more faithful)

- builtin: +0.659 [+0.637, +0.679] (n=500)
- GNNExplainer: +0.403 [+0.380, +0.427] (n=500)
- GradExplainer: +0.478 [+0.457, +0.499] (n=500)
- IntegratedGradExplainer: +0.475 [+0.453, +0.497] (n=500)
- GNNExplainer - GradExplainer: -0.074 [-0.085, -0.065] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.072 [-0.081, -0.062] (n=500)
- GNNExplainer - builtin: -0.255 [-0.274, -0.236] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.003 [-0.003, +0.008] (n=500)
- GradExplainer - builtin: -0.181 [-0.198, -0.164] (n=500)
- IntegratedGradExplainer - builtin: -0.183 [-0.202, -0.165] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: -0.011 [-0.012, -0.009] (n=500)
- GNNExplainer: +0.207 [+0.195, +0.218] (n=500)
- GradExplainer: +0.253 [+0.241, +0.264] (n=500)
- IntegratedGradExplainer: +0.276 [+0.264, +0.287] (n=500)
- GNNExplainer - GradExplainer: -0.046 [-0.055, -0.037] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.069 [-0.078, -0.061] (n=500)
- GNNExplainer - builtin: +0.218 [+0.207, +0.230] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.023 [-0.027, -0.020] (n=500)
- GradExplainer - builtin: +0.264 [+0.252, +0.275] (n=500)
- IntegratedGradExplainer - builtin: +0.287 [+0.275, +0.299] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: +0.000 [-0.001, +0.001] (n=500)
- GNNExplainer: -0.132 [-0.141, -0.123] (n=500)
- GradExplainer: -0.083 [-0.092, -0.074] (n=500)
- IntegratedGradExplainer: -0.089 [-0.098, -0.080] (n=500)
- GNNExplainer - GradExplainer: -0.049 [-0.054, -0.044] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.043 [-0.048, -0.038] (n=500)
- GNNExplainer - builtin: -0.132 [-0.141, -0.124] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.006 [+0.004, +0.008] (n=500)
- GradExplainer - builtin: -0.083 [-0.092, -0.074] (n=500)
- IntegratedGradExplainer - builtin: -0.089 [-0.099, -0.080] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.931 [+0.928, +0.934] (n=500)
- GNNExplainer: +0.103 [+0.102, +0.104] (n=500)
- GradExplainer: +0.603 [+0.596, +0.611] (n=500)
- IntegratedGradExplainer: +0.560 [+0.553, +0.567] (n=500)
- GNNExplainer - GradExplainer: -0.500 [-0.508, -0.493] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.457 [-0.464, -0.450] (n=500)
- GNNExplainer - builtin: -0.828 [-0.831, -0.826] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.043 [+0.038, +0.049] (n=500)
- GradExplainer - builtin: -0.328 [-0.335, -0.321] (n=500)
- IntegratedGradExplainer - builtin: -0.371 [-0.378, -0.365] (n=500)

### Same explainer across methods (paired over patients)

- fidelity+, GNNExplainer, top20/cei_mp2 - top20/graphcare: +0.120 [+0.101, +0.139] (n=500)
- fidelity+, GNNExplainer, top20/cei_mp2 - top20/gsat: +0.052 [+0.033, +0.073] (n=500)
- fidelity+, GNNExplainer, top20/cei_mp2 - top20/protgnn: +0.001 [-0.015, +0.018] (n=500)
- fidelity+, GNNExplainer, top20/graphcare - top20/gsat: -0.068 [-0.091, -0.045] (n=500)
- fidelity+, GNNExplainer, top20/graphcare - top20/protgnn: -0.119 [-0.138, -0.100] (n=500)
- fidelity+, GNNExplainer, top20/gsat - top20/protgnn: -0.051 [-0.073, -0.029] (n=500)
- fidelity+, GradExplainer, top20/cei_mp2 - top20/graphcare: -0.030 [-0.046, -0.015] (n=500)
- fidelity+, GradExplainer, top20/cei_mp2 - top20/gsat: -0.058 [-0.073, -0.045] (n=500)
- fidelity+, GradExplainer, top20/cei_mp2 - top20/protgnn: -0.036 [-0.052, -0.017] (n=500)
- fidelity+, GradExplainer, top20/graphcare - top20/gsat: -0.028 [-0.041, -0.016] (n=500)
- fidelity+, GradExplainer, top20/graphcare - top20/protgnn: -0.006 [-0.026, +0.014] (n=500)
- fidelity+, GradExplainer, top20/gsat - top20/protgnn: +0.022 [+0.003, +0.042] (n=500)
- fidelity+, IntegratedGradExplainer, top20/cei_mp2 - top20/graphcare: -0.047 [-0.061, -0.033] (n=500)
- fidelity+, IntegratedGradExplainer, top20/cei_mp2 - top20/gsat: -0.064 [-0.075, -0.052] (n=500)
- fidelity+, IntegratedGradExplainer, top20/cei_mp2 - top20/protgnn: -0.090 [-0.103, -0.076] (n=500)
- fidelity+, IntegratedGradExplainer, top20/graphcare - top20/gsat: -0.016 [-0.028, -0.005] (n=500)
- fidelity+, IntegratedGradExplainer, top20/graphcare - top20/protgnn: -0.043 [-0.060, -0.025] (n=500)
- fidelity+, IntegratedGradExplainer, top20/gsat - top20/protgnn: -0.026 [-0.042, -0.011] (n=500)
- fidelity+, builtin, top20/cei_mp2 - top20/graphcare: +0.462 [+0.443, +0.482] (n=500)
- fidelity+, builtin, top20/cei_mp2 - top20/gsat: -0.007 [-0.021, +0.006] (n=500)
- fidelity+, builtin, top20/cei_mp2 - top20/protgnn: +0.501 [+0.480, +0.522] (n=500)
- fidelity+, builtin, top20/graphcare - top20/gsat: -0.469 [-0.490, -0.448] (n=500)
- fidelity+, builtin, top20/graphcare - top20/protgnn: +0.039 [+0.032, +0.046] (n=500)
- fidelity+, builtin, top20/gsat - top20/protgnn: +0.508 [+0.485, +0.530] (n=500)
- fidelity-, GNNExplainer, top20/cei_mp2 - top20/graphcare: -0.043 [-0.070, -0.016] (n=500)
- fidelity-, GNNExplainer, top20/cei_mp2 - top20/gsat: -0.038 [-0.060, -0.014] (n=500)
- fidelity-, GNNExplainer, top20/cei_mp2 - top20/protgnn: -0.389 [-0.415, -0.364] (n=500)
- fidelity-, GNNExplainer, top20/graphcare - top20/gsat: +0.005 [-0.024, +0.036] (n=500)
- fidelity-, GNNExplainer, top20/graphcare - top20/protgnn: -0.346 [-0.377, -0.317] (n=500)
- fidelity-, GNNExplainer, top20/gsat - top20/protgnn: -0.351 [-0.376, -0.326] (n=500)
- fidelity-, GradExplainer, top20/cei_mp2 - top20/graphcare: +0.158 [+0.142, +0.176] (n=500)
- fidelity-, GradExplainer, top20/cei_mp2 - top20/gsat: +0.100 [+0.086, +0.116] (n=500)
- fidelity-, GradExplainer, top20/cei_mp2 - top20/protgnn: -0.428 [-0.451, -0.405] (n=500)
- fidelity-, GradExplainer, top20/graphcare - top20/gsat: -0.058 [-0.075, -0.041] (n=500)
- fidelity-, GradExplainer, top20/graphcare - top20/protgnn: -0.587 [-0.611, -0.562] (n=500)
- fidelity-, GradExplainer, top20/gsat - top20/protgnn: -0.529 [-0.555, -0.503] (n=500)
- fidelity-, IntegratedGradExplainer, top20/cei_mp2 - top20/graphcare: +0.168 [+0.152, +0.184] (n=500)
- fidelity-, IntegratedGradExplainer, top20/cei_mp2 - top20/gsat: +0.105 [+0.089, +0.120] (n=500)
- fidelity-, IntegratedGradExplainer, top20/cei_mp2 - top20/protgnn: -0.437 [-0.462, -0.413] (n=500)
- fidelity-, IntegratedGradExplainer, top20/graphcare - top20/gsat: -0.063 [-0.080, -0.047] (n=500)
- fidelity-, IntegratedGradExplainer, top20/graphcare - top20/protgnn: -0.605 [-0.628, -0.582] (n=500)
- fidelity-, IntegratedGradExplainer, top20/gsat - top20/protgnn: -0.542 [-0.568, -0.517] (n=500)
- fidelity-, builtin, top20/cei_mp2 - top20/graphcare: -0.403 [-0.425, -0.380] (n=500)
- fidelity-, builtin, top20/cei_mp2 - top20/gsat: +0.041 [+0.023, +0.056] (n=500)
- fidelity-, builtin, top20/cei_mp2 - top20/protgnn: -0.667 [-0.688, -0.645] (n=500)
- fidelity-, builtin, top20/graphcare - top20/gsat: +0.443 [+0.416, +0.470] (n=500)
- fidelity-, builtin, top20/graphcare - top20/protgnn: -0.264 [-0.282, -0.246] (n=500)
- fidelity-, builtin, top20/gsat - top20/protgnn: -0.707 [-0.735, -0.679] (n=500)
- deletion vs random, GNNExplainer, top20/cei_mp2 - top20/graphcare: +0.057 [+0.045, +0.069] (n=500)
- deletion vs random, GNNExplainer, top20/cei_mp2 - top20/gsat: -0.027 [-0.035, -0.018] (n=500)
- deletion vs random, GNNExplainer, top20/cei_mp2 - top20/protgnn: +0.023 [+0.014, +0.031] (n=500)
- deletion vs random, GNNExplainer, top20/graphcare - top20/gsat: -0.084 [-0.097, -0.070] (n=500)
- deletion vs random, GNNExplainer, top20/graphcare - top20/protgnn: -0.034 [-0.046, -0.023] (n=500)
- deletion vs random, GNNExplainer, top20/gsat - top20/protgnn: +0.049 [+0.037, +0.061] (n=500)
- deletion vs random, GradExplainer, top20/cei_mp2 - top20/graphcare: -0.090 [-0.100, -0.082] (n=500)
- deletion vs random, GradExplainer, top20/cei_mp2 - top20/gsat: -0.048 [-0.054, -0.042] (n=500)
- deletion vs random, GradExplainer, top20/cei_mp2 - top20/protgnn: -0.001 [-0.009, +0.006] (n=500)
- deletion vs random, GradExplainer, top20/graphcare - top20/gsat: +0.042 [+0.034, +0.051] (n=500)
- deletion vs random, GradExplainer, top20/graphcare - top20/protgnn: +0.089 [+0.077, +0.101] (n=500)
- deletion vs random, GradExplainer, top20/gsat - top20/protgnn: +0.047 [+0.037, +0.056] (n=500)
- deletion vs random, IntegratedGradExplainer, top20/cei_mp2 - top20/graphcare: -0.056 [-0.062, -0.049] (n=500)
- deletion vs random, IntegratedGradExplainer, top20/cei_mp2 - top20/gsat: -0.052 [-0.058, -0.046] (n=500)
- deletion vs random, IntegratedGradExplainer, top20/cei_mp2 - top20/protgnn: -0.020 [-0.027, -0.012] (n=500)
- deletion vs random, IntegratedGradExplainer, top20/graphcare - top20/gsat: +0.003 [-0.003, +0.010] (n=500)
- deletion vs random, IntegratedGradExplainer, top20/graphcare - top20/protgnn: +0.036 [+0.026, +0.046] (n=500)
- deletion vs random, IntegratedGradExplainer, top20/gsat - top20/protgnn: +0.032 [+0.023, +0.042] (n=500)
- deletion vs random, builtin, top20/cei_mp2 - top20/graphcare: +0.284 [+0.272, +0.294] (n=500)
- deletion vs random, builtin, top20/cei_mp2 - top20/gsat: -0.016 [-0.023, -0.009] (n=500)
- deletion vs random, builtin, top20/cei_mp2 - top20/protgnn: +0.284 [+0.273, +0.295] (n=500)
- deletion vs random, builtin, top20/graphcare - top20/gsat: -0.299 [-0.312, -0.286] (n=500)
- deletion vs random, builtin, top20/graphcare - top20/protgnn: +0.001 [-0.004, +0.005] (n=500)
- deletion vs random, builtin, top20/gsat - top20/protgnn: +0.300 [+0.287, +0.313] (n=500)
- insertion vs random, GNNExplainer, top20/cei_mp2 - top20/graphcare: -0.028 [-0.039, -0.017] (n=500)
- insertion vs random, GNNExplainer, top20/cei_mp2 - top20/gsat: +0.035 [+0.026, +0.044] (n=500)
- insertion vs random, GNNExplainer, top20/cei_mp2 - top20/protgnn: -0.075 [-0.084, -0.066] (n=500)
- insertion vs random, GNNExplainer, top20/graphcare - top20/gsat: +0.063 [+0.050, +0.077] (n=500)
- insertion vs random, GNNExplainer, top20/graphcare - top20/protgnn: -0.047 [-0.059, -0.035] (n=500)
- insertion vs random, GNNExplainer, top20/gsat - top20/protgnn: -0.110 [-0.120, -0.099] (n=500)
- insertion vs random, GradExplainer, top20/cei_mp2 - top20/graphcare: +0.130 [+0.119, +0.141] (n=500)
- insertion vs random, GradExplainer, top20/cei_mp2 - top20/gsat: +0.064 [+0.057, +0.071] (n=500)
- insertion vs random, GradExplainer, top20/cei_mp2 - top20/protgnn: -0.106 [-0.116, -0.097] (n=500)
- insertion vs random, GradExplainer, top20/graphcare - top20/gsat: -0.065 [-0.076, -0.054] (n=500)
- insertion vs random, GradExplainer, top20/graphcare - top20/protgnn: -0.236 [-0.251, -0.221] (n=500)
- insertion vs random, GradExplainer, top20/gsat - top20/protgnn: -0.170 [-0.181, -0.160] (n=500)
- insertion vs random, IntegratedGradExplainer, top20/cei_mp2 - top20/graphcare: +0.079 [+0.071, +0.087] (n=500)
- insertion vs random, IntegratedGradExplainer, top20/cei_mp2 - top20/gsat: +0.072 [+0.065, +0.080] (n=500)
- insertion vs random, IntegratedGradExplainer, top20/cei_mp2 - top20/protgnn: -0.102 [-0.113, -0.092] (n=500)
- insertion vs random, IntegratedGradExplainer, top20/graphcare - top20/gsat: -0.007 [-0.015, +0.001] (n=500)
- insertion vs random, IntegratedGradExplainer, top20/graphcare - top20/protgnn: -0.181 [-0.193, -0.170] (n=500)
- insertion vs random, IntegratedGradExplainer, top20/gsat - top20/protgnn: -0.174 [-0.185, -0.164] (n=500)
- insertion vs random, builtin, top20/cei_mp2 - top20/graphcare: -0.231 [-0.242, -0.222] (n=500)
- insertion vs random, builtin, top20/cei_mp2 - top20/gsat: +0.036 [+0.029, +0.044] (n=500)
- insertion vs random, builtin, top20/cei_mp2 - top20/protgnn: -0.218 [-0.228, -0.209] (n=500)
- insertion vs random, builtin, top20/graphcare - top20/gsat: +0.267 [+0.256, +0.279] (n=500)
- insertion vs random, builtin, top20/graphcare - top20/protgnn: +0.013 [+0.009, +0.017] (n=500)
- insertion vs random, builtin, top20/gsat - top20/protgnn: -0.254 [-0.265, -0.243] (n=500)
- sparsity, GNNExplainer, top20/cei_mp2 - top20/graphcare: +0.001 [+0.000, +0.002] (n=500)
- sparsity, GNNExplainer, top20/cei_mp2 - top20/gsat: -0.010 [-0.011, -0.010] (n=500)
- sparsity, GNNExplainer, top20/cei_mp2 - top20/protgnn: -0.002 [-0.002, -0.001] (n=500)
- sparsity, GNNExplainer, top20/graphcare - top20/gsat: -0.011 [-0.012, -0.011] (n=500)
- sparsity, GNNExplainer, top20/graphcare - top20/protgnn: -0.003 [-0.003, -0.002] (n=500)
- sparsity, GNNExplainer, top20/gsat - top20/protgnn: +0.009 [+0.008, +0.010] (n=500)
- sparsity, GradExplainer, top20/cei_mp2 - top20/graphcare: -0.355 [-0.362, -0.348] (n=500)
- sparsity, GradExplainer, top20/cei_mp2 - top20/gsat: -0.069 [-0.076, -0.062] (n=500)
- sparsity, GradExplainer, top20/cei_mp2 - top20/protgnn: -0.221 [-0.227, -0.215] (n=500)
- sparsity, GradExplainer, top20/graphcare - top20/gsat: +0.286 [+0.278, +0.294] (n=500)
- sparsity, GradExplainer, top20/graphcare - top20/protgnn: +0.134 [+0.128, +0.139] (n=500)
- sparsity, GradExplainer, top20/gsat - top20/protgnn: -0.152 [-0.158, -0.146] (n=500)
- sparsity, IntegratedGradExplainer, top20/cei_mp2 - top20/graphcare: -0.221 [-0.230, -0.211] (n=500)
- sparsity, IntegratedGradExplainer, top20/cei_mp2 - top20/gsat: -0.071 [-0.077, -0.064] (n=500)
- sparsity, IntegratedGradExplainer, top20/cei_mp2 - top20/protgnn: -0.235 [-0.241, -0.229] (n=500)
- sparsity, IntegratedGradExplainer, top20/graphcare - top20/gsat: +0.150 [+0.140, +0.159] (n=500)
- sparsity, IntegratedGradExplainer, top20/graphcare - top20/protgnn: -0.014 [-0.023, -0.004] (n=500)
- sparsity, IntegratedGradExplainer, top20/gsat - top20/protgnn: -0.164 [-0.170, -0.157] (n=500)
- sparsity, builtin, top20/cei_mp2 - top20/graphcare: +0.526 [+0.510, +0.542] (n=500)
- sparsity, builtin, top20/cei_mp2 - top20/gsat: +0.645 [+0.634, +0.656] (n=500)
- sparsity, builtin, top20/cei_mp2 - top20/protgnn: -0.189 [-0.201, -0.178] (n=500)
- sparsity, builtin, top20/graphcare - top20/gsat: +0.119 [+0.105, +0.133] (n=500)
- sparsity, builtin, top20/graphcare - top20/protgnn: -0.715 [-0.728, -0.702] (n=500)
- sparsity, builtin, top20/gsat - top20/protgnn: -0.834 [-0.837, -0.832] (n=500)

