# Fairness checks on the explanation results

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## 1. How many nodes the main fidelity cut removes

Main cut = top 20% of the visit's eligible evidence nodes, capped at the explanation's own support. `k` is the number actually deleted; `eligible` the visit's eligible nodes; `capped` the share of visits where the support, not the 20%, set k.

**top10/cei_mp2**

- builtin: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)

**top10/graphcare**

- builtin: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)

**top10/gsat**

- builtin: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)

**top10/protgnn**

- builtin: k = 1.9, eligible = 48.6, capped by support in 92% of patients (n=500)
- GNNExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)

## 2. Matched evidence budgets

Every source scored at the same fixed k. Cells: probability drop with the top-k deleted (fidelity+, higher = more faithful) / with only the top-k kept (fidelity-, lower = more faithful); then the difference to random deletion of the same size on the same visits (+ column: higher is better; - column: lower is better). A source with fewer than k eligible nodes of nonzero importance is `beyond support` at k and is not padded.

### top10/cei_mp2

- builtin, k=1: fidelity+ +0.343 [+0.323, +0.364] (n=500), vs random +0.331 [+0.311, +0.351] (n=500); fidelity- +0.045 [+0.032, +0.058] (n=500), vs random -0.475 [-0.496, -0.454] (n=500)
- builtin, k=2: fidelity+ +0.428 [+0.408, +0.448] (n=500), vs random +0.404 [+0.385, +0.423] (n=500); fidelity- +0.006 [-0.005, +0.016] (n=500), vs random -0.485 [-0.506, -0.466] (n=500)
- builtin, k=3: fidelity+ +0.466 [+0.446, +0.486] (n=500), vs random +0.430 [+0.411, +0.449] (n=500); fidelity- -0.006 [-0.015, +0.003] (n=500), vs random -0.472 [-0.491, -0.453] (n=500)
- builtin, k=5: fidelity+ +0.495 [+0.475, +0.515] (n=500), vs random +0.434 [+0.415, +0.453] (n=500); fidelity- -0.015 [-0.023, -0.008] (n=500), vs random -0.436 [-0.453, -0.419] (n=500)
- builtin, k=10: fidelity+ +0.522 [+0.501, +0.542] (n=493), vs random +0.397 [+0.378, +0.414] (n=493); fidelity- -0.024 [-0.030, -0.019] (n=493), vs random -0.357 [-0.373, -0.339] (n=493)
- GNNExplainer, k=1: fidelity+ +0.190 [+0.174, +0.208] (n=500), vs random +0.178 [+0.162, +0.195] (n=500); fidelity- +0.256 [+0.236, +0.276] (n=500), vs random -0.264 [-0.283, -0.245] (n=500)
- GNNExplainer, k=2: fidelity+ +0.281 [+0.263, +0.300] (n=500), vs random +0.256 [+0.239, +0.276] (n=500); fidelity- +0.139 [+0.122, +0.155] (n=500), vs random -0.352 [-0.374, -0.332] (n=500)
- GNNExplainer, k=3: fidelity+ +0.334 [+0.314, +0.353] (n=500), vs random +0.298 [+0.279, +0.317] (n=500); fidelity- +0.084 [+0.069, +0.099] (n=500), vs random -0.382 [-0.403, -0.361] (n=500)
- GNNExplainer, k=5: fidelity+ +0.392 [+0.372, +0.412] (n=500), vs random +0.331 [+0.312, +0.349] (n=500); fidelity- +0.033 [+0.020, +0.046] (n=500), vs random -0.388 [-0.408, -0.367] (n=500)
- GNNExplainer, k=10: fidelity+ +0.453 [+0.432, +0.472] (n=493), vs random +0.327 [+0.309, +0.344] (n=493); fidelity- -0.006 [-0.015, +0.003] (n=493), vs random -0.338 [-0.355, -0.320] (n=493)
- GradExplainer, k=1: fidelity+ +0.256 [+0.236, +0.277] (n=500), vs random +0.244 [+0.224, +0.265] (n=500); fidelity- +0.168 [+0.150, +0.186] (n=500), vs random -0.352 [-0.373, -0.331] (n=500)
- GradExplainer, k=2: fidelity+ +0.355 [+0.334, +0.376] (n=500), vs random +0.330 [+0.310, +0.351] (n=500); fidelity- +0.071 [+0.057, +0.085] (n=500), vs random -0.420 [-0.441, -0.400] (n=500)
- GradExplainer, k=3: fidelity+ +0.406 [+0.385, +0.427] (n=500), vs random +0.370 [+0.350, +0.390] (n=500); fidelity- +0.039 [+0.028, +0.050] (n=500), vs random -0.427 [-0.447, -0.408] (n=500)
- GradExplainer, k=5: fidelity+ +0.462 [+0.442, +0.483] (n=500), vs random +0.401 [+0.382, +0.421] (n=500); fidelity- +0.020 [+0.011, +0.028] (n=500), vs random -0.401 [-0.420, -0.383] (n=500)
- GradExplainer, k=10: fidelity+ +0.489 [+0.468, +0.510] (n=493), vs random +0.364 [+0.345, +0.382] (n=493); fidelity- +0.014 [+0.008, +0.020] (n=493), vs random -0.319 [-0.336, -0.300] (n=493)
- IntegratedGradExplainer, k=1: fidelity+ +0.331 [+0.309, +0.353] (n=500), vs random +0.318 [+0.298, +0.341] (n=500); fidelity- +0.062 [+0.048, +0.075] (n=500), vs random -0.458 [-0.481, -0.436] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.404 [+0.384, +0.425] (n=500), vs random +0.380 [+0.360, +0.399] (n=500); fidelity- +0.034 [+0.023, +0.045] (n=500), vs random -0.457 [-0.477, -0.437] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.440 [+0.420, +0.460] (n=500), vs random +0.405 [+0.386, +0.423] (n=500); fidelity- +0.021 [+0.011, +0.030] (n=500), vs random -0.445 [-0.465, -0.426] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.468 [+0.448, +0.487] (n=500), vs random +0.407 [+0.388, +0.425] (n=500); fidelity- +0.017 [+0.010, +0.025] (n=500), vs random -0.404 [-0.422, -0.386] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.487 [+0.466, +0.509] (n=493), vs random +0.362 [+0.343, +0.380] (n=493); fidelity- +0.010 [+0.004, +0.016] (n=493), vs random -0.322 [-0.339, -0.304] (n=493)

### top10/graphcare

- builtin, k=1: fidelity+ +0.008 [+0.005, +0.010] (n=500), vs random -0.003 [-0.006, -0.000] (n=500); fidelity- +0.540 [+0.515, +0.565] (n=500), vs random +0.001 [-0.007, +0.009] (n=500)
- builtin, k=2: fidelity+ +0.023 [+0.018, +0.028] (n=500), vs random +0.001 [-0.004, +0.006] (n=500); fidelity- +0.503 [+0.480, +0.527] (n=500), vs random +0.004 [-0.007, +0.013] (n=500)
- builtin, k=3: fidelity+ +0.030 [+0.025, +0.036] (n=500), vs random -0.002 [-0.007, +0.002] (n=500); fidelity- +0.470 [+0.448, +0.492] (n=500), vs random +0.005 [-0.006, +0.014] (n=500)
- builtin, k=5: fidelity+ +0.053 [+0.045, +0.062] (n=500), vs random -0.004 [-0.011, +0.002] (n=500); fidelity- +0.417 [+0.397, +0.439] (n=500), vs random +0.007 [-0.005, +0.018] (n=500)
- builtin, k=10: fidelity+ +0.113 [+0.099, +0.130] (n=493), vs random -0.009 [-0.017, +0.000] (n=493); fidelity- +0.319 [+0.298, +0.338] (n=493), vs random +0.011 [+0.000, +0.022] (n=493)
- GNNExplainer, k=1: fidelity+ +0.198 [+0.178, +0.220] (n=500), vs random +0.187 [+0.168, +0.209] (n=500); fidelity- +0.256 [+0.223, +0.288] (n=500), vs random -0.283 [-0.315, -0.253] (n=500)
- GNNExplainer, k=2: fidelity+ +0.227 [+0.207, +0.250] (n=500), vs random +0.205 [+0.185, +0.227] (n=500); fidelity- +0.202 [+0.170, +0.233] (n=500), vs random -0.297 [-0.329, -0.268] (n=500)
- GNNExplainer, k=3: fidelity+ +0.248 [+0.227, +0.272] (n=500), vs random +0.215 [+0.195, +0.238] (n=500); fidelity- +0.174 [+0.142, +0.204] (n=500), vs random -0.292 [-0.325, -0.263] (n=500)
- GNNExplainer, k=5: fidelity+ +0.278 [+0.255, +0.301] (n=500), vs random +0.220 [+0.199, +0.244] (n=500); fidelity- +0.118 [+0.087, +0.147] (n=500), vs random -0.293 [-0.326, -0.264] (n=500)
- GNNExplainer, k=10: fidelity+ +0.337 [+0.312, +0.363] (n=493), vs random +0.215 [+0.191, +0.237] (n=493); fidelity- +0.039 [+0.014, +0.061] (n=493), vs random -0.269 [-0.296, -0.243] (n=493)
- GradExplainer, k=1: fidelity+ +0.189 [+0.168, +0.210] (n=500), vs random +0.178 [+0.158, +0.198] (n=500); fidelity- +0.179 [+0.152, +0.208] (n=500), vs random -0.360 [-0.393, -0.327] (n=500)
- GradExplainer, k=2: fidelity+ +0.315 [+0.292, +0.339] (n=500), vs random +0.293 [+0.271, +0.316] (n=500); fidelity- +0.063 [+0.038, +0.087] (n=500), vs random -0.436 [-0.468, -0.405] (n=500)
- GradExplainer, k=3: fidelity+ +0.401 [+0.377, +0.425] (n=500), vs random +0.368 [+0.347, +0.391] (n=500); fidelity- -0.002 [-0.022, +0.019] (n=500), vs random -0.468 [-0.495, -0.441] (n=500)
- GradExplainer, k=5: fidelity+ +0.484 [+0.459, +0.507] (n=500), vs random +0.426 [+0.404, +0.448] (n=500); fidelity- -0.064 [-0.082, -0.047] (n=500), vs random -0.475 [-0.500, -0.451] (n=500)
- GradExplainer, k=10: fidelity+ +0.545 [+0.522, +0.569] (n=493), vs random +0.423 [+0.403, +0.442] (n=493); fidelity- -0.102 [-0.115, -0.088] (n=493), vs random -0.410 [-0.430, -0.388] (n=493)
- IntegratedGradExplainer, k=1: fidelity+ +0.355 [+0.330, +0.379] (n=500), vs random +0.344 [+0.320, +0.368] (n=500); fidelity- -0.028 [-0.050, -0.005] (n=500), vs random -0.567 [-0.599, -0.535] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.453 [+0.429, +0.476] (n=500), vs random +0.431 [+0.408, +0.453] (n=500); fidelity- -0.056 [-0.077, -0.037] (n=500), vs random -0.555 [-0.583, -0.527] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.496 [+0.474, +0.519] (n=500), vs random +0.463 [+0.442, +0.484] (n=500); fidelity- -0.073 [-0.091, -0.055] (n=500), vs random -0.539 [-0.563, -0.514] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.526 [+0.504, +0.549] (n=500), vs random +0.469 [+0.449, +0.490] (n=500); fidelity- -0.096 [-0.111, -0.080] (n=500), vs random -0.506 [-0.529, -0.485] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.557 [+0.535, +0.579] (n=493), vs random +0.435 [+0.415, +0.453] (n=493); fidelity- -0.113 [-0.127, -0.099] (n=493), vs random -0.421 [-0.440, -0.401] (n=493)

### top10/gsat

- builtin, k=1: fidelity+ +0.273 [+0.251, +0.295] (n=500), vs random +0.261 [+0.240, +0.283] (n=500); fidelity- +0.122 [+0.099, +0.144] (n=500), vs random -0.434 [-0.465, -0.404] (n=500)
- builtin, k=2: fidelity+ +0.406 [+0.383, +0.429] (n=500), vs random +0.382 [+0.361, +0.404] (n=500); fidelity- +0.017 [-0.003, +0.036] (n=500), vs random -0.506 [-0.534, -0.480] (n=500)
- builtin, k=3: fidelity+ +0.461 [+0.438, +0.484] (n=500), vs random +0.427 [+0.405, +0.449] (n=500); fidelity- -0.023 [-0.039, -0.006] (n=500), vs random -0.518 [-0.544, -0.493] (n=500)
- builtin, k=5: fidelity+ +0.496 [+0.475, +0.518] (n=500), vs random +0.438 [+0.418, +0.459] (n=500); fidelity- -0.053 [-0.068, -0.039] (n=500), vs random -0.495 [-0.518, -0.473] (n=500)
- builtin, k=10: fidelity+ +0.533 [+0.511, +0.555] (n=493), vs random +0.411 [+0.390, +0.431] (n=493); fidelity- -0.062 [-0.074, -0.050] (n=493), vs random -0.401 [-0.421, -0.381] (n=493)
- GNNExplainer, k=1: fidelity+ +0.161 [+0.142, +0.181] (n=500), vs random +0.149 [+0.130, +0.169] (n=500); fidelity- +0.294 [+0.269, +0.319] (n=500), vs random -0.262 [-0.288, -0.237] (n=500)
- GNNExplainer, k=2: fidelity+ +0.247 [+0.225, +0.271] (n=500), vs random +0.223 [+0.202, +0.247] (n=500); fidelity- +0.190 [+0.169, +0.212] (n=500), vs random -0.333 [-0.360, -0.307] (n=500)
- GNNExplainer, k=3: fidelity+ +0.308 [+0.285, +0.332] (n=500), vs random +0.273 [+0.252, +0.296] (n=500); fidelity- +0.118 [+0.099, +0.138] (n=500), vs random -0.377 [-0.402, -0.353] (n=500)
- GNNExplainer, k=5: fidelity+ +0.395 [+0.372, +0.420] (n=500), vs random +0.337 [+0.315, +0.359] (n=500); fidelity- +0.042 [+0.026, +0.058] (n=500), vs random -0.400 [-0.422, -0.379] (n=500)
- GNNExplainer, k=10: fidelity+ +0.484 [+0.460, +0.510] (n=493), vs random +0.363 [+0.342, +0.384] (n=493); fidelity- -0.034 [-0.048, -0.021] (n=493), vs random -0.373 [-0.391, -0.354] (n=493)
- GradExplainer, k=1: fidelity+ +0.270 [+0.248, +0.293] (n=500), vs random +0.258 [+0.237, +0.282] (n=500); fidelity- +0.098 [+0.071, +0.123] (n=500), vs random -0.458 [-0.493, -0.425] (n=500)
- GradExplainer, k=2: fidelity+ +0.402 [+0.379, +0.425] (n=500), vs random +0.379 [+0.357, +0.401] (n=500); fidelity- -0.010 [-0.029, +0.010] (n=500), vs random -0.533 [-0.560, -0.505] (n=500)
- GradExplainer, k=3: fidelity+ +0.473 [+0.450, +0.496] (n=500), vs random +0.438 [+0.417, +0.460] (n=500); fidelity- -0.050 [-0.068, -0.033] (n=500), vs random -0.545 [-0.570, -0.520] (n=500)
- GradExplainer, k=5: fidelity+ +0.518 [+0.496, +0.540] (n=500), vs random +0.459 [+0.439, +0.479] (n=500); fidelity- -0.075 [-0.090, -0.061] (n=500), vs random -0.517 [-0.538, -0.495] (n=500)
- GradExplainer, k=10: fidelity+ +0.546 [+0.523, +0.568] (n=493), vs random +0.424 [+0.404, +0.444] (n=493); fidelity- -0.074 [-0.085, -0.063] (n=493), vs random -0.413 [-0.432, -0.392] (n=493)
- IntegratedGradExplainer, k=1: fidelity+ +0.339 [+0.316, +0.362] (n=500), vs random +0.327 [+0.305, +0.349] (n=500); fidelity- +0.013 [-0.011, +0.038] (n=500), vs random -0.542 [-0.576, -0.509] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.442 [+0.419, +0.465] (n=500), vs random +0.418 [+0.397, +0.440] (n=500); fidelity- -0.044 [-0.061, -0.028] (n=500), vs random -0.567 [-0.593, -0.541] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.495 [+0.472, +0.517] (n=500), vs random +0.460 [+0.439, +0.481] (n=500); fidelity- -0.061 [-0.078, -0.045] (n=500), vs random -0.556 [-0.581, -0.531] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.526 [+0.504, +0.549] (n=500), vs random +0.468 [+0.447, +0.489] (n=500); fidelity- -0.075 [-0.090, -0.060] (n=500), vs random -0.517 [-0.539, -0.495] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.549 [+0.525, +0.570] (n=493), vs random +0.427 [+0.406, +0.447] (n=493); fidelity- -0.074 [-0.084, -0.062] (n=493), vs random -0.413 [-0.432, -0.391] (n=493)

### top10/protgnn

- builtin, k=1: fidelity+ +0.006 [+0.005, +0.008] (n=500), vs random -0.009 [-0.011, -0.007] (n=500); fidelity- +0.662 [+0.643, +0.680] (n=500), vs random -0.003 [-0.004, -0.001] (n=500)
- builtin, k=2: fidelity+ +0.015 [+0.011, +0.020] (n=447), vs random -0.015 [-0.019, -0.012] (n=447); fidelity- +0.625 [+0.605, +0.646] (n=447), vs random -0.007 [-0.011, -0.003] (n=447); reached by 60% of patients
- builtin, k=3: fidelity+ +0.022 [+0.015, +0.031] (n=320), vs random -0.019 [-0.026, -0.012] (n=320); fidelity- +0.583 [+0.556, +0.611] (n=320), vs random -0.014 [-0.022, -0.005] (n=320); reached by 28% of patients
- builtin, k=5: fidelity+ +0.050 [+0.016, +0.090] (n=24), vs random +0.020 [-0.017, +0.063] (n=24); fidelity- +0.430 [+0.308, +0.555] (n=24), vs random -0.051 [-0.117, +0.010] (n=24); reached by 2% of patients
- builtin, k=10: beyond support on every visit
- GNNExplainer, k=1: fidelity+ +0.157 [+0.141, +0.175] (n=500), vs random +0.143 [+0.126, +0.161] (n=500); fidelity- +0.669 [+0.651, +0.688] (n=500), vs random +0.005 [+0.003, +0.007] (n=500)
- GNNExplainer, k=2: fidelity+ +0.239 [+0.219, +0.260] (n=500), vs random +0.208 [+0.189, +0.228] (n=500); fidelity- +0.645 [+0.626, +0.664] (n=500), vs random +0.006 [+0.002, +0.012] (n=500)
- GNNExplainer, k=3: fidelity+ +0.290 [+0.269, +0.311] (n=500), vs random +0.244 [+0.225, +0.265] (n=500); fidelity- +0.612 [+0.592, +0.630] (n=500), vs random -0.002 [-0.009, +0.005] (n=500)
- GNNExplainer, k=5: fidelity+ +0.365 [+0.344, +0.388] (n=500), vs random +0.287 [+0.267, +0.310] (n=500); fidelity- +0.519 [+0.499, +0.539] (n=500), vs random -0.038 [-0.050, -0.027] (n=500)
- GNNExplainer, k=10: fidelity+ +0.454 [+0.432, +0.476] (n=493), vs random +0.304 [+0.284, +0.324] (n=493); fidelity- +0.248 [+0.230, +0.265] (n=493), vs random -0.164 [-0.180, -0.148] (n=493)
- GradExplainer, k=1: fidelity+ +0.133 [+0.117, +0.150] (n=500), vs random +0.118 [+0.102, +0.135] (n=500); fidelity- +0.675 [+0.656, +0.692] (n=500), vs random +0.010 [+0.008, +0.013] (n=500)
- GradExplainer, k=2: fidelity+ +0.241 [+0.221, +0.262] (n=500), vs random +0.210 [+0.190, +0.230] (n=500); fidelity- +0.656 [+0.638, +0.675] (n=500), vs random +0.018 [+0.013, +0.023] (n=500)
- GradExplainer, k=3: fidelity+ +0.304 [+0.282, +0.326] (n=500), vs random +0.258 [+0.236, +0.279] (n=500); fidelity- +0.633 [+0.614, +0.651] (n=500), vs random +0.019 [+0.012, +0.027] (n=500)
- GradExplainer, k=5: fidelity+ +0.399 [+0.377, +0.421] (n=500), vs random +0.320 [+0.300, +0.342] (n=500); fidelity- +0.561 [+0.542, +0.580] (n=500), vs random +0.003 [-0.009, +0.014] (n=500)
- GradExplainer, k=10: fidelity+ +0.507 [+0.485, +0.530] (n=493), vs random +0.358 [+0.336, +0.378] (n=493); fidelity- +0.300 [+0.282, +0.318] (n=493), vs random -0.112 [-0.128, -0.095] (n=493)
- IntegratedGradExplainer, k=1: fidelity+ +0.315 [+0.292, +0.339] (n=500), vs random +0.300 [+0.278, +0.324] (n=500); fidelity- +0.679 [+0.660, +0.697] (n=500), vs random +0.015 [+0.012, +0.018] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.419 [+0.395, +0.443] (n=500), vs random +0.388 [+0.365, +0.411] (n=500); fidelity- +0.667 [+0.648, +0.686] (n=500), vs random +0.028 [+0.022, +0.034] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.472 [+0.450, +0.495] (n=500), vs random +0.426 [+0.405, +0.448] (n=500); fidelity- +0.645 [+0.625, +0.664] (n=500), vs random +0.032 [+0.023, +0.040] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.521 [+0.499, +0.543] (n=500), vs random +0.443 [+0.421, +0.465] (n=500); fidelity- +0.570 [+0.550, +0.591] (n=500), vs random +0.013 [-0.001, +0.026] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.564 [+0.541, +0.585] (n=493), vs random +0.415 [+0.391, +0.435] (n=493); fidelity- +0.305 [+0.286, +0.324] (n=493), vs random -0.107 [-0.125, -0.089] (n=493)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

### top10/protgnn

visits with a check: 1902; with at least one removable node in the subgraph: 1743

- subgraph size: 4.2 nodes, of which removable 1.8; the visit has 48 eligible nodes
- node kinds in the subgraphs: measurement 37%, analyte 23%, knowledge 23%, visit 9%, vital 8%, complaint 0%, diagnosis 0%

Change from the untouched visit (positive = the quantity went down). Deleting the subgraph's eligible nodes vs deleting the same number of random eligible nodes; the last column is the paired difference (subgraph - random).

| quantity | subgraph deleted | random deleted | difference |
|---|---|---|---|
| prototype similarity (activation) | +0.059 [+0.051, +0.069] (n=500) | +0.058 [+0.050, +0.067] (n=500) | +0.001 [-0.003, +0.005] (n=500) |
| prototype's share of the class logit | +0.070 [+0.060, +0.083] (n=500) | +0.069 [+0.059, +0.081] (n=500) | +0.001 [-0.004, +0.007] (n=500) |
| class probability | +0.016 [+0.012, +0.020] (n=500) | +0.029 [+0.025, +0.033] (n=500) | -0.013 [-0.017, -0.010] (n=500) |

Keeping only the subgraph's eligible nodes (all other eligible nodes deleted) vs keeping the same number of random ones: values after the intervention, as a fraction of the untouched visit's value (1 = fully preserved).

| quantity | subgraph kept | random kept | difference |
|---|---|---|---|
| prototype similarity (activation) | +0.253 [+0.243, +0.264] (n=500) | +0.243 [+0.236, +0.250] (n=500) | +0.010 [+0.005, +0.016] (n=500) |
| class probability | +0.226 [+0.214, +0.238] (n=500) | +0.215 [+0.205, +0.225] (n=500) | +0.011 [+0.007, +0.015] (n=500) |

On the untouched visits the explained prototype contributes 2.21 to the class logit, whose value is -8.37 (means over patients; the logit is the sum over all prototypes of the classifier weight times the activation).

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top10/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.498 [+0.478, +0.518] (n=500)
- GNNExplainer: +0.397 [+0.378, +0.417] (n=500)
- GradExplainer: +0.463 [+0.443, +0.484] (n=500)
- IntegratedGradExplainer: +0.475 [+0.456, +0.495] (n=500)
- GNNExplainer - GradExplainer: -0.066 [-0.079, -0.053] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.078 [-0.093, -0.064] (n=500)
- GNNExplainer - builtin: -0.101 [-0.116, -0.087] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.012 [-0.020, -0.005] (n=500)
- GradExplainer - builtin: -0.035 [-0.042, -0.027] (n=500)
- IntegratedGradExplainer - builtin: -0.023 [-0.029, -0.017] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.014 [-0.020, -0.008] (n=500)
- GNNExplainer: +0.045 [+0.030, +0.060] (n=500)
- GradExplainer: +0.029 [+0.021, +0.037] (n=500)
- IntegratedGradExplainer: +0.017 [+0.011, +0.024] (n=500)
- GNNExplainer - GradExplainer: +0.015 [+0.002, +0.029] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.028 [+0.014, +0.042] (n=500)
- GNNExplainer - builtin: +0.059 [+0.045, +0.072] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.012 [+0.006, +0.019] (n=500)
- GradExplainer - builtin: +0.043 [+0.036, +0.050] (n=500)
- IntegratedGradExplainer - builtin: +0.031 [+0.025, +0.036] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.272 [+0.261, +0.283] (n=500)
- GNNExplainer: +0.203 [+0.193, +0.213] (n=500)
- GradExplainer: +0.243 [+0.233, +0.254] (n=500)
- IntegratedGradExplainer: +0.247 [+0.236, +0.258] (n=500)
- GNNExplainer - GradExplainer: -0.040 [-0.047, -0.033] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.044 [-0.052, -0.036] (n=500)
- GNNExplainer - builtin: -0.068 [-0.077, -0.060] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.004 [-0.007, -0.001] (n=500)
- GradExplainer - builtin: -0.028 [-0.033, -0.024] (n=500)
- IntegratedGradExplainer - builtin: -0.025 [-0.029, -0.020] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.213 [-0.221, -0.205] (n=500)
- GNNExplainer: -0.185 [-0.194, -0.177] (n=500)
- GradExplainer: -0.192 [-0.201, -0.184] (n=500)
- IntegratedGradExplainer: -0.196 [-0.205, -0.188] (n=500)
- GNNExplainer - GradExplainer: +0.007 [+0.001, +0.012] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.011 [+0.005, +0.017] (n=500)
- GNNExplainer - builtin: +0.028 [+0.022, +0.034] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.004 [+0.002, +0.007] (n=500)
- GradExplainer - builtin: +0.021 [+0.018, +0.025] (n=500)
- IntegratedGradExplainer - builtin: +0.017 [+0.014, +0.020] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.700 [+0.684, +0.715] (n=500)
- GNNExplainer: +0.100 [+0.100, +0.101] (n=500)
- GradExplainer: +0.418 [+0.409, +0.427] (n=500)
- IntegratedGradExplainer: +0.385 [+0.377, +0.394] (n=500)
- GNNExplainer - GradExplainer: -0.318 [-0.327, -0.309] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.285 [-0.294, -0.277] (n=500)
- GNNExplainer - builtin: -0.599 [-0.615, -0.584] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.033 [+0.027, +0.039] (n=500)
- GradExplainer - builtin: -0.282 [-0.297, -0.267] (n=500)
- IntegratedGradExplainer - builtin: -0.315 [-0.327, -0.302] (n=500)

### top10/graphcare

**fidelity+** (higher = more faithful)

- builtin: +0.047 [+0.041, +0.054] (n=500)
- GNNExplainer: +0.275 [+0.253, +0.299] (n=500)
- GradExplainer: +0.506 [+0.483, +0.530] (n=500)
- IntegratedGradExplainer: +0.537 [+0.515, +0.560] (n=500)
- GNNExplainer - GradExplainer: -0.231 [-0.256, -0.206] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.262 [-0.288, -0.237] (n=500)
- GNNExplainer - builtin: +0.228 [+0.206, +0.251] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.031 [-0.040, -0.022] (n=500)
- GradExplainer - builtin: +0.459 [+0.436, +0.482] (n=500)
- IntegratedGradExplainer - builtin: +0.490 [+0.468, +0.512] (n=500)

**fidelity-** (lower = more faithful)

- builtin: +0.409 [+0.389, +0.429] (n=500)
- GNNExplainer: +0.130 [+0.098, +0.162] (n=500)
- GradExplainer: -0.081 [-0.098, -0.063] (n=500)
- IntegratedGradExplainer: -0.110 [-0.125, -0.096] (n=500)
- GNNExplainer - GradExplainer: +0.211 [+0.180, +0.239] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.240 [+0.211, +0.269] (n=500)
- GNNExplainer - builtin: -0.279 [-0.309, -0.250] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.030 [+0.018, +0.042] (n=500)
- GradExplainer - builtin: -0.490 [-0.512, -0.466] (n=500)
- IntegratedGradExplainer - builtin: -0.519 [-0.539, -0.498] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: -0.014 [-0.020, -0.009] (n=500)
- GNNExplainer: +0.143 [+0.130, +0.158] (n=500)
- GradExplainer: +0.309 [+0.297, +0.322] (n=500)
- IntegratedGradExplainer: +0.318 [+0.306, +0.331] (n=500)
- GNNExplainer - GradExplainer: -0.167 [-0.183, -0.150] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.175 [-0.192, -0.159] (n=500)
- GNNExplainer - builtin: +0.157 [+0.142, +0.173] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.009 [-0.012, -0.006] (n=500)
- GradExplainer - builtin: +0.324 [+0.310, +0.337] (n=500)
- IntegratedGradExplainer - builtin: +0.333 [+0.319, +0.346] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: +0.021 [+0.015, +0.028] (n=500)
- GNNExplainer: -0.148 [-0.163, -0.135] (n=500)
- GradExplainer: -0.252 [-0.262, -0.243] (n=500)
- IntegratedGradExplainer: -0.262 [-0.271, -0.253] (n=500)
- GNNExplainer - GradExplainer: +0.104 [+0.091, +0.117] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.113 [+0.100, +0.126] (n=500)
- GNNExplainer - builtin: -0.170 [-0.186, -0.154] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.009 [+0.006, +0.013] (n=500)
- GradExplainer - builtin: -0.274 [-0.285, -0.263] (n=500)
- IntegratedGradExplainer - builtin: -0.283 [-0.294, -0.272] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.208 [+0.194, +0.223] (n=500)
- GNNExplainer: +0.099 [+0.098, +0.100] (n=500)
- GradExplainer: +0.680 [+0.672, +0.688] (n=500)
- IntegratedGradExplainer: +0.547 [+0.538, +0.556] (n=500)
- GNNExplainer - GradExplainer: -0.580 [-0.589, -0.572] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.448 [-0.457, -0.438] (n=500)
- GNNExplainer - builtin: -0.109 [-0.124, -0.094] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.133 [+0.124, +0.143] (n=500)
- GradExplainer - builtin: +0.471 [+0.459, +0.484] (n=500)
- IntegratedGradExplainer - builtin: +0.339 [+0.321, +0.355] (n=500)

### top10/gsat

**fidelity+** (higher = more faithful)

- builtin: +0.524 [+0.502, +0.545] (n=500)
- GNNExplainer: +0.433 [+0.410, +0.457] (n=500)
- GradExplainer: +0.529 [+0.507, +0.551] (n=500)
- IntegratedGradExplainer: +0.537 [+0.514, +0.560] (n=500)
- GNNExplainer - GradExplainer: -0.096 [-0.114, -0.078] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.104 [-0.124, -0.085] (n=500)
- GNNExplainer - builtin: -0.091 [-0.110, -0.072] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.008 [-0.014, -0.003] (n=500)
- GradExplainer - builtin: +0.005 [-0.003, +0.013] (n=500)
- IntegratedGradExplainer - builtin: +0.013 [+0.004, +0.022] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.060 [-0.074, -0.046] (n=500)
- GNNExplainer: +0.003 [-0.015, +0.021] (n=500)
- GradExplainer: -0.071 [-0.084, -0.057] (n=500)
- IntegratedGradExplainer: -0.070 [-0.083, -0.056] (n=500)
- GNNExplainer - GradExplainer: +0.074 [+0.054, +0.093] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.073 [+0.052, +0.093] (n=500)
- GNNExplainer - builtin: +0.063 [+0.042, +0.085] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.001 [-0.008, +0.006] (n=500)
- GradExplainer - builtin: -0.010 [-0.021, -0.001] (n=500)
- IntegratedGradExplainer - builtin: -0.009 [-0.021, +0.002] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.305 [+0.293, +0.317] (n=500)
- GNNExplainer: +0.271 [+0.259, +0.283] (n=500)
- GradExplainer: +0.299 [+0.286, +0.311] (n=500)
- IntegratedGradExplainer: +0.304 [+0.291, +0.317] (n=500)
- GNNExplainer - GradExplainer: -0.028 [-0.037, -0.020] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.033 [-0.042, -0.025] (n=500)
- GNNExplainer - builtin: -0.035 [-0.043, -0.026] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.005 [-0.008, -0.002] (n=500)
- GradExplainer - builtin: -0.007 [-0.012, -0.001] (n=500)
- IntegratedGradExplainer - builtin: -0.001 [-0.007, +0.004] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.244 [-0.254, -0.234] (n=500)
- GNNExplainer: -0.238 [-0.247, -0.230] (n=500)
- GradExplainer: -0.247 [-0.257, -0.238] (n=500)
- IntegratedGradExplainer: -0.247 [-0.257, -0.237] (n=500)
- GNNExplainer - GradExplainer: +0.009 [+0.001, +0.017] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.009 [+0.001, +0.018] (n=500)
- GNNExplainer - builtin: +0.006 [-0.003, +0.015] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.000 [-0.002, +0.002] (n=500)
- GradExplainer - builtin: -0.003 [-0.007, +0.001] (n=500)
- IntegratedGradExplainer - builtin: -0.003 [-0.007, +0.001] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.095 [+0.094, +0.096] (n=500)
- GNNExplainer: +0.110 [+0.109, +0.111] (n=500)
- GradExplainer: +0.490 [+0.481, +0.499] (n=500)
- IntegratedGradExplainer: +0.450 [+0.441, +0.459] (n=500)
- GNNExplainer - GradExplainer: -0.380 [-0.390, -0.371] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.340 [-0.350, -0.331] (n=500)
- GNNExplainer - builtin: +0.015 [+0.014, +0.016] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.040 [+0.034, +0.046] (n=500)
- GradExplainer - builtin: +0.395 [+0.387, +0.404] (n=500)
- IntegratedGradExplainer - builtin: +0.355 [+0.346, +0.365] (n=500)

### top10/protgnn

**fidelity+** (higher = more faithful)

- builtin: +0.013 [+0.010, +0.016] (n=500)
- GNNExplainer: +0.392 [+0.370, +0.414] (n=500)
- GradExplainer: +0.456 [+0.434, +0.480] (n=500)
- IntegratedGradExplainer: +0.541 [+0.518, +0.564] (n=500)
- GNNExplainer - GradExplainer: -0.065 [-0.084, -0.045] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.149 [-0.169, -0.130] (n=500)
- GNNExplainer - builtin: +0.379 [+0.355, +0.404] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.084 [-0.101, -0.069] (n=500)
- GradExplainer - builtin: +0.444 [+0.419, +0.469] (n=500)
- IntegratedGradExplainer - builtin: +0.528 [+0.504, +0.553] (n=500)

**fidelity-** (lower = more faithful)

- builtin: +0.632 [+0.614, +0.651] (n=500)
- GNNExplainer: +0.357 [+0.333, +0.378] (n=500)
- GradExplainer: +0.406 [+0.383, +0.429] (n=500)
- IntegratedGradExplainer: +0.416 [+0.390, +0.439] (n=500)
- GNNExplainer - GradExplainer: -0.049 [-0.061, -0.038] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.059 [-0.070, -0.047] (n=500)
- GNNExplainer - builtin: -0.276 [-0.297, -0.256] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.009 [-0.017, -0.002] (n=500)
- GradExplainer - builtin: -0.226 [-0.245, -0.207] (n=500)
- IntegratedGradExplainer - builtin: -0.217 [-0.237, -0.197] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: -0.007 [-0.008, -0.005] (n=500)
- GNNExplainer: +0.186 [+0.175, +0.197] (n=500)
- GradExplainer: +0.231 [+0.219, +0.243] (n=500)
- IntegratedGradExplainer: +0.257 [+0.245, +0.270] (n=500)
- GNNExplainer - GradExplainer: -0.045 [-0.054, -0.037] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.072 [-0.081, -0.062] (n=500)
- GNNExplainer - builtin: +0.193 [+0.181, +0.205] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.026 [-0.031, -0.022] (n=500)
- GradExplainer - builtin: +0.238 [+0.225, +0.251] (n=500)
- IntegratedGradExplainer - builtin: +0.264 [+0.251, +0.278] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.003 [-0.004, -0.002] (n=500)
- GNNExplainer: -0.117 [-0.126, -0.108] (n=500)
- GradExplainer: -0.096 [-0.107, -0.087] (n=500)
- IntegratedGradExplainer: -0.095 [-0.105, -0.085] (n=500)
- GNNExplainer - GradExplainer: -0.020 [-0.025, -0.016] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.022 [-0.027, -0.017] (n=500)
- GNNExplainer - builtin: -0.114 [-0.123, -0.105] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.002 [-0.004, +0.000] (n=500)
- GradExplainer - builtin: -0.093 [-0.103, -0.083] (n=500)
- IntegratedGradExplainer - builtin: -0.091 [-0.101, -0.081] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.942 [+0.938, +0.946] (n=500)
- GNNExplainer: +0.101 [+0.100, +0.102] (n=500)
- GradExplainer: +0.628 [+0.619, +0.639] (n=500)
- IntegratedGradExplainer: +0.608 [+0.600, +0.616] (n=500)
- GNNExplainer - GradExplainer: -0.527 [-0.538, -0.517] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.507 [-0.515, -0.498] (n=500)
- GNNExplainer - builtin: -0.841 [-0.844, -0.837] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.020 [+0.013, +0.028] (n=500)
- GradExplainer - builtin: -0.314 [-0.322, -0.305] (n=500)
- IntegratedGradExplainer - builtin: -0.334 [-0.342, -0.327] (n=500)

### Same explainer across methods (paired over patients)

- fidelity+, GNNExplainer, top10/cei_mp2 - top10/graphcare: +0.122 [+0.102, +0.141] (n=500)
- fidelity+, GNNExplainer, top10/cei_mp2 - top10/gsat: -0.035 [-0.054, -0.017] (n=500)
- fidelity+, GNNExplainer, top10/cei_mp2 - top10/protgnn: +0.006 [-0.014, +0.025] (n=500)
- fidelity+, GNNExplainer, top10/graphcare - top10/gsat: -0.157 [-0.181, -0.133] (n=500)
- fidelity+, GNNExplainer, top10/graphcare - top10/protgnn: -0.117 [-0.136, -0.095] (n=500)
- fidelity+, GNNExplainer, top10/gsat - top10/protgnn: +0.041 [+0.018, +0.064] (n=500)
- fidelity+, GradExplainer, top10/cei_mp2 - top10/graphcare: -0.043 [-0.059, -0.027] (n=500)
- fidelity+, GradExplainer, top10/cei_mp2 - top10/gsat: -0.066 [-0.080, -0.052] (n=500)
- fidelity+, GradExplainer, top10/cei_mp2 - top10/protgnn: +0.007 [-0.015, +0.030] (n=500)
- fidelity+, GradExplainer, top10/graphcare - top10/gsat: -0.022 [-0.035, -0.010] (n=500)
- fidelity+, GradExplainer, top10/graphcare - top10/protgnn: +0.050 [+0.026, +0.074] (n=500)
- fidelity+, GradExplainer, top10/gsat - top10/protgnn: +0.072 [+0.051, +0.093] (n=500)
- fidelity+, IntegratedGradExplainer, top10/cei_mp2 - top10/graphcare: -0.062 [-0.074, -0.050] (n=500)
- fidelity+, IntegratedGradExplainer, top10/cei_mp2 - top10/gsat: -0.062 [-0.074, -0.049] (n=500)
- fidelity+, IntegratedGradExplainer, top10/cei_mp2 - top10/protgnn: -0.065 [-0.083, -0.048] (n=500)
- fidelity+, IntegratedGradExplainer, top10/graphcare - top10/gsat: +0.000 [-0.012, +0.012] (n=500)
- fidelity+, IntegratedGradExplainer, top10/graphcare - top10/protgnn: -0.003 [-0.022, +0.015] (n=500)
- fidelity+, IntegratedGradExplainer, top10/gsat - top10/protgnn: -0.004 [-0.021, +0.013] (n=500)
- fidelity+, builtin, top10/cei_mp2 - top10/graphcare: +0.451 [+0.431, +0.470] (n=500)
- fidelity+, builtin, top10/cei_mp2 - top10/gsat: -0.026 [-0.039, -0.012] (n=500)
- fidelity+, builtin, top10/cei_mp2 - top10/protgnn: +0.486 [+0.465, +0.506] (n=500)
- fidelity+, builtin, top10/graphcare - top10/gsat: -0.477 [-0.497, -0.454] (n=500)
- fidelity+, builtin, top10/graphcare - top10/protgnn: +0.034 [+0.027, +0.042] (n=500)
- fidelity+, builtin, top10/gsat - top10/protgnn: +0.511 [+0.489, +0.533] (n=500)
- fidelity-, GNNExplainer, top10/cei_mp2 - top10/graphcare: -0.085 [-0.111, -0.057] (n=500)
- fidelity-, GNNExplainer, top10/cei_mp2 - top10/gsat: +0.042 [+0.023, +0.061] (n=500)
- fidelity-, GNNExplainer, top10/cei_mp2 - top10/protgnn: -0.312 [-0.334, -0.289] (n=500)
- fidelity-, GNNExplainer, top10/graphcare - top10/gsat: +0.127 [+0.094, +0.157] (n=500)
- fidelity-, GNNExplainer, top10/graphcare - top10/protgnn: -0.227 [-0.256, -0.198] (n=500)
- fidelity-, GNNExplainer, top10/gsat - top10/protgnn: -0.354 [-0.376, -0.330] (n=500)
- fidelity-, GradExplainer, top10/cei_mp2 - top10/graphcare: +0.110 [+0.093, +0.128] (n=500)
- fidelity-, GradExplainer, top10/cei_mp2 - top10/gsat: +0.100 [+0.086, +0.114] (n=500)
- fidelity-, GradExplainer, top10/cei_mp2 - top10/protgnn: -0.377 [-0.399, -0.352] (n=500)
- fidelity-, GradExplainer, top10/graphcare - top10/gsat: -0.010 [-0.026, +0.006] (n=500)
- fidelity-, GradExplainer, top10/graphcare - top10/protgnn: -0.487 [-0.510, -0.463] (n=500)
- fidelity-, GradExplainer, top10/gsat - top10/protgnn: -0.477 [-0.499, -0.452] (n=500)
- fidelity-, IntegratedGradExplainer, top10/cei_mp2 - top10/graphcare: +0.128 [+0.111, +0.143] (n=500)
- fidelity-, IntegratedGradExplainer, top10/cei_mp2 - top10/gsat: +0.087 [+0.072, +0.101] (n=500)
- fidelity-, IntegratedGradExplainer, top10/cei_mp2 - top10/protgnn: -0.398 [-0.421, -0.373] (n=500)
- fidelity-, IntegratedGradExplainer, top10/graphcare - top10/gsat: -0.041 [-0.056, -0.025] (n=500)
- fidelity-, IntegratedGradExplainer, top10/graphcare - top10/protgnn: -0.526 [-0.547, -0.502] (n=500)
- fidelity-, IntegratedGradExplainer, top10/gsat - top10/protgnn: -0.485 [-0.510, -0.459] (n=500)
- fidelity-, builtin, top10/cei_mp2 - top10/graphcare: -0.423 [-0.443, -0.402] (n=500)
- fidelity-, builtin, top10/cei_mp2 - top10/gsat: +0.047 [+0.032, +0.060] (n=500)
- fidelity-, builtin, top10/cei_mp2 - top10/protgnn: -0.646 [-0.666, -0.625] (n=500)
- fidelity-, builtin, top10/graphcare - top10/gsat: +0.469 [+0.447, +0.491] (n=500)
- fidelity-, builtin, top10/graphcare - top10/protgnn: -0.223 [-0.239, -0.206] (n=500)
- fidelity-, builtin, top10/gsat - top10/protgnn: -0.692 [-0.716, -0.668] (n=500)
- deletion vs random, GNNExplainer, top10/cei_mp2 - top10/graphcare: +0.060 [+0.048, +0.073] (n=500)
- deletion vs random, GNNExplainer, top10/cei_mp2 - top10/gsat: -0.067 [-0.077, -0.057] (n=500)
- deletion vs random, GNNExplainer, top10/cei_mp2 - top10/protgnn: +0.017 [+0.008, +0.028] (n=500)
- deletion vs random, GNNExplainer, top10/graphcare - top10/gsat: -0.128 [-0.142, -0.112] (n=500)
- deletion vs random, GNNExplainer, top10/graphcare - top10/protgnn: -0.043 [-0.055, -0.031] (n=500)
- deletion vs random, GNNExplainer, top10/gsat - top10/protgnn: +0.085 [+0.073, +0.097] (n=500)
- deletion vs random, GradExplainer, top10/cei_mp2 - top10/graphcare: -0.066 [-0.073, -0.059] (n=500)
- deletion vs random, GradExplainer, top10/cei_mp2 - top10/gsat: -0.055 [-0.063, -0.048] (n=500)
- deletion vs random, GradExplainer, top10/cei_mp2 - top10/protgnn: +0.012 [+0.001, +0.023] (n=500)
- deletion vs random, GradExplainer, top10/graphcare - top10/gsat: +0.011 [+0.004, +0.017] (n=500)
- deletion vs random, GradExplainer, top10/graphcare - top10/protgnn: +0.078 [+0.067, +0.090] (n=500)
- deletion vs random, GradExplainer, top10/gsat - top10/protgnn: +0.067 [+0.057, +0.078] (n=500)
- deletion vs random, IntegratedGradExplainer, top10/cei_mp2 - top10/graphcare: -0.071 [-0.078, -0.064] (n=500)
- deletion vs random, IntegratedGradExplainer, top10/cei_mp2 - top10/gsat: -0.057 [-0.064, -0.049] (n=500)
- deletion vs random, IntegratedGradExplainer, top10/cei_mp2 - top10/protgnn: -0.010 [-0.021, -0.000] (n=500)
- deletion vs random, IntegratedGradExplainer, top10/graphcare - top10/gsat: +0.015 [+0.008, +0.021] (n=500)
- deletion vs random, IntegratedGradExplainer, top10/graphcare - top10/protgnn: +0.061 [+0.050, +0.071] (n=500)
- deletion vs random, IntegratedGradExplainer, top10/gsat - top10/protgnn: +0.046 [+0.036, +0.057] (n=500)
- deletion vs random, builtin, top10/cei_mp2 - top10/graphcare: +0.286 [+0.274, +0.298] (n=500)
- deletion vs random, builtin, top10/cei_mp2 - top10/gsat: -0.034 [-0.041, -0.025] (n=500)
- deletion vs random, builtin, top10/cei_mp2 - top10/protgnn: +0.278 [+0.267, +0.289] (n=500)
- deletion vs random, builtin, top10/graphcare - top10/gsat: -0.320 [-0.334, -0.306] (n=500)
- deletion vs random, builtin, top10/graphcare - top10/protgnn: -0.008 [-0.013, -0.002] (n=500)
- deletion vs random, builtin, top10/gsat - top10/protgnn: +0.312 [+0.299, +0.324] (n=500)
- insertion vs random, GNNExplainer, top10/cei_mp2 - top10/graphcare: -0.037 [-0.049, -0.025] (n=500)
- insertion vs random, GNNExplainer, top10/cei_mp2 - top10/gsat: +0.053 [+0.044, +0.061] (n=500)
- insertion vs random, GNNExplainer, top10/cei_mp2 - top10/protgnn: -0.069 [-0.078, -0.059] (n=500)
- insertion vs random, GNNExplainer, top10/graphcare - top10/gsat: +0.090 [+0.076, +0.103] (n=500)
- insertion vs random, GNNExplainer, top10/graphcare - top10/protgnn: -0.032 [-0.044, -0.019] (n=500)
- insertion vs random, GNNExplainer, top10/gsat - top10/protgnn: -0.121 [-0.131, -0.112] (n=500)
- insertion vs random, GradExplainer, top10/cei_mp2 - top10/graphcare: +0.060 [+0.053, +0.068] (n=500)
- insertion vs random, GradExplainer, top10/cei_mp2 - top10/gsat: +0.055 [+0.048, +0.063] (n=500)
- insertion vs random, GradExplainer, top10/cei_mp2 - top10/protgnn: -0.096 [-0.106, -0.085] (n=500)
- insertion vs random, GradExplainer, top10/graphcare - top10/gsat: -0.005 [-0.013, +0.003] (n=500)
- insertion vs random, GradExplainer, top10/graphcare - top10/protgnn: -0.156 [-0.166, -0.146] (n=500)
- insertion vs random, GradExplainer, top10/gsat - top10/protgnn: -0.151 [-0.161, -0.140] (n=500)
- insertion vs random, IntegratedGradExplainer, top10/cei_mp2 - top10/graphcare: +0.065 [+0.058, +0.073] (n=500)
- insertion vs random, IntegratedGradExplainer, top10/cei_mp2 - top10/gsat: +0.051 [+0.043, +0.058] (n=500)
- insertion vs random, IntegratedGradExplainer, top10/cei_mp2 - top10/protgnn: -0.102 [-0.113, -0.091] (n=500)
- insertion vs random, IntegratedGradExplainer, top10/graphcare - top10/gsat: -0.014 [-0.022, -0.007] (n=500)
- insertion vs random, IntegratedGradExplainer, top10/graphcare - top10/protgnn: -0.167 [-0.177, -0.157] (n=500)
- insertion vs random, IntegratedGradExplainer, top10/gsat - top10/protgnn: -0.153 [-0.164, -0.142] (n=500)
- insertion vs random, builtin, top10/cei_mp2 - top10/graphcare: -0.235 [-0.245, -0.225] (n=500)
- insertion vs random, builtin, top10/cei_mp2 - top10/gsat: +0.031 [+0.024, +0.039] (n=500)
- insertion vs random, builtin, top10/cei_mp2 - top10/protgnn: -0.210 [-0.218, -0.202] (n=500)
- insertion vs random, builtin, top10/graphcare - top10/gsat: +0.266 [+0.254, +0.278] (n=500)
- insertion vs random, builtin, top10/graphcare - top10/protgnn: +0.025 [+0.018, +0.031] (n=500)
- insertion vs random, builtin, top10/gsat - top10/protgnn: -0.241 [-0.251, -0.231] (n=500)
- sparsity, GNNExplainer, top10/cei_mp2 - top10/graphcare: +0.001 [+0.000, +0.002] (n=500)
- sparsity, GNNExplainer, top10/cei_mp2 - top10/gsat: -0.010 [-0.010, -0.009] (n=500)
- sparsity, GNNExplainer, top10/cei_mp2 - top10/protgnn: -0.001 [-0.001, -0.000] (n=500)
- sparsity, GNNExplainer, top10/graphcare - top10/gsat: -0.011 [-0.011, -0.010] (n=500)
- sparsity, GNNExplainer, top10/graphcare - top10/protgnn: -0.002 [-0.002, -0.001] (n=500)
- sparsity, GNNExplainer, top10/gsat - top10/protgnn: +0.009 [+0.008, +0.010] (n=500)
- sparsity, GradExplainer, top10/cei_mp2 - top10/graphcare: -0.262 [-0.269, -0.254] (n=500)
- sparsity, GradExplainer, top10/cei_mp2 - top10/gsat: -0.072 [-0.080, -0.064] (n=500)
- sparsity, GradExplainer, top10/cei_mp2 - top10/protgnn: -0.210 [-0.218, -0.203] (n=500)
- sparsity, GradExplainer, top10/graphcare - top10/gsat: +0.190 [+0.180, +0.199] (n=500)
- sparsity, GradExplainer, top10/graphcare - top10/protgnn: +0.051 [+0.044, +0.058] (n=500)
- sparsity, GradExplainer, top10/gsat - top10/protgnn: -0.138 [-0.148, -0.128] (n=500)
- sparsity, IntegratedGradExplainer, top10/cei_mp2 - top10/graphcare: -0.162 [-0.171, -0.152] (n=500)
- sparsity, IntegratedGradExplainer, top10/cei_mp2 - top10/gsat: -0.065 [-0.073, -0.058] (n=500)
- sparsity, IntegratedGradExplainer, top10/cei_mp2 - top10/protgnn: -0.223 [-0.230, -0.216] (n=500)
- sparsity, IntegratedGradExplainer, top10/graphcare - top10/gsat: +0.097 [+0.086, +0.107] (n=500)
- sparsity, IntegratedGradExplainer, top10/graphcare - top10/protgnn: -0.061 [-0.072, -0.051] (n=500)
- sparsity, IntegratedGradExplainer, top10/gsat - top10/protgnn: -0.158 [-0.168, -0.148] (n=500)
- sparsity, builtin, top10/cei_mp2 - top10/graphcare: +0.492 [+0.473, +0.511] (n=500)
- sparsity, builtin, top10/cei_mp2 - top10/gsat: +0.605 [+0.590, +0.620] (n=500)
- sparsity, builtin, top10/cei_mp2 - top10/protgnn: -0.242 [-0.257, -0.227] (n=500)
- sparsity, builtin, top10/graphcare - top10/gsat: +0.113 [+0.100, +0.128] (n=500)
- sparsity, builtin, top10/graphcare - top10/protgnn: -0.734 [-0.747, -0.720] (n=500)
- sparsity, builtin, top10/gsat - top10/protgnn: -0.847 [-0.850, -0.844] (n=500)

