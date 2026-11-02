# Fairness checks on the explanation results

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## 1. How many nodes the main fidelity cut removes

Main cut = top 20% of the visit's eligible evidence nodes, capped at the explanation's own support. `k` is the number actually deleted; `eligible` the visit's eligible nodes; `capped` the share of visits where the support, not the 20%, set k.

**top30/cei_mp2**

- builtin: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.1, eligible = 47.7, capped by support in 0% of patients (n=500)

## 2. Matched evidence budgets

Every source scored at the same fixed k. Cells: probability drop with the top-k deleted (fidelity+, higher = more faithful) / with only the top-k kept (fidelity-, lower = more faithful); then the difference to random deletion of the same size on the same visits (+ column: higher is better; - column: lower is better). A source with fewer than k eligible nodes of nonzero importance is `beyond support` at k and is not padded.

### top30/cei_mp2

- builtin, k=1: fidelity+ +0.407 [+0.385, +0.430] (n=500), vs random +0.394 [+0.372, +0.415] (n=500); fidelity- +0.041 [+0.025, +0.059] (n=500), vs random -0.555 [-0.579, -0.529] (n=500)
- builtin, k=2: fidelity+ +0.512 [+0.489, +0.534] (n=500), vs random +0.484 [+0.462, +0.505] (n=500); fidelity- +0.004 [-0.008, +0.019] (n=500), vs random -0.560 [-0.581, -0.537] (n=500)
- builtin, k=3: fidelity+ +0.553 [+0.530, +0.575] (n=500), vs random +0.513 [+0.491, +0.533] (n=500); fidelity- -0.003 [-0.014, +0.009] (n=500), vs random -0.541 [-0.562, -0.519] (n=500)
- builtin, k=5: fidelity+ +0.580 [+0.557, +0.602] (n=500), vs random +0.514 [+0.492, +0.534] (n=500); fidelity- -0.009 [-0.019, +0.002] (n=500), vs random -0.497 [-0.516, -0.477] (n=500)
- builtin, k=10: fidelity+ +0.606 [+0.584, +0.627] (n=492), vs random +0.470 [+0.451, +0.490] (n=492); fidelity- -0.017 [-0.026, -0.008] (n=492), vs random -0.407 [-0.427, -0.388] (n=492)
- GNNExplainer, k=1: fidelity+ +0.279 [+0.255, +0.302] (n=500), vs random +0.266 [+0.242, +0.288] (n=500); fidelity- +0.222 [+0.199, +0.246] (n=500), vs random -0.374 [-0.403, -0.346] (n=500)
- GNNExplainer, k=2: fidelity+ +0.372 [+0.349, +0.397] (n=500), vs random +0.345 [+0.322, +0.368] (n=500); fidelity- +0.109 [+0.090, +0.130] (n=500), vs random -0.455 [-0.480, -0.429] (n=500)
- GNNExplainer, k=3: fidelity+ +0.420 [+0.397, +0.444] (n=500), vs random +0.381 [+0.358, +0.403] (n=500); fidelity- +0.059 [+0.042, +0.078] (n=500), vs random -0.479 [-0.503, -0.453] (n=500)
- GNNExplainer, k=5: fidelity+ +0.473 [+0.451, +0.495] (n=500), vs random +0.406 [+0.385, +0.426] (n=500); fidelity- +0.016 [+0.002, +0.031] (n=500), vs random -0.472 [-0.494, -0.449] (n=500)
- GNNExplainer, k=10: fidelity+ +0.519 [+0.500, +0.540] (n=492), vs random +0.384 [+0.365, +0.404] (n=492); fidelity- -0.024 [-0.036, -0.013] (n=492), vs random -0.415 [-0.435, -0.393] (n=492)
- GradExplainer, k=1: fidelity+ +0.311 [+0.288, +0.334] (n=500), vs random +0.297 [+0.274, +0.320] (n=500); fidelity- +0.163 [+0.144, +0.184] (n=500), vs random -0.433 [-0.458, -0.408] (n=500)
- GradExplainer, k=2: fidelity+ +0.442 [+0.418, +0.465] (n=500), vs random +0.414 [+0.392, +0.436] (n=500); fidelity- +0.067 [+0.053, +0.082] (n=500), vs random -0.498 [-0.521, -0.473] (n=500)
- GradExplainer, k=3: fidelity+ +0.498 [+0.474, +0.520] (n=500), vs random +0.458 [+0.436, +0.480] (n=500); fidelity- +0.052 [+0.040, +0.065] (n=500), vs random -0.486 [-0.508, -0.463] (n=500)
- GradExplainer, k=5: fidelity+ +0.544 [+0.521, +0.567] (n=500), vs random +0.478 [+0.455, +0.499] (n=500); fidelity- +0.048 [+0.037, +0.060] (n=500), vs random -0.440 [-0.461, -0.417] (n=500)
- GradExplainer, k=10: fidelity+ +0.573 [+0.550, +0.595] (n=492), vs random +0.438 [+0.417, +0.458] (n=492); fidelity- +0.039 [+0.029, +0.048] (n=492), vs random -0.352 [-0.372, -0.330] (n=492)
- IntegratedGradExplainer, k=1: fidelity+ +0.377 [+0.353, +0.400] (n=500), vs random +0.363 [+0.340, +0.386] (n=500); fidelity- +0.082 [+0.066, +0.101] (n=500), vs random -0.514 [-0.540, -0.488] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.479 [+0.456, +0.501] (n=500), vs random +0.451 [+0.429, +0.472] (n=500); fidelity- +0.038 [+0.025, +0.052] (n=500), vs random -0.527 [-0.549, -0.502] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.530 [+0.506, +0.552] (n=500), vs random +0.490 [+0.468, +0.511] (n=500); fidelity- +0.029 [+0.018, +0.041] (n=500), vs random -0.509 [-0.530, -0.486] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.560 [+0.537, +0.582] (n=500), vs random +0.494 [+0.472, +0.514] (n=500); fidelity- +0.031 [+0.020, +0.043] (n=500), vs random -0.457 [-0.478, -0.434] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.575 [+0.553, +0.597] (n=492), vs random +0.440 [+0.419, +0.460] (n=492); fidelity- +0.034 [+0.024, +0.044] (n=492), vs random -0.357 [-0.378, -0.336] (n=492)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top30/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.589 [+0.567, +0.611] (n=500)
- GNNExplainer: +0.474 [+0.451, +0.496] (n=500)
- GradExplainer: +0.555 [+0.532, +0.577] (n=500)
- IntegratedGradExplainer: +0.564 [+0.542, +0.587] (n=500)
- GNNExplainer - GradExplainer: -0.081 [-0.099, -0.063] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.090 [-0.109, -0.071] (n=500)
- GNNExplainer - builtin: -0.115 [-0.134, -0.097] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.009 [-0.014, -0.004] (n=500)
- GradExplainer - builtin: -0.034 [-0.040, -0.029] (n=500)
- IntegratedGradExplainer - builtin: -0.025 [-0.030, -0.020] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.005 [-0.015, +0.005] (n=500)
- GNNExplainer: +0.026 [+0.008, +0.045] (n=500)
- GradExplainer: +0.051 [+0.040, +0.062] (n=500)
- IntegratedGradExplainer: +0.040 [+0.030, +0.051] (n=500)
- GNNExplainer - GradExplainer: -0.025 [-0.044, -0.006] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.014 [-0.033, +0.005] (n=500)
- GNNExplainer - builtin: +0.031 [+0.013, +0.049] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.011 [+0.006, +0.016] (n=500)
- GradExplainer - builtin: +0.056 [+0.048, +0.065] (n=500)
- IntegratedGradExplainer - builtin: +0.045 [+0.037, +0.054] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.320 [+0.307, +0.331] (n=500)
- GNNExplainer: +0.240 [+0.229, +0.250] (n=500)
- GradExplainer: +0.287 [+0.274, +0.299] (n=500)
- IntegratedGradExplainer: +0.290 [+0.277, +0.303] (n=500)
- GNNExplainer - GradExplainer: -0.047 [-0.057, -0.037] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.051 [-0.061, -0.040] (n=500)
- GNNExplainer - builtin: -0.080 [-0.089, -0.069] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.003 [-0.006, -0.001] (n=500)
- GradExplainer - builtin: -0.033 [-0.037, -0.028] (n=500)
- IntegratedGradExplainer - builtin: -0.029 [-0.035, -0.024] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.247 [-0.256, -0.238] (n=500)
- GNNExplainer: -0.228 [-0.238, -0.218] (n=500)
- GradExplainer: -0.212 [-0.222, -0.201] (n=500)
- IntegratedGradExplainer: -0.215 [-0.225, -0.204] (n=500)
- GNNExplainer - GradExplainer: -0.016 [-0.024, -0.009] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.013 [-0.021, -0.006] (n=500)
- GNNExplainer - builtin: +0.019 [+0.011, +0.026] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.003 [+0.001, +0.005] (n=500)
- GradExplainer - builtin: +0.035 [+0.030, +0.040] (n=500)
- IntegratedGradExplainer - builtin: +0.032 [+0.027, +0.037] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.689 [+0.675, +0.702] (n=500)
- GNNExplainer: +0.101 [+0.100, +0.102] (n=500)
- GradExplainer: +0.404 [+0.396, +0.413] (n=500)
- IntegratedGradExplainer: +0.362 [+0.356, +0.370] (n=500)
- GNNExplainer - GradExplainer: -0.303 [-0.311, -0.294] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.261 [-0.268, -0.254] (n=500)
- GNNExplainer - builtin: -0.587 [-0.601, -0.574] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.041 [+0.036, +0.047] (n=500)
- GradExplainer - builtin: -0.285 [-0.296, -0.273] (n=500)
- IntegratedGradExplainer - builtin: -0.326 [-0.337, -0.316] (n=500)

