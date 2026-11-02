# Fairness checks on the explanation results

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## 1. How many nodes the main fidelity cut removes

Main cut = top 20% of the visit's eligible evidence nodes, capped at the explanation's own support. `k` is the number actually deleted; `eligible` the visit's eligible nodes; `capped` the share of visits where the support, not the 20%, set k.

**top20/cei_mp2**

- builtin: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.2, eligible = 48.3, capped by support in 0% of patients (n=500)

## 2. Matched evidence budgets

Every source scored at the same fixed k. Cells: probability drop with the top-k deleted (fidelity+, higher = more faithful) / with only the top-k kept (fidelity-, lower = more faithful); then the difference to random deletion of the same size on the same visits (+ column: higher is better; - column: lower is better). A source with fewer than k eligible nodes of nonzero importance is `beyond support` at k and is not padded.

### top20/cei_mp2

- builtin, k=1: fidelity+ +0.370 [+0.350, +0.390] (n=500), vs random +0.358 [+0.338, +0.376] (n=500); fidelity- +0.032 [+0.016, +0.047] (n=500), vs random -0.499 [-0.520, -0.477] (n=500)
- builtin, k=2: fidelity+ +0.455 [+0.435, +0.476] (n=500), vs random +0.431 [+0.412, +0.450] (n=500); fidelity- -0.007 [-0.019, +0.005] (n=500), vs random -0.511 [-0.531, -0.489] (n=500)
- builtin, k=3: fidelity+ +0.482 [+0.463, +0.502] (n=500), vs random +0.447 [+0.428, +0.465] (n=500); fidelity- -0.008 [-0.018, +0.002] (n=500), vs random -0.487 [-0.507, -0.469] (n=500)
- builtin, k=5: fidelity+ +0.507 [+0.488, +0.527] (n=500), vs random +0.448 [+0.429, +0.466] (n=500); fidelity- -0.014 [-0.022, -0.005] (n=500), vs random -0.449 [-0.467, -0.431] (n=500)
- builtin, k=10: fidelity+ +0.529 [+0.509, +0.549] (n=494), vs random +0.410 [+0.391, +0.428] (n=494); fidelity- -0.017 [-0.023, -0.010] (n=494), vs random -0.365 [-0.382, -0.349] (n=494)
- GNNExplainer, k=1: fidelity+ +0.189 [+0.172, +0.207] (n=500), vs random +0.177 [+0.160, +0.194] (n=500); fidelity- +0.264 [+0.242, +0.287] (n=500), vs random -0.267 [-0.289, -0.245] (n=500)
- GNNExplainer, k=2: fidelity+ +0.292 [+0.272, +0.312] (n=500), vs random +0.267 [+0.247, +0.287] (n=500); fidelity- +0.134 [+0.114, +0.155] (n=500), vs random -0.369 [-0.394, -0.347] (n=500)
- GNNExplainer, k=3: fidelity+ +0.357 [+0.337, +0.378] (n=500), vs random +0.321 [+0.300, +0.341] (n=500); fidelity- +0.070 [+0.052, +0.088] (n=500), vs random -0.410 [-0.433, -0.387] (n=500)
- GNNExplainer, k=5: fidelity+ +0.431 [+0.412, +0.452] (n=500), vs random +0.372 [+0.353, +0.391] (n=500); fidelity- +0.002 [-0.012, +0.017] (n=500), vs random -0.433 [-0.454, -0.412] (n=500)
- GNNExplainer, k=10: fidelity+ +0.487 [+0.467, +0.507] (n=494), vs random +0.367 [+0.350, +0.385] (n=494); fidelity- -0.027 [-0.037, -0.017] (n=494), vs random -0.375 [-0.394, -0.357] (n=494)
- GradExplainer, k=1: fidelity+ +0.255 [+0.235, +0.276] (n=500), vs random +0.243 [+0.223, +0.263] (n=500); fidelity- +0.181 [+0.160, +0.202] (n=500), vs random -0.350 [-0.371, -0.328] (n=500)
- GradExplainer, k=2: fidelity+ +0.370 [+0.350, +0.391] (n=500), vs random +0.345 [+0.326, +0.365] (n=500); fidelity- +0.083 [+0.068, +0.098] (n=500), vs random -0.420 [-0.441, -0.399] (n=500)
- GradExplainer, k=3: fidelity+ +0.430 [+0.408, +0.450] (n=500), vs random +0.394 [+0.374, +0.413] (n=500); fidelity- +0.049 [+0.037, +0.061] (n=500), vs random -0.431 [-0.450, -0.410] (n=500)
- GradExplainer, k=5: fidelity+ +0.476 [+0.453, +0.498] (n=500), vs random +0.416 [+0.395, +0.436] (n=500); fidelity- +0.044 [+0.034, +0.054] (n=500), vs random -0.392 [-0.411, -0.372] (n=500)
- GradExplainer, k=10: fidelity+ +0.500 [+0.478, +0.521] (n=494), vs random +0.380 [+0.361, +0.400] (n=494); fidelity- +0.034 [+0.027, +0.041] (n=494), vs random -0.314 [-0.332, -0.297] (n=494)
- IntegratedGradExplainer, k=1: fidelity+ +0.333 [+0.312, +0.354] (n=500), vs random +0.321 [+0.300, +0.342] (n=500); fidelity- +0.082 [+0.066, +0.100] (n=500), vs random -0.449 [-0.471, -0.426] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.429 [+0.409, +0.449] (n=500), vs random +0.405 [+0.386, +0.424] (n=500); fidelity- +0.028 [+0.016, +0.040] (n=500), vs random -0.476 [-0.497, -0.454] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.463 [+0.442, +0.485] (n=500), vs random +0.427 [+0.408, +0.447] (n=500); fidelity- +0.025 [+0.014, +0.035] (n=500), vs random -0.455 [-0.475, -0.434] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.490 [+0.469, +0.511] (n=500), vs random +0.430 [+0.410, +0.450] (n=500); fidelity- +0.031 [+0.021, +0.041] (n=500), vs random -0.405 [-0.425, -0.385] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.505 [+0.483, +0.527] (n=494), vs random +0.386 [+0.366, +0.405] (n=494); fidelity- +0.030 [+0.023, +0.038] (n=494), vs random -0.318 [-0.336, -0.301] (n=494)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top20/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.513 [+0.495, +0.533] (n=500)
- GNNExplainer: +0.432 [+0.411, +0.453] (n=500)
- GradExplainer: +0.476 [+0.456, +0.498] (n=500)
- IntegratedGradExplainer: +0.493 [+0.473, +0.513] (n=500)
- GNNExplainer - GradExplainer: -0.045 [-0.059, -0.032] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.061 [-0.076, -0.048] (n=500)
- GNNExplainer - builtin: -0.082 [-0.096, -0.068] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.016 [-0.023, -0.011] (n=500)
- GradExplainer - builtin: -0.037 [-0.044, -0.030] (n=500)
- IntegratedGradExplainer - builtin: -0.021 [-0.026, -0.015] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.010 [-0.017, -0.002] (n=500)
- GNNExplainer: +0.022 [+0.007, +0.039] (n=500)
- GradExplainer: +0.052 [+0.044, +0.062] (n=500)
- IntegratedGradExplainer: +0.038 [+0.030, +0.047] (n=500)
- GNNExplainer - GradExplainer: -0.030 [-0.046, -0.015] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.016 [-0.032, +0.001] (n=500)
- GNNExplainer - builtin: +0.032 [+0.017, +0.048] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.014 [+0.008, +0.020] (n=500)
- GradExplainer - builtin: +0.062 [+0.054, +0.070] (n=500)
- IntegratedGradExplainer - builtin: +0.048 [+0.040, +0.056] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.275 [+0.264, +0.285] (n=500)
- GNNExplainer: +0.226 [+0.216, +0.236] (n=500)
- GradExplainer: +0.250 [+0.240, +0.261] (n=500)
- IntegratedGradExplainer: +0.256 [+0.245, +0.267] (n=500)
- GNNExplainer - GradExplainer: -0.024 [-0.031, -0.018] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.030 [-0.037, -0.023] (n=500)
- GNNExplainer - builtin: -0.049 [-0.056, -0.042] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.006 [-0.007, -0.004] (n=500)
- GradExplainer - builtin: -0.024 [-0.028, -0.021] (n=500)
- IntegratedGradExplainer - builtin: -0.019 [-0.022, -0.015] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.218 [-0.227, -0.210] (n=500)
- GNNExplainer: -0.204 [-0.213, -0.195] (n=500)
- GradExplainer: -0.188 [-0.197, -0.179] (n=500)
- IntegratedGradExplainer: -0.190 [-0.200, -0.181] (n=500)
- GNNExplainer - GradExplainer: -0.016 [-0.023, -0.010] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.014 [-0.021, -0.008] (n=500)
- GNNExplainer - builtin: +0.014 [+0.008, +0.021] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.002 [+0.000, +0.004] (n=500)
- GradExplainer - builtin: +0.030 [+0.027, +0.034] (n=500)
- IntegratedGradExplainer - builtin: +0.028 [+0.025, +0.033] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.736 [+0.724, +0.747] (n=500)
- GNNExplainer: +0.101 [+0.101, +0.102] (n=500)
- GradExplainer: +0.379 [+0.370, +0.387] (n=500)
- IntegratedGradExplainer: +0.319 [+0.313, +0.325] (n=500)
- GNNExplainer - GradExplainer: -0.277 [-0.286, -0.269] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.218 [-0.224, -0.212] (n=500)
- GNNExplainer - builtin: -0.635 [-0.646, -0.623] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.060 [+0.055, +0.064] (n=500)
- GradExplainer - builtin: -0.358 [-0.369, -0.346] (n=500)
- IntegratedGradExplainer - builtin: -0.417 [-0.427, -0.407] (n=500)

