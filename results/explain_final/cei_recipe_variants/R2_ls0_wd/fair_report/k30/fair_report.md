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

- builtin, k=1: fidelity+ +0.410 [+0.387, +0.433] (n=500), vs random +0.397 [+0.374, +0.418] (n=500); fidelity- +0.038 [+0.021, +0.055] (n=500), vs random -0.558 [-0.582, -0.532] (n=500)
- builtin, k=2: fidelity+ +0.518 [+0.494, +0.540] (n=500), vs random +0.490 [+0.468, +0.511] (n=500); fidelity- -0.005 [-0.017, +0.010] (n=500), vs random -0.568 [-0.590, -0.545] (n=500)
- builtin, k=3: fidelity+ +0.560 [+0.536, +0.582] (n=500), vs random +0.520 [+0.498, +0.541] (n=500); fidelity- -0.016 [-0.028, -0.003] (n=500), vs random -0.553 [-0.574, -0.531] (n=500)
- builtin, k=5: fidelity+ +0.585 [+0.562, +0.607] (n=500), vs random +0.519 [+0.498, +0.540] (n=500); fidelity- -0.023 [-0.035, -0.012] (n=500), vs random -0.510 [-0.529, -0.489] (n=500)
- builtin, k=10: fidelity+ +0.609 [+0.587, +0.631] (n=492), vs random +0.474 [+0.453, +0.493] (n=492); fidelity- -0.024 [-0.033, -0.016] (n=492), vs random -0.414 [-0.434, -0.395] (n=492)
- GNNExplainer, k=1: fidelity+ +0.290 [+0.266, +0.313] (n=500), vs random +0.277 [+0.253, +0.300] (n=500); fidelity- +0.202 [+0.180, +0.225] (n=500), vs random -0.393 [-0.421, -0.364] (n=500)
- GNNExplainer, k=2: fidelity+ +0.382 [+0.358, +0.406] (n=500), vs random +0.354 [+0.331, +0.377] (n=500); fidelity- +0.096 [+0.077, +0.119] (n=500), vs random -0.467 [-0.493, -0.440] (n=500)
- GNNExplainer, k=3: fidelity+ +0.434 [+0.409, +0.457] (n=500), vs random +0.394 [+0.371, +0.416] (n=500); fidelity- +0.051 [+0.034, +0.071] (n=500), vs random -0.486 [-0.511, -0.460] (n=500)
- GNNExplainer, k=5: fidelity+ +0.481 [+0.459, +0.503] (n=500), vs random +0.415 [+0.394, +0.435] (n=500); fidelity- +0.007 [-0.007, +0.022] (n=500), vs random -0.480 [-0.502, -0.457] (n=500)
- GNNExplainer, k=10: fidelity+ +0.526 [+0.506, +0.548] (n=492), vs random +0.391 [+0.371, +0.411] (n=492); fidelity- -0.025 [-0.036, -0.014] (n=492), vs random -0.415 [-0.435, -0.394] (n=492)
- GradExplainer, k=1: fidelity+ +0.312 [+0.288, +0.335] (n=500), vs random +0.299 [+0.275, +0.321] (n=500); fidelity- +0.161 [+0.142, +0.182] (n=500), vs random -0.434 [-0.459, -0.408] (n=500)
- GradExplainer, k=2: fidelity+ +0.449 [+0.425, +0.471] (n=500), vs random +0.421 [+0.399, +0.442] (n=500); fidelity- +0.058 [+0.044, +0.073] (n=500), vs random -0.506 [-0.528, -0.480] (n=500)
- GradExplainer, k=3: fidelity+ +0.508 [+0.483, +0.530] (n=500), vs random +0.468 [+0.446, +0.489] (n=500); fidelity- +0.038 [+0.026, +0.051] (n=500), vs random -0.499 [-0.521, -0.476] (n=500)
- GradExplainer, k=5: fidelity+ +0.550 [+0.525, +0.573] (n=500), vs random +0.483 [+0.461, +0.505] (n=500); fidelity- +0.039 [+0.028, +0.050] (n=500), vs random -0.448 [-0.470, -0.426] (n=500)
- GradExplainer, k=10: fidelity+ +0.577 [+0.553, +0.599] (n=492), vs random +0.442 [+0.421, +0.463] (n=492); fidelity- +0.031 [+0.023, +0.041] (n=492), vs random -0.358 [-0.380, -0.337] (n=492)
- IntegratedGradExplainer, k=1: fidelity+ +0.381 [+0.357, +0.405] (n=500), vs random +0.368 [+0.344, +0.391] (n=500); fidelity- +0.078 [+0.062, +0.097] (n=500), vs random -0.517 [-0.542, -0.491] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.489 [+0.466, +0.511] (n=500), vs random +0.461 [+0.440, +0.482] (n=500); fidelity- +0.031 [+0.018, +0.044] (n=500), vs random -0.533 [-0.556, -0.509] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.536 [+0.512, +0.557] (n=500), vs random +0.496 [+0.473, +0.517] (n=500); fidelity- +0.021 [+0.010, +0.034] (n=500), vs random -0.516 [-0.538, -0.493] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.561 [+0.537, +0.583] (n=500), vs random +0.495 [+0.473, +0.516] (n=500); fidelity- +0.026 [+0.015, +0.038] (n=500), vs random -0.461 [-0.483, -0.439] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.575 [+0.553, +0.598] (n=492), vs random +0.441 [+0.420, +0.461] (n=492); fidelity- +0.032 [+0.023, +0.041] (n=492), vs random -0.358 [-0.379, -0.338] (n=492)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top30/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.591 [+0.569, +0.613] (n=500)
- GNNExplainer: +0.482 [+0.459, +0.504] (n=500)
- GradExplainer: +0.560 [+0.536, +0.583] (n=500)
- IntegratedGradExplainer: +0.565 [+0.542, +0.588] (n=500)
- GNNExplainer - GradExplainer: -0.079 [-0.096, -0.061] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.084 [-0.102, -0.065] (n=500)
- GNNExplainer - builtin: -0.110 [-0.127, -0.092] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.005 [-0.010, -0.000] (n=500)
- GradExplainer - builtin: -0.031 [-0.036, -0.026] (n=500)
- IntegratedGradExplainer - builtin: -0.026 [-0.031, -0.021] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.013 [-0.023, -0.003] (n=500)
- GNNExplainer: +0.019 [+0.002, +0.038] (n=500)
- GradExplainer: +0.043 [+0.032, +0.054] (n=500)
- IntegratedGradExplainer: +0.035 [+0.025, +0.046] (n=500)
- GNNExplainer - GradExplainer: -0.024 [-0.043, -0.006] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.016 [-0.035, +0.002] (n=500)
- GNNExplainer - builtin: +0.032 [+0.014, +0.050] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.008 [+0.003, +0.013] (n=500)
- GradExplainer - builtin: +0.056 [+0.048, +0.064] (n=500)
- IntegratedGradExplainer - builtin: +0.048 [+0.040, +0.056] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.322 [+0.310, +0.334] (n=500)
- GNNExplainer: +0.245 [+0.234, +0.256] (n=500)
- GradExplainer: +0.290 [+0.278, +0.302] (n=500)
- IntegratedGradExplainer: +0.292 [+0.279, +0.305] (n=500)
- GNNExplainer - GradExplainer: -0.045 [-0.055, -0.036] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.047 [-0.057, -0.037] (n=500)
- GNNExplainer - builtin: -0.077 [-0.087, -0.068] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.002 [-0.004, +0.001] (n=500)
- GradExplainer - builtin: -0.032 [-0.037, -0.028] (n=500)
- IntegratedGradExplainer - builtin: -0.030 [-0.035, -0.026] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.252 [-0.261, -0.244] (n=500)
- GNNExplainer: -0.231 [-0.241, -0.220] (n=500)
- GradExplainer: -0.215 [-0.225, -0.204] (n=500)
- IntegratedGradExplainer: -0.216 [-0.227, -0.205] (n=500)
- GNNExplainer - GradExplainer: -0.016 [-0.023, -0.008] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.014 [-0.023, -0.007] (n=500)
- GNNExplainer - builtin: +0.022 [+0.014, +0.029] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.001 [-0.001, +0.003] (n=500)
- GradExplainer - builtin: +0.037 [+0.033, +0.042] (n=500)
- IntegratedGradExplainer - builtin: +0.036 [+0.032, +0.041] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.689 [+0.675, +0.703] (n=500)
- GNNExplainer: +0.101 [+0.100, +0.102] (n=500)
- GradExplainer: +0.404 [+0.395, +0.413] (n=500)
- IntegratedGradExplainer: +0.354 [+0.348, +0.361] (n=500)
- GNNExplainer - GradExplainer: -0.303 [-0.311, -0.295] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.253 [-0.260, -0.247] (n=500)
- GNNExplainer - builtin: -0.588 [-0.602, -0.574] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.050 [+0.044, +0.055] (n=500)
- GradExplainer - builtin: -0.285 [-0.297, -0.273] (n=500)
- IntegratedGradExplainer - builtin: -0.335 [-0.345, -0.324] (n=500)

