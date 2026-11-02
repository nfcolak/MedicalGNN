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

- builtin, k=1: fidelity+ +0.397 [+0.374, +0.419] (n=500), vs random +0.383 [+0.362, +0.406] (n=500); fidelity- +0.048 [+0.031, +0.066] (n=500), vs random -0.553 [-0.578, -0.527] (n=500)
- builtin, k=2: fidelity+ +0.499 [+0.478, +0.521] (n=500), vs random +0.473 [+0.452, +0.494] (n=500); fidelity- +0.019 [+0.005, +0.034] (n=500), vs random -0.551 [-0.574, -0.528] (n=500)
- builtin, k=3: fidelity+ +0.540 [+0.518, +0.561] (n=500), vs random +0.501 [+0.480, +0.521] (n=500); fidelity- +0.006 [-0.006, +0.018] (n=500), vs random -0.536 [-0.557, -0.514] (n=500)
- builtin, k=5: fidelity+ +0.571 [+0.550, +0.592] (n=500), vs random +0.506 [+0.486, +0.525] (n=500); fidelity- +0.000 [-0.010, +0.011] (n=500), vs random -0.491 [-0.511, -0.471] (n=500)
- builtin, k=10: fidelity+ +0.603 [+0.581, +0.625] (n=494), vs random +0.473 [+0.454, +0.493] (n=494); fidelity- -0.007 [-0.015, +0.002] (n=494), vs random -0.398 [-0.417, -0.379] (n=494)
- GNNExplainer, k=1: fidelity+ +0.231 [+0.210, +0.254] (n=500), vs random +0.218 [+0.197, +0.240] (n=500); fidelity- +0.259 [+0.233, +0.284] (n=500), vs random -0.343 [-0.371, -0.317] (n=500)
- GNNExplainer, k=2: fidelity+ +0.327 [+0.304, +0.351] (n=500), vs random +0.301 [+0.278, +0.324] (n=500); fidelity- +0.149 [+0.126, +0.172] (n=500), vs random -0.421 [-0.448, -0.395] (n=500)
- GNNExplainer, k=3: fidelity+ +0.385 [+0.361, +0.409] (n=500), vs random +0.346 [+0.323, +0.369] (n=500); fidelity- +0.093 [+0.073, +0.113] (n=500), vs random -0.449 [-0.474, -0.424] (n=500)
- GNNExplainer, k=5: fidelity+ +0.446 [+0.423, +0.469] (n=500), vs random +0.381 [+0.358, +0.404] (n=500); fidelity- +0.044 [+0.027, +0.061] (n=500), vs random -0.447 [-0.471, -0.423] (n=500)
- GNNExplainer, k=10: fidelity+ +0.514 [+0.493, +0.536] (n=494), vs random +0.384 [+0.364, +0.405] (n=494); fidelity- +0.001 [-0.012, +0.014] (n=494), vs random -0.391 [-0.412, -0.370] (n=494)
- GradExplainer, k=1: fidelity+ +0.289 [+0.267, +0.312] (n=500), vs random +0.276 [+0.254, +0.299] (n=500); fidelity- +0.196 [+0.175, +0.218] (n=500), vs random -0.405 [-0.431, -0.379] (n=500)
- GradExplainer, k=2: fidelity+ +0.419 [+0.395, +0.442] (n=500), vs random +0.392 [+0.370, +0.415] (n=500); fidelity- +0.088 [+0.072, +0.104] (n=500), vs random -0.482 [-0.507, -0.457] (n=500)
- GradExplainer, k=3: fidelity+ +0.477 [+0.453, +0.501] (n=500), vs random +0.438 [+0.416, +0.460] (n=500); fidelity- +0.065 [+0.052, +0.078] (n=500), vs random -0.477 [-0.500, -0.453] (n=500)
- GradExplainer, k=5: fidelity+ +0.526 [+0.503, +0.549] (n=500), vs random +0.461 [+0.440, +0.481] (n=500); fidelity- +0.058 [+0.046, +0.070] (n=500), vs random -0.434 [-0.456, -0.411] (n=500)
- GradExplainer, k=10: fidelity+ +0.560 [+0.537, +0.584] (n=494), vs random +0.430 [+0.409, +0.452] (n=494); fidelity- +0.049 [+0.040, +0.060] (n=494), vs random -0.342 [-0.363, -0.323] (n=494)
- IntegratedGradExplainer, k=1: fidelity+ +0.367 [+0.342, +0.391] (n=500), vs random +0.353 [+0.329, +0.378] (n=500); fidelity- +0.098 [+0.079, +0.117] (n=500), vs random -0.504 [-0.529, -0.477] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.469 [+0.446, +0.492] (n=500), vs random +0.442 [+0.421, +0.464] (n=500); fidelity- +0.050 [+0.036, +0.065] (n=500), vs random -0.520 [-0.544, -0.496] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.507 [+0.485, +0.530] (n=500), vs random +0.469 [+0.447, +0.490] (n=500); fidelity- +0.040 [+0.027, +0.054] (n=500), vs random -0.501 [-0.525, -0.478] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.547 [+0.524, +0.569] (n=500), vs random +0.482 [+0.460, +0.502] (n=500); fidelity- +0.043 [+0.030, +0.056] (n=500), vs random -0.448 [-0.470, -0.427] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.568 [+0.545, +0.591] (n=494), vs random +0.438 [+0.418, +0.459] (n=494); fidelity- +0.045 [+0.034, +0.056] (n=494), vs random -0.347 [-0.367, -0.327] (n=494)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top20/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.578 [+0.558, +0.600] (n=500)
- GNNExplainer: +0.458 [+0.435, +0.482] (n=500)
- GradExplainer: +0.537 [+0.515, +0.560] (n=500)
- IntegratedGradExplainer: +0.554 [+0.532, +0.576] (n=500)
- GNNExplainer - GradExplainer: -0.079 [-0.096, -0.061] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.096 [-0.113, -0.078] (n=500)
- GNNExplainer - builtin: -0.120 [-0.137, -0.102] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.017 [-0.023, -0.010] (n=500)
- GradExplainer - builtin: -0.041 [-0.049, -0.033] (n=500)
- IntegratedGradExplainer - builtin: -0.024 [-0.032, -0.017] (n=500)

**fidelity-** (lower = more faithful)

- builtin: +0.002 [-0.007, +0.013] (n=500)
- GNNExplainer: +0.052 [+0.033, +0.071] (n=500)
- GradExplainer: +0.062 [+0.052, +0.073] (n=500)
- IntegratedGradExplainer: +0.051 [+0.039, +0.063] (n=500)
- GNNExplainer - GradExplainer: -0.010 [-0.029, +0.008] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.001 [-0.018, +0.019] (n=500)
- GNNExplainer - builtin: +0.049 [+0.031, +0.068] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.011 [+0.004, +0.017] (n=500)
- GradExplainer - builtin: +0.059 [+0.050, +0.069] (n=500)
- IntegratedGradExplainer - builtin: +0.049 [+0.039, +0.059] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.322 [+0.310, +0.333] (n=500)
- GNNExplainer: +0.238 [+0.227, +0.250] (n=500)
- GradExplainer: +0.282 [+0.270, +0.294] (n=500)
- IntegratedGradExplainer: +0.290 [+0.278, +0.303] (n=500)
- GNNExplainer - GradExplainer: -0.044 [-0.053, -0.034] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.052 [-0.062, -0.042] (n=500)
- GNNExplainer - builtin: -0.084 [-0.093, -0.074] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.008 [-0.011, -0.005] (n=500)
- GradExplainer - builtin: -0.040 [-0.045, -0.034] (n=500)
- IntegratedGradExplainer - builtin: -0.032 [-0.037, -0.026] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.238 [-0.248, -0.229] (n=500)
- GNNExplainer: -0.214 [-0.224, -0.203] (n=500)
- GradExplainer: -0.206 [-0.217, -0.196] (n=500)
- IntegratedGradExplainer: -0.210 [-0.221, -0.199] (n=500)
- GNNExplainer - GradExplainer: -0.008 [-0.016, +0.000] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.004 [-0.012, +0.004] (n=500)
- GNNExplainer - builtin: +0.025 [+0.017, +0.033] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.004 [+0.001, +0.006] (n=500)
- GradExplainer - builtin: +0.032 [+0.028, +0.038] (n=500)
- IntegratedGradExplainer - builtin: +0.029 [+0.024, +0.035] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.701 [+0.688, +0.713] (n=500)
- GNNExplainer: +0.101 [+0.101, +0.102] (n=500)
- GradExplainer: +0.390 [+0.382, +0.398] (n=500)
- IntegratedGradExplainer: +0.315 [+0.309, +0.321] (n=500)
- GNNExplainer - GradExplainer: -0.289 [-0.296, -0.281] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.214 [-0.219, -0.208] (n=500)
- GNNExplainer - builtin: -0.600 [-0.612, -0.588] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.075 [+0.070, +0.081] (n=500)
- GradExplainer - builtin: -0.311 [-0.322, -0.300] (n=500)
- IntegratedGradExplainer - builtin: -0.386 [-0.396, -0.376] (n=500)

