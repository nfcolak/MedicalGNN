# Fairness checks on the explanation results

Patient-level bootstrap, 2000 resamples, 95% percentile intervals; intervals are withheld below 10 patients.

## 1. How many nodes the main fidelity cut removes

Main cut = top 20% of the visit's eligible evidence nodes, capped at the explanation's own support. `k` is the number actually deleted; `eligible` the visit's eligible nodes; `capped` the share of visits where the support, not the 20%, set k.

**top10/cei_mp2**

- builtin: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- GNNExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- GradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)
- IntegratedGradExplainer: k = 9.3, eligible = 48.4, capped by support in 0% of patients (n=500)

## 2. Matched evidence budgets

Every source scored at the same fixed k. Cells: probability drop with the top-k deleted (fidelity+, higher = more faithful) / with only the top-k kept (fidelity-, lower = more faithful); then the difference to random deletion of the same size on the same visits (+ column: higher is better; - column: lower is better). A source with fewer than k eligible nodes of nonzero importance is `beyond support` at k and is not padded.

### top10/cei_mp2

- builtin, k=1: fidelity+ +0.345 [+0.325, +0.365] (n=500), vs random +0.332 [+0.313, +0.352] (n=500); fidelity- +0.043 [+0.030, +0.056] (n=500), vs random -0.474 [-0.495, -0.454] (n=500)
- builtin, k=2: fidelity+ +0.432 [+0.413, +0.452] (n=500), vs random +0.408 [+0.389, +0.427] (n=500); fidelity- -0.002 [-0.012, +0.008] (n=500), vs random -0.490 [-0.509, -0.471] (n=500)
- builtin, k=3: fidelity+ +0.474 [+0.454, +0.494] (n=500), vs random +0.438 [+0.420, +0.457] (n=500); fidelity- -0.015 [-0.024, -0.007] (n=500), vs random -0.479 [-0.496, -0.460] (n=500)
- builtin, k=5: fidelity+ +0.506 [+0.487, +0.525] (n=500), vs random +0.445 [+0.427, +0.463] (n=500); fidelity- -0.027 [-0.034, -0.021] (n=500), vs random -0.446 [-0.462, -0.429] (n=500)
- builtin, k=10: fidelity+ +0.525 [+0.505, +0.545] (n=493), vs random +0.400 [+0.382, +0.418] (n=493); fidelity- -0.029 [-0.034, -0.024] (n=493), vs random -0.359 [-0.375, -0.342] (n=493)
- GNNExplainer, k=1: fidelity+ +0.194 [+0.178, +0.212] (n=500), vs random +0.182 [+0.165, +0.200] (n=500); fidelity- +0.251 [+0.230, +0.270] (n=500), vs random -0.266 [-0.287, -0.245] (n=500)
- GNNExplainer, k=2: fidelity+ +0.275 [+0.258, +0.295] (n=500), vs random +0.251 [+0.234, +0.270] (n=500); fidelity- +0.145 [+0.127, +0.162] (n=500), vs random -0.343 [-0.365, -0.322] (n=500)
- GNNExplainer, k=3: fidelity+ +0.329 [+0.310, +0.348] (n=500), vs random +0.293 [+0.274, +0.311] (n=500); fidelity- +0.084 [+0.067, +0.099] (n=500), vs random -0.379 [-0.400, -0.358] (n=500)
- GNNExplainer, k=5: fidelity+ +0.389 [+0.370, +0.409] (n=500), vs random +0.328 [+0.310, +0.348] (n=500); fidelity- +0.025 [+0.013, +0.038] (n=500), vs random -0.393 [-0.413, -0.372] (n=500)
- GNNExplainer, k=10: fidelity+ +0.454 [+0.434, +0.473] (n=493), vs random +0.329 [+0.312, +0.346] (n=493); fidelity- -0.016 [-0.024, -0.007] (n=493), vs random -0.346 [-0.362, -0.327] (n=493)
- GradExplainer, k=1: fidelity+ +0.272 [+0.252, +0.293] (n=500), vs random +0.260 [+0.240, +0.281] (n=500); fidelity- +0.143 [+0.125, +0.162] (n=500), vs random -0.374 [-0.396, -0.352] (n=500)
- GradExplainer, k=2: fidelity+ +0.367 [+0.346, +0.388] (n=500), vs random +0.343 [+0.323, +0.363] (n=500); fidelity- +0.055 [+0.042, +0.069] (n=500), vs random -0.433 [-0.453, -0.413] (n=500)
- GradExplainer, k=3: fidelity+ +0.414 [+0.394, +0.435] (n=500), vs random +0.378 [+0.359, +0.398] (n=500); fidelity- +0.029 [+0.018, +0.039] (n=500), vs random -0.435 [-0.454, -0.415] (n=500)
- GradExplainer, k=5: fidelity+ +0.465 [+0.445, +0.485] (n=500), vs random +0.404 [+0.386, +0.423] (n=500); fidelity- +0.010 [+0.002, +0.017] (n=500), vs random -0.409 [-0.428, -0.391] (n=500)
- GradExplainer, k=10: fidelity+ +0.491 [+0.469, +0.511] (n=493), vs random +0.366 [+0.347, +0.383] (n=493); fidelity- +0.005 [-0.001, +0.010] (n=493), vs random -0.326 [-0.342, -0.308] (n=493)
- IntegratedGradExplainer, k=1: fidelity+ +0.332 [+0.311, +0.353] (n=500), vs random +0.320 [+0.299, +0.341] (n=500); fidelity- +0.060 [+0.046, +0.073] (n=500), vs random -0.457 [-0.479, -0.435] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.404 [+0.384, +0.424] (n=500), vs random +0.380 [+0.361, +0.399] (n=500); fidelity- +0.030 [+0.019, +0.042] (n=500), vs random -0.458 [-0.478, -0.438] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.438 [+0.418, +0.457] (n=500), vs random +0.402 [+0.384, +0.421] (n=500); fidelity- +0.020 [+0.010, +0.030] (n=500), vs random -0.443 [-0.463, -0.424] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.462 [+0.442, +0.482] (n=500), vs random +0.401 [+0.383, +0.419] (n=500); fidelity- +0.015 [+0.007, +0.023] (n=500), vs random -0.403 [-0.422, -0.384] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.484 [+0.463, +0.505] (n=493), vs random +0.359 [+0.340, +0.377] (n=493); fidelity- +0.005 [-0.001, +0.011] (n=493), vs random -0.325 [-0.342, -0.308] (n=493)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top10/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.508 [+0.488, +0.528] (n=500)
- GNNExplainer: +0.397 [+0.376, +0.417] (n=500)
- GradExplainer: +0.466 [+0.446, +0.487] (n=500)
- IntegratedGradExplainer: +0.470 [+0.450, +0.490] (n=500)
- GNNExplainer - GradExplainer: -0.070 [-0.083, -0.056] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.074 [-0.089, -0.060] (n=500)
- GNNExplainer - builtin: -0.112 [-0.127, -0.097] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.004 [-0.012, +0.004] (n=500)
- GradExplainer - builtin: -0.042 [-0.050, -0.034] (n=500)
- IntegratedGradExplainer - builtin: -0.038 [-0.043, -0.032] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.022 [-0.028, -0.017] (n=500)
- GNNExplainer: +0.039 [+0.023, +0.054] (n=500)
- GradExplainer: +0.019 [+0.011, +0.027] (n=500)
- IntegratedGradExplainer: +0.013 [+0.006, +0.020] (n=500)
- GNNExplainer - GradExplainer: +0.020 [+0.007, +0.033] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.026 [+0.012, +0.040] (n=500)
- GNNExplainer - builtin: +0.061 [+0.048, +0.075] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.006 [-0.000, +0.012] (n=500)
- GradExplainer - builtin: +0.041 [+0.035, +0.049] (n=500)
- IntegratedGradExplainer - builtin: +0.035 [+0.029, +0.041] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.276 [+0.265, +0.286] (n=500)
- GNNExplainer: +0.203 [+0.193, +0.213] (n=500)
- GradExplainer: +0.245 [+0.234, +0.256] (n=500)
- IntegratedGradExplainer: +0.243 [+0.232, +0.254] (n=500)
- GNNExplainer - GradExplainer: -0.042 [-0.050, -0.035] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.041 [-0.048, -0.033] (n=500)
- GNNExplainer - builtin: -0.073 [-0.081, -0.064] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.002 [-0.001, +0.005] (n=500)
- GradExplainer - builtin: -0.030 [-0.034, -0.026] (n=500)
- IntegratedGradExplainer - builtin: -0.032 [-0.037, -0.028] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.216 [-0.224, -0.208] (n=500)
- GNNExplainer: -0.186 [-0.195, -0.177] (n=500)
- GradExplainer: -0.195 [-0.203, -0.186] (n=500)
- IntegratedGradExplainer: -0.196 [-0.204, -0.187] (n=500)
- GNNExplainer - GradExplainer: +0.009 [+0.003, +0.014] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.010 [+0.003, +0.016] (n=500)
- GNNExplainer - builtin: +0.030 [+0.024, +0.036] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.001 [-0.001, +0.004] (n=500)
- GradExplainer - builtin: +0.021 [+0.018, +0.024] (n=500)
- IntegratedGradExplainer - builtin: +0.020 [+0.017, +0.023] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.699 [+0.684, +0.714] (n=500)
- GNNExplainer: +0.100 [+0.100, +0.101] (n=500)
- GradExplainer: +0.414 [+0.404, +0.423] (n=500)
- IntegratedGradExplainer: +0.381 [+0.372, +0.390] (n=500)
- GNNExplainer - GradExplainer: -0.313 [-0.323, -0.304] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.281 [-0.290, -0.272] (n=500)
- GNNExplainer - builtin: -0.598 [-0.614, -0.583] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.033 [+0.027, +0.038] (n=500)
- GradExplainer - builtin: -0.285 [-0.300, -0.271] (n=500)
- IntegratedGradExplainer - builtin: -0.318 [-0.330, -0.305] (n=500)

