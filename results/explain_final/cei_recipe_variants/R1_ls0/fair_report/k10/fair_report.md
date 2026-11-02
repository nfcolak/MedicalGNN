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

- builtin, k=1: fidelity+ +0.355 [+0.333, +0.378] (n=500), vs random +0.342 [+0.321, +0.364] (n=500); fidelity- +0.046 [+0.031, +0.060] (n=500), vs random -0.536 [-0.559, -0.512] (n=500)
- builtin, k=2: fidelity+ +0.454 [+0.431, +0.476] (n=500), vs random +0.429 [+0.408, +0.450] (n=500); fidelity- +0.018 [+0.005, +0.030] (n=500), vs random -0.530 [-0.552, -0.508] (n=500)
- builtin, k=3: fidelity+ +0.504 [+0.483, +0.525] (n=500), vs random +0.467 [+0.447, +0.486] (n=500); fidelity- +0.012 [+0.001, +0.023] (n=500), vs random -0.505 [-0.526, -0.485] (n=500)
- builtin, k=5: fidelity+ +0.549 [+0.528, +0.570] (n=500), vs random +0.485 [+0.464, +0.504] (n=500); fidelity- -0.002 [-0.011, +0.007] (n=500), vs random -0.466 [-0.485, -0.447] (n=500)
- builtin, k=10: fidelity+ +0.596 [+0.575, +0.618] (n=493), vs random +0.463 [+0.443, +0.482] (n=493); fidelity- -0.016 [-0.023, -0.009] (n=493), vs random -0.380 [-0.398, -0.362] (n=493)
- GNNExplainer, k=1: fidelity+ +0.259 [+0.238, +0.284] (n=500), vs random +0.247 [+0.225, +0.271] (n=500); fidelity- +0.194 [+0.172, +0.216] (n=500), vs random -0.388 [-0.414, -0.361] (n=500)
- GNNExplainer, k=2: fidelity+ +0.338 [+0.315, +0.362] (n=500), vs random +0.312 [+0.290, +0.336] (n=500); fidelity- +0.109 [+0.089, +0.130] (n=500), vs random -0.438 [-0.464, -0.413] (n=500)
- GNNExplainer, k=3: fidelity+ +0.377 [+0.354, +0.401] (n=500), vs random +0.340 [+0.318, +0.363] (n=500); fidelity- +0.064 [+0.046, +0.082] (n=500), vs random -0.453 [-0.478, -0.430] (n=500)
- GNNExplainer, k=5: fidelity+ +0.427 [+0.405, +0.449] (n=500), vs random +0.362 [+0.342, +0.384] (n=500); fidelity- +0.026 [+0.011, +0.040] (n=500), vs random -0.438 [-0.461, -0.417] (n=500)
- GNNExplainer, k=10: fidelity+ +0.469 [+0.447, +0.491] (n=493), vs random +0.336 [+0.315, +0.356] (n=493); fidelity- -0.001 [-0.012, +0.009] (n=493), vs random -0.365 [-0.385, -0.345] (n=493)
- GradExplainer, k=1: fidelity+ +0.274 [+0.251, +0.297] (n=500), vs random +0.261 [+0.239, +0.285] (n=500); fidelity- +0.158 [+0.138, +0.178] (n=500), vs random -0.424 [-0.450, -0.398] (n=500)
- GradExplainer, k=2: fidelity+ +0.373 [+0.350, +0.397] (n=500), vs random +0.348 [+0.325, +0.371] (n=500); fidelity- +0.079 [+0.064, +0.094] (n=500), vs random -0.468 [-0.493, -0.445] (n=500)
- GradExplainer, k=3: fidelity+ +0.427 [+0.404, +0.450] (n=500), vs random +0.390 [+0.368, +0.412] (n=500); fidelity- +0.053 [+0.039, +0.067] (n=500), vs random -0.464 [-0.487, -0.441] (n=500)
- GradExplainer, k=5: fidelity+ +0.486 [+0.464, +0.508] (n=500), vs random +0.421 [+0.400, +0.442] (n=500); fidelity- +0.044 [+0.034, +0.055] (n=500), vs random -0.420 [-0.441, -0.399] (n=500)
- GradExplainer, k=10: fidelity+ +0.529 [+0.506, +0.551] (n=493), vs random +0.396 [+0.374, +0.416] (n=493); fidelity- +0.028 [+0.020, +0.035] (n=493), vs random -0.337 [-0.356, -0.317] (n=493)
- IntegratedGradExplainer, k=1: fidelity+ +0.341 [+0.317, +0.365] (n=500), vs random +0.328 [+0.305, +0.351] (n=500); fidelity- +0.070 [+0.053, +0.085] (n=500), vs random -0.512 [-0.538, -0.488] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.432 [+0.409, +0.455] (n=500), vs random +0.407 [+0.385, +0.429] (n=500); fidelity- +0.037 [+0.025, +0.050] (n=500), vs random -0.510 [-0.533, -0.487] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.472 [+0.450, +0.494] (n=500), vs random +0.435 [+0.414, +0.455] (n=500); fidelity- +0.035 [+0.024, +0.047] (n=500), vs random -0.482 [-0.504, -0.459] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.509 [+0.487, +0.531] (n=500), vs random +0.445 [+0.424, +0.465] (n=500); fidelity- +0.031 [+0.022, +0.041] (n=500), vs random -0.433 [-0.454, -0.412] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.542 [+0.520, +0.565] (n=493), vs random +0.409 [+0.388, +0.429] (n=493); fidelity- +0.024 [+0.016, +0.032] (n=493), vs random -0.340 [-0.359, -0.321] (n=493)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top10/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.562 [+0.540, +0.583] (n=500)
- GNNExplainer: +0.426 [+0.404, +0.448] (n=500)
- GradExplainer: +0.501 [+0.478, +0.523] (n=500)
- IntegratedGradExplainer: +0.519 [+0.497, +0.541] (n=500)
- GNNExplainer - GradExplainer: -0.075 [-0.093, -0.057] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.093 [-0.113, -0.073] (n=500)
- GNNExplainer - builtin: -0.136 [-0.155, -0.117] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.018 [-0.028, -0.010] (n=500)
- GradExplainer - builtin: -0.061 [-0.072, -0.051] (n=500)
- IntegratedGradExplainer - builtin: -0.043 [-0.050, -0.035] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.007 [-0.015, +0.001] (n=500)
- GNNExplainer: +0.047 [+0.030, +0.064] (n=500)
- GradExplainer: +0.039 [+0.030, +0.049] (n=500)
- IntegratedGradExplainer: +0.029 [+0.020, +0.037] (n=500)
- GNNExplainer - GradExplainer: +0.007 [-0.009, +0.024] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.018 [+0.000, +0.036] (n=500)
- GNNExplainer - builtin: +0.054 [+0.036, +0.071] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.010 [+0.004, +0.017] (n=500)
- GradExplainer - builtin: +0.046 [+0.038, +0.055] (n=500)
- IntegratedGradExplainer - builtin: +0.036 [+0.028, +0.044] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.329 [+0.317, +0.341] (n=500)
- GNNExplainer: +0.216 [+0.205, +0.228] (n=500)
- GradExplainer: +0.269 [+0.257, +0.281] (n=500)
- IntegratedGradExplainer: +0.287 [+0.274, +0.299] (n=500)
- GNNExplainer - GradExplainer: -0.053 [-0.063, -0.042] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.071 [-0.083, -0.059] (n=500)
- GNNExplainer - builtin: -0.113 [-0.125, -0.101] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.018 [-0.023, -0.014] (n=500)
- GradExplainer - builtin: -0.060 [-0.067, -0.053] (n=500)
- IntegratedGradExplainer - builtin: -0.042 [-0.048, -0.036] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.229 [-0.237, -0.221] (n=500)
- GNNExplainer: -0.202 [-0.212, -0.192] (n=500)
- GradExplainer: -0.205 [-0.215, -0.196] (n=500)
- IntegratedGradExplainer: -0.210 [-0.220, -0.200] (n=500)
- GNNExplainer - GradExplainer: +0.003 [-0.004, +0.011] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.008 [+0.001, +0.016] (n=500)
- GNNExplainer - builtin: +0.027 [+0.020, +0.035] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.005 [+0.002, +0.007] (n=500)
- GradExplainer - builtin: +0.024 [+0.019, +0.028] (n=500)
- IntegratedGradExplainer - builtin: +0.019 [+0.015, +0.024] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.654 [+0.640, +0.668] (n=500)
- GNNExplainer: +0.100 [+0.099, +0.101] (n=500)
- GradExplainer: +0.422 [+0.414, +0.430] (n=500)
- IntegratedGradExplainer: +0.383 [+0.376, +0.391] (n=500)
- GNNExplainer - GradExplainer: -0.322 [-0.331, -0.315] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.283 [-0.291, -0.276] (n=500)
- GNNExplainer - builtin: -0.555 [-0.568, -0.541] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.039 [+0.034, +0.045] (n=500)
- GradExplainer - builtin: -0.232 [-0.244, -0.220] (n=500)
- IntegratedGradExplainer - builtin: -0.271 [-0.281, -0.261] (n=500)

