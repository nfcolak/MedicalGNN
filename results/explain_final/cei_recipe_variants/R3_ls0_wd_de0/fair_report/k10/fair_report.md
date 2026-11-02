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

- builtin, k=1: fidelity+ +0.354 [+0.331, +0.377] (n=500), vs random +0.341 [+0.319, +0.363] (n=500); fidelity- +0.057 [+0.042, +0.073] (n=500), vs random -0.525 [-0.549, -0.502] (n=500)
- builtin, k=2: fidelity+ +0.458 [+0.436, +0.480] (n=500), vs random +0.432 [+0.412, +0.453] (n=500); fidelity- +0.027 [+0.014, +0.040] (n=500), vs random -0.521 [-0.543, -0.499] (n=500)
- builtin, k=3: fidelity+ +0.504 [+0.483, +0.526] (n=500), vs random +0.467 [+0.446, +0.486] (n=500); fidelity- +0.020 [+0.008, +0.032] (n=500), vs random -0.498 [-0.519, -0.477] (n=500)
- builtin, k=5: fidelity+ +0.544 [+0.522, +0.565] (n=500), vs random +0.478 [+0.458, +0.498] (n=500); fidelity- +0.007 [-0.003, +0.017] (n=500), vs random -0.459 [-0.478, -0.439] (n=500)
- builtin, k=10: fidelity+ +0.583 [+0.561, +0.605] (n=493), vs random +0.448 [+0.427, +0.467] (n=493); fidelity- -0.007 [-0.015, +0.000] (n=493), vs random -0.373 [-0.391, -0.354] (n=493)
- GNNExplainer, k=1: fidelity+ +0.246 [+0.225, +0.270] (n=500), vs random +0.233 [+0.213, +0.256] (n=500); fidelity- +0.217 [+0.194, +0.240] (n=500), vs random -0.365 [-0.390, -0.339] (n=500)
- GNNExplainer, k=2: fidelity+ +0.331 [+0.308, +0.355] (n=500), vs random +0.306 [+0.284, +0.329] (n=500); fidelity- +0.124 [+0.105, +0.144] (n=500), vs random -0.424 [-0.448, -0.401] (n=500)
- GNNExplainer, k=3: fidelity+ +0.380 [+0.358, +0.402] (n=500), vs random +0.342 [+0.321, +0.364] (n=500); fidelity- +0.078 [+0.061, +0.096] (n=500), vs random -0.441 [-0.463, -0.417] (n=500)
- GNNExplainer, k=5: fidelity+ +0.429 [+0.406, +0.451] (n=500), vs random +0.363 [+0.342, +0.384] (n=500); fidelity- +0.035 [+0.020, +0.050] (n=500), vs random -0.431 [-0.453, -0.409] (n=500)
- GNNExplainer, k=10: fidelity+ +0.476 [+0.453, +0.498] (n=493), vs random +0.341 [+0.318, +0.360] (n=493); fidelity- +0.005 [-0.007, +0.016] (n=493), vs random -0.361 [-0.380, -0.339] (n=493)
- GradExplainer, k=1: fidelity+ +0.274 [+0.252, +0.297] (n=500), vs random +0.261 [+0.240, +0.284] (n=500); fidelity- +0.180 [+0.159, +0.202] (n=500), vs random -0.402 [-0.427, -0.377] (n=500)
- GradExplainer, k=2: fidelity+ +0.382 [+0.359, +0.406] (n=500), vs random +0.356 [+0.334, +0.380] (n=500); fidelity- +0.081 [+0.065, +0.097] (n=500), vs random -0.468 [-0.492, -0.444] (n=500)
- GradExplainer, k=3: fidelity+ +0.433 [+0.410, +0.457] (n=500), vs random +0.396 [+0.374, +0.418] (n=500); fidelity- +0.061 [+0.048, +0.075] (n=500), vs random -0.457 [-0.480, -0.435] (n=500)
- GradExplainer, k=5: fidelity+ +0.500 [+0.478, +0.523] (n=500), vs random +0.435 [+0.414, +0.457] (n=500); fidelity- +0.047 [+0.036, +0.059] (n=500), vs random -0.419 [-0.440, -0.397] (n=500)
- GradExplainer, k=10: fidelity+ +0.536 [+0.512, +0.558] (n=493), vs random +0.401 [+0.379, +0.421] (n=493); fidelity- +0.035 [+0.026, +0.044] (n=493), vs random -0.331 [-0.350, -0.311] (n=493)
- IntegratedGradExplainer, k=1: fidelity+ +0.342 [+0.317, +0.365] (n=500), vs random +0.329 [+0.305, +0.352] (n=500); fidelity- +0.079 [+0.062, +0.096] (n=500), vs random -0.503 [-0.528, -0.478] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.437 [+0.413, +0.459] (n=500), vs random +0.411 [+0.388, +0.433] (n=500); fidelity- +0.046 [+0.032, +0.060] (n=500), vs random -0.502 [-0.526, -0.479] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.480 [+0.458, +0.503] (n=500), vs random +0.443 [+0.421, +0.464] (n=500); fidelity- +0.039 [+0.028, +0.052] (n=500), vs random -0.479 [-0.501, -0.456] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.517 [+0.495, +0.540] (n=500), vs random +0.451 [+0.430, +0.473] (n=500); fidelity- +0.040 [+0.030, +0.051] (n=500), vs random -0.426 [-0.447, -0.404] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.545 [+0.523, +0.568] (n=493), vs random +0.410 [+0.389, +0.429] (n=493); fidelity- +0.029 [+0.020, +0.037] (n=493), vs random -0.337 [-0.356, -0.318] (n=493)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top10/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.557 [+0.534, +0.579] (n=500)
- GNNExplainer: +0.426 [+0.404, +0.448] (n=500)
- GradExplainer: +0.502 [+0.479, +0.525] (n=500)
- IntegratedGradExplainer: +0.525 [+0.502, +0.547] (n=500)
- GNNExplainer - GradExplainer: -0.076 [-0.093, -0.059] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.100 [-0.120, -0.081] (n=500)
- GNNExplainer - builtin: -0.131 [-0.149, -0.114] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.023 [-0.034, -0.014] (n=500)
- GradExplainer - builtin: -0.055 [-0.067, -0.044] (n=500)
- IntegratedGradExplainer - builtin: -0.032 [-0.040, -0.023] (n=500)

**fidelity-** (lower = more faithful)

- builtin: +0.001 [-0.008, +0.009] (n=500)
- GNNExplainer: +0.053 [+0.036, +0.070] (n=500)
- GradExplainer: +0.049 [+0.038, +0.060] (n=500)
- IntegratedGradExplainer: +0.038 [+0.028, +0.047] (n=500)
- GNNExplainer - GradExplainer: +0.004 [-0.012, +0.021] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.015 [-0.002, +0.033] (n=500)
- GNNExplainer - builtin: +0.052 [+0.036, +0.069] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.011 [+0.004, +0.019] (n=500)
- GradExplainer - builtin: +0.048 [+0.039, +0.059] (n=500)
- IntegratedGradExplainer - builtin: +0.037 [+0.029, +0.045] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.321 [+0.309, +0.333] (n=500)
- GNNExplainer: +0.219 [+0.207, +0.230] (n=500)
- GradExplainer: +0.268 [+0.255, +0.281] (n=500)
- IntegratedGradExplainer: +0.286 [+0.272, +0.298] (n=500)
- GNNExplainer - GradExplainer: -0.050 [-0.060, -0.039] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.067 [-0.078, -0.056] (n=500)
- GNNExplainer - builtin: -0.102 [-0.113, -0.091] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.017 [-0.022, -0.012] (n=500)
- GradExplainer - builtin: -0.053 [-0.059, -0.045] (n=500)
- IntegratedGradExplainer - builtin: -0.035 [-0.042, -0.028] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.227 [-0.236, -0.218] (n=500)
- GNNExplainer: -0.200 [-0.210, -0.191] (n=500)
- GradExplainer: -0.205 [-0.214, -0.195] (n=500)
- IntegratedGradExplainer: -0.208 [-0.219, -0.198] (n=500)
- GNNExplainer - GradExplainer: +0.004 [-0.003, +0.012] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.008 [+0.000, +0.016] (n=500)
- GNNExplainer - builtin: +0.026 [+0.019, +0.034] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.004 [+0.001, +0.007] (n=500)
- GradExplainer - builtin: +0.022 [+0.018, +0.027] (n=500)
- IntegratedGradExplainer - builtin: +0.018 [+0.014, +0.023] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.655 [+0.642, +0.668] (n=500)
- GNNExplainer: +0.100 [+0.099, +0.101] (n=500)
- GradExplainer: +0.416 [+0.408, +0.424] (n=500)
- IntegratedGradExplainer: +0.362 [+0.355, +0.369] (n=500)
- GNNExplainer - GradExplainer: -0.315 [-0.324, -0.308] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.262 [-0.269, -0.255] (n=500)
- GNNExplainer - builtin: -0.555 [-0.568, -0.542] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.054 [+0.048, +0.059] (n=500)
- GradExplainer - builtin: -0.239 [-0.251, -0.228] (n=500)
- IntegratedGradExplainer - builtin: -0.293 [-0.302, -0.283] (n=500)

