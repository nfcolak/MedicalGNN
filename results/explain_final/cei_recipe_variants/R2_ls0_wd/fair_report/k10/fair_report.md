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

- builtin, k=1: fidelity+ +0.356 [+0.334, +0.379] (n=500), vs random +0.343 [+0.322, +0.365] (n=500); fidelity- +0.045 [+0.030, +0.060] (n=500), vs random -0.539 [-0.562, -0.516] (n=500)
- builtin, k=2: fidelity+ +0.457 [+0.434, +0.479] (n=500), vs random +0.431 [+0.410, +0.452] (n=500); fidelity- +0.017 [+0.005, +0.030] (n=500), vs random -0.532 [-0.554, -0.510] (n=500)
- builtin, k=3: fidelity+ +0.505 [+0.484, +0.526] (n=500), vs random +0.468 [+0.448, +0.487] (n=500); fidelity- +0.011 [+0.000, +0.022] (n=500), vs random -0.508 [-0.528, -0.487] (n=500)
- builtin, k=5: fidelity+ +0.556 [+0.534, +0.577] (n=500), vs random +0.491 [+0.470, +0.511] (n=500); fidelity- +0.000 [-0.009, +0.010] (n=500), vs random -0.465 [-0.485, -0.446] (n=500)
- builtin, k=10: fidelity+ +0.601 [+0.579, +0.623] (n=493), vs random +0.467 [+0.447, +0.487] (n=493); fidelity- -0.017 [-0.024, -0.010] (n=493), vs random -0.382 [-0.399, -0.364] (n=493)
- GNNExplainer, k=1: fidelity+ +0.259 [+0.238, +0.282] (n=500), vs random +0.246 [+0.225, +0.269] (n=500); fidelity- +0.194 [+0.172, +0.215] (n=500), vs random -0.390 [-0.417, -0.364] (n=500)
- GNNExplainer, k=2: fidelity+ +0.335 [+0.312, +0.359] (n=500), vs random +0.309 [+0.287, +0.333] (n=500); fidelity- +0.112 [+0.092, +0.131] (n=500), vs random -0.437 [-0.464, -0.413] (n=500)
- GNNExplainer, k=3: fidelity+ +0.375 [+0.352, +0.398] (n=500), vs random +0.337 [+0.315, +0.361] (n=500); fidelity- +0.073 [+0.054, +0.092] (n=500), vs random -0.446 [-0.470, -0.422] (n=500)
- GNNExplainer, k=5: fidelity+ +0.424 [+0.402, +0.447] (n=500), vs random +0.360 [+0.339, +0.381] (n=500); fidelity- +0.027 [+0.012, +0.041] (n=500), vs random -0.439 [-0.462, -0.418] (n=500)
- GNNExplainer, k=10: fidelity+ +0.468 [+0.445, +0.490] (n=493), vs random +0.334 [+0.313, +0.354] (n=493); fidelity- +0.001 [-0.010, +0.011] (n=493), vs random -0.365 [-0.384, -0.343] (n=493)
- GradExplainer, k=1: fidelity+ +0.272 [+0.249, +0.296] (n=500), vs random +0.259 [+0.237, +0.284] (n=500); fidelity- +0.165 [+0.144, +0.186] (n=500), vs random -0.419 [-0.446, -0.393] (n=500)
- GradExplainer, k=2: fidelity+ +0.374 [+0.351, +0.398] (n=500), vs random +0.349 [+0.326, +0.372] (n=500); fidelity- +0.084 [+0.069, +0.100] (n=500), vs random -0.465 [-0.489, -0.442] (n=500)
- GradExplainer, k=3: fidelity+ +0.427 [+0.403, +0.450] (n=500), vs random +0.389 [+0.367, +0.411] (n=500); fidelity- +0.054 [+0.040, +0.067] (n=500), vs random -0.465 [-0.488, -0.442] (n=500)
- GradExplainer, k=5: fidelity+ +0.488 [+0.466, +0.510] (n=500), vs random +0.423 [+0.403, +0.445] (n=500); fidelity- +0.046 [+0.035, +0.056] (n=500), vs random -0.420 [-0.442, -0.398] (n=500)
- GradExplainer, k=10: fidelity+ +0.531 [+0.509, +0.552] (n=493), vs random +0.397 [+0.376, +0.416] (n=493); fidelity- +0.027 [+0.019, +0.035] (n=493), vs random -0.338 [-0.357, -0.318] (n=493)
- IntegratedGradExplainer, k=1: fidelity+ +0.340 [+0.317, +0.364] (n=500), vs random +0.328 [+0.305, +0.351] (n=500); fidelity- +0.070 [+0.053, +0.085] (n=500), vs random -0.514 [-0.540, -0.490] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.430 [+0.406, +0.453] (n=500), vs random +0.404 [+0.382, +0.426] (n=500); fidelity- +0.040 [+0.027, +0.054] (n=500), vs random -0.509 [-0.532, -0.486] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.474 [+0.452, +0.495] (n=500), vs random +0.436 [+0.415, +0.456] (n=500); fidelity- +0.038 [+0.026, +0.049] (n=500), vs random -0.481 [-0.503, -0.459] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.507 [+0.485, +0.528] (n=500), vs random +0.442 [+0.422, +0.462] (n=500); fidelity- +0.034 [+0.024, +0.044] (n=500), vs random -0.432 [-0.453, -0.411] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.538 [+0.516, +0.561] (n=493), vs random +0.405 [+0.383, +0.424] (n=493); fidelity- +0.026 [+0.017, +0.033] (n=493), vs random -0.340 [-0.358, -0.320] (n=493)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top10/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.567 [+0.545, +0.588] (n=500)
- GNNExplainer: +0.421 [+0.399, +0.443] (n=500)
- GradExplainer: +0.498 [+0.475, +0.520] (n=500)
- IntegratedGradExplainer: +0.516 [+0.494, +0.537] (n=500)
- GNNExplainer - GradExplainer: -0.077 [-0.094, -0.060] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.095 [-0.115, -0.076] (n=500)
- GNNExplainer - builtin: -0.146 [-0.165, -0.127] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.018 [-0.028, -0.009] (n=500)
- GradExplainer - builtin: -0.069 [-0.080, -0.059] (n=500)
- IntegratedGradExplainer - builtin: -0.051 [-0.058, -0.043] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.009 [-0.017, -0.001] (n=500)
- GNNExplainer: +0.051 [+0.033, +0.068] (n=500)
- GradExplainer: +0.044 [+0.034, +0.054] (n=500)
- IntegratedGradExplainer: +0.031 [+0.022, +0.040] (n=500)
- GNNExplainer - GradExplainer: +0.007 [-0.010, +0.024] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.020 [+0.002, +0.038] (n=500)
- GNNExplainer - builtin: +0.060 [+0.043, +0.077] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.013 [+0.006, +0.019] (n=500)
- GradExplainer - builtin: +0.053 [+0.044, +0.062] (n=500)
- IntegratedGradExplainer - builtin: +0.040 [+0.032, +0.049] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.332 [+0.320, +0.343] (n=500)
- GNNExplainer: +0.216 [+0.205, +0.228] (n=500)
- GradExplainer: +0.267 [+0.255, +0.279] (n=500)
- IntegratedGradExplainer: +0.284 [+0.271, +0.296] (n=500)
- GNNExplainer - GradExplainer: -0.051 [-0.061, -0.041] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.068 [-0.080, -0.057] (n=500)
- GNNExplainer - builtin: -0.116 [-0.128, -0.104] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.017 [-0.022, -0.012] (n=500)
- GradExplainer - builtin: -0.065 [-0.071, -0.058] (n=500)
- IntegratedGradExplainer - builtin: -0.048 [-0.054, -0.041] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.230 [-0.239, -0.222] (n=500)
- GNNExplainer: -0.202 [-0.212, -0.191] (n=500)
- GradExplainer: -0.205 [-0.215, -0.195] (n=500)
- IntegratedGradExplainer: -0.210 [-0.220, -0.200] (n=500)
- GNNExplainer - GradExplainer: +0.003 [-0.004, +0.010] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.008 [+0.000, +0.016] (n=500)
- GNNExplainer - builtin: +0.028 [+0.021, +0.036] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.005 [+0.002, +0.007] (n=500)
- GradExplainer - builtin: +0.025 [+0.021, +0.030] (n=500)
- IntegratedGradExplainer - builtin: +0.020 [+0.016, +0.025] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.651 [+0.637, +0.664] (n=500)
- GNNExplainer: +0.100 [+0.099, +0.101] (n=500)
- GradExplainer: +0.421 [+0.413, +0.429] (n=500)
- IntegratedGradExplainer: +0.383 [+0.375, +0.391] (n=500)
- GNNExplainer - GradExplainer: -0.321 [-0.329, -0.313] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.283 [-0.291, -0.276] (n=500)
- GNNExplainer - builtin: -0.551 [-0.565, -0.537] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.038 [+0.033, +0.044] (n=500)
- GradExplainer - builtin: -0.230 [-0.242, -0.218] (n=500)
- IntegratedGradExplainer - builtin: -0.268 [-0.278, -0.258] (n=500)

