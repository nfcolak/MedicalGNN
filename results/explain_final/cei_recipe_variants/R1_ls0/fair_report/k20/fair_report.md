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

- builtin, k=1: fidelity+ +0.396 [+0.373, +0.418] (n=500), vs random +0.382 [+0.360, +0.404] (n=500); fidelity- +0.039 [+0.022, +0.056] (n=500), vs random -0.569 [-0.592, -0.543] (n=500)
- builtin, k=2: fidelity+ +0.503 [+0.481, +0.525] (n=500), vs random +0.476 [+0.456, +0.497] (n=500); fidelity- +0.005 [-0.007, +0.019] (n=500), vs random -0.570 [-0.592, -0.547] (n=500)
- builtin, k=3: fidelity+ +0.542 [+0.521, +0.564] (n=500), vs random +0.503 [+0.483, +0.523] (n=500); fidelity- +0.004 [-0.007, +0.016] (n=500), vs random -0.543 [-0.564, -0.521] (n=500)
- builtin, k=5: fidelity+ +0.575 [+0.555, +0.596] (n=500), vs random +0.510 [+0.490, +0.529] (n=500); fidelity- -0.001 [-0.011, +0.009] (n=500), vs random -0.497 [-0.517, -0.477] (n=500)
- builtin, k=10: fidelity+ +0.607 [+0.586, +0.628] (n=494), vs random +0.475 [+0.456, +0.495] (n=494); fidelity- -0.005 [-0.013, +0.004] (n=494), vs random -0.398 [-0.418, -0.380] (n=494)
- GNNExplainer, k=1: fidelity+ +0.254 [+0.232, +0.277] (n=500), vs random +0.241 [+0.219, +0.263] (n=500); fidelity- +0.232 [+0.209, +0.256] (n=500), vs random -0.376 [-0.403, -0.349] (n=500)
- GNNExplainer, k=2: fidelity+ +0.359 [+0.335, +0.384] (n=500), vs random +0.332 [+0.309, +0.356] (n=500); fidelity- +0.121 [+0.102, +0.141] (n=500), vs random -0.455 [-0.480, -0.430] (n=500)
- GNNExplainer, k=3: fidelity+ +0.411 [+0.388, +0.437] (n=500), vs random +0.372 [+0.349, +0.397] (n=500); fidelity- +0.068 [+0.050, +0.087] (n=500), vs random -0.479 [-0.503, -0.455] (n=500)
- GNNExplainer, k=5: fidelity+ +0.474 [+0.452, +0.497] (n=500), vs random +0.408 [+0.387, +0.430] (n=500); fidelity- +0.019 [+0.004, +0.034] (n=500), vs random -0.477 [-0.499, -0.454] (n=500)
- GNNExplainer, k=10: fidelity+ +0.523 [+0.502, +0.545] (n=494), vs random +0.391 [+0.372, +0.411] (n=494); fidelity- -0.012 [-0.024, +0.000] (n=494), vs random -0.405 [-0.426, -0.386] (n=494)
- GradExplainer, k=1: fidelity+ +0.285 [+0.262, +0.308] (n=500), vs random +0.271 [+0.250, +0.294] (n=500); fidelity- +0.199 [+0.176, +0.221] (n=500), vs random -0.409 [-0.435, -0.384] (n=500)
- GradExplainer, k=2: fidelity+ +0.412 [+0.389, +0.436] (n=500), vs random +0.385 [+0.362, +0.409] (n=500); fidelity- +0.091 [+0.075, +0.108] (n=500), vs random -0.485 [-0.508, -0.460] (n=500)
- GradExplainer, k=3: fidelity+ +0.478 [+0.455, +0.502] (n=500), vs random +0.439 [+0.417, +0.461] (n=500); fidelity- +0.067 [+0.053, +0.081] (n=500), vs random -0.480 [-0.502, -0.456] (n=500)
- GradExplainer, k=5: fidelity+ +0.528 [+0.505, +0.552] (n=500), vs random +0.463 [+0.441, +0.484] (n=500); fidelity- +0.061 [+0.048, +0.073] (n=500), vs random -0.435 [-0.457, -0.413] (n=500)
- GradExplainer, k=10: fidelity+ +0.562 [+0.538, +0.585] (n=494), vs random +0.430 [+0.408, +0.451] (n=494); fidelity- +0.049 [+0.040, +0.059] (n=494), vs random -0.344 [-0.365, -0.324] (n=494)
- IntegratedGradExplainer, k=1: fidelity+ +0.361 [+0.337, +0.386] (n=500), vs random +0.348 [+0.324, +0.372] (n=500); fidelity- +0.096 [+0.077, +0.115] (n=500), vs random -0.512 [-0.538, -0.486] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.468 [+0.445, +0.492] (n=500), vs random +0.442 [+0.420, +0.464] (n=500); fidelity- +0.048 [+0.034, +0.063] (n=500), vs random -0.528 [-0.551, -0.503] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.518 [+0.496, +0.541] (n=500), vs random +0.479 [+0.457, +0.500] (n=500); fidelity- +0.041 [+0.029, +0.054] (n=500), vs random -0.505 [-0.528, -0.481] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.549 [+0.526, +0.572] (n=500), vs random +0.483 [+0.461, +0.504] (n=500); fidelity- +0.051 [+0.039, +0.063] (n=500), vs random -0.444 [-0.467, -0.422] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.570 [+0.547, +0.593] (n=494), vs random +0.438 [+0.417, +0.459] (n=494); fidelity- +0.047 [+0.036, +0.058] (n=494), vs random -0.346 [-0.367, -0.327] (n=494)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top20/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.585 [+0.566, +0.606] (n=500)
- GNNExplainer: +0.474 [+0.451, +0.497] (n=500)
- GradExplainer: +0.540 [+0.518, +0.562] (n=500)
- IntegratedGradExplainer: +0.556 [+0.535, +0.579] (n=500)
- GNNExplainer - GradExplainer: -0.067 [-0.084, -0.049] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.083 [-0.101, -0.064] (n=500)
- GNNExplainer - builtin: -0.112 [-0.129, -0.095] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.016 [-0.022, -0.010] (n=500)
- GradExplainer - builtin: -0.045 [-0.054, -0.037] (n=500)
- IntegratedGradExplainer - builtin: -0.029 [-0.037, -0.022] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.000 [-0.009, +0.010] (n=500)
- GNNExplainer: +0.034 [+0.017, +0.052] (n=500)
- GradExplainer: +0.061 [+0.051, +0.072] (n=500)
- IntegratedGradExplainer: +0.053 [+0.042, +0.064] (n=500)
- GNNExplainer - GradExplainer: -0.027 [-0.045, -0.009] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.019 [-0.038, -0.000] (n=500)
- GNNExplainer - builtin: +0.034 [+0.016, +0.053] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.008 [+0.002, +0.014] (n=500)
- GradExplainer - builtin: +0.061 [+0.051, +0.071] (n=500)
- IntegratedGradExplainer - builtin: +0.053 [+0.043, +0.063] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.323 [+0.312, +0.333] (n=500)
- GNNExplainer: +0.244 [+0.232, +0.255] (n=500)
- GradExplainer: +0.283 [+0.271, +0.295] (n=500)
- IntegratedGradExplainer: +0.291 [+0.278, +0.303] (n=500)
- GNNExplainer - GradExplainer: -0.039 [-0.049, -0.029] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.047 [-0.058, -0.037] (n=500)
- GNNExplainer - builtin: -0.079 [-0.089, -0.069] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.008 [-0.010, -0.005] (n=500)
- GradExplainer - builtin: -0.040 [-0.045, -0.034] (n=500)
- IntegratedGradExplainer - builtin: -0.032 [-0.038, -0.026] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.240 [-0.249, -0.230] (n=500)
- GNNExplainer: -0.223 [-0.234, -0.213] (n=500)
- GradExplainer: -0.207 [-0.218, -0.197] (n=500)
- IntegratedGradExplainer: -0.211 [-0.222, -0.201] (n=500)
- GNNExplainer - GradExplainer: -0.016 [-0.024, -0.009] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.013 [-0.021, -0.005] (n=500)
- GNNExplainer - builtin: +0.016 [+0.008, +0.024] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.004 [+0.001, +0.006] (n=500)
- GradExplainer - builtin: +0.032 [+0.028, +0.037] (n=500)
- IntegratedGradExplainer - builtin: +0.029 [+0.024, +0.034] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.703 [+0.691, +0.715] (n=500)
- GNNExplainer: +0.101 [+0.100, +0.102] (n=500)
- GradExplainer: +0.391 [+0.384, +0.400] (n=500)
- IntegratedGradExplainer: +0.329 [+0.323, +0.335] (n=500)
- GNNExplainer - GradExplainer: -0.291 [-0.298, -0.283] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.228 [-0.234, -0.222] (n=500)
- GNNExplainer - builtin: -0.603 [-0.614, -0.591] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.063 [+0.058, +0.067] (n=500)
- GradExplainer - builtin: -0.312 [-0.323, -0.301] (n=500)
- IntegratedGradExplainer - builtin: -0.375 [-0.384, -0.365] (n=500)

