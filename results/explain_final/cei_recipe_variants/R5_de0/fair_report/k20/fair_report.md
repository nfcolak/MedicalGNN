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

- builtin, k=1: fidelity+ +0.368 [+0.347, +0.388] (n=500), vs random +0.355 [+0.335, +0.374] (n=500); fidelity- +0.057 [+0.040, +0.073] (n=500), vs random -0.474 [-0.494, -0.452] (n=500)
- builtin, k=2: fidelity+ +0.459 [+0.439, +0.480] (n=500), vs random +0.434 [+0.415, +0.454] (n=500); fidelity- +0.008 [-0.004, +0.021] (n=500), vs random -0.496 [-0.516, -0.475] (n=500)
- builtin, k=3: fidelity+ +0.487 [+0.466, +0.507] (n=500), vs random +0.450 [+0.431, +0.469] (n=500); fidelity- +0.003 [-0.007, +0.012] (n=500), vs random -0.478 [-0.498, -0.458] (n=500)
- builtin, k=5: fidelity+ +0.511 [+0.491, +0.532] (n=500), vs random +0.451 [+0.433, +0.470] (n=500); fidelity- -0.008 [-0.016, +0.001] (n=500), vs random -0.446 [-0.465, -0.427] (n=500)
- builtin, k=10: fidelity+ +0.532 [+0.511, +0.552] (n=494), vs random +0.412 [+0.393, +0.430] (n=494); fidelity- -0.014 [-0.020, -0.007] (n=494), vs random -0.364 [-0.382, -0.348] (n=494)
- GNNExplainer, k=1: fidelity+ +0.167 [+0.151, +0.184] (n=500), vs random +0.155 [+0.139, +0.172] (n=500); fidelity- +0.295 [+0.273, +0.318] (n=500), vs random -0.236 [-0.258, -0.216] (n=500)
- GNNExplainer, k=2: fidelity+ +0.261 [+0.241, +0.282] (n=500), vs random +0.236 [+0.217, +0.257] (n=500); fidelity- +0.176 [+0.155, +0.197] (n=500), vs random -0.328 [-0.353, -0.305] (n=500)
- GNNExplainer, k=3: fidelity+ +0.325 [+0.306, +0.347] (n=500), vs random +0.289 [+0.269, +0.310] (n=500); fidelity- +0.111 [+0.092, +0.129] (n=500), vs random -0.370 [-0.394, -0.348] (n=500)
- GNNExplainer, k=5: fidelity+ +0.409 [+0.389, +0.431] (n=500), vs random +0.349 [+0.329, +0.371] (n=500); fidelity- +0.032 [+0.016, +0.049] (n=500), vs random -0.406 [-0.429, -0.383] (n=500)
- GNNExplainer, k=10: fidelity+ +0.486 [+0.467, +0.506] (n=494), vs random +0.366 [+0.349, +0.384] (n=494); fidelity- -0.022 [-0.033, -0.011] (n=494), vs random -0.373 [-0.391, -0.355] (n=494)
- GradExplainer, k=1: fidelity+ +0.261 [+0.242, +0.283] (n=500), vs random +0.249 [+0.229, +0.270] (n=500); fidelity- +0.197 [+0.177, +0.217] (n=500), vs random -0.334 [-0.355, -0.313] (n=500)
- GradExplainer, k=2: fidelity+ +0.378 [+0.358, +0.399] (n=500), vs random +0.353 [+0.335, +0.373] (n=500); fidelity- +0.099 [+0.084, +0.114] (n=500), vs random -0.406 [-0.425, -0.386] (n=500)
- GradExplainer, k=3: fidelity+ +0.438 [+0.417, +0.459] (n=500), vs random +0.402 [+0.382, +0.421] (n=500); fidelity- +0.059 [+0.047, +0.071] (n=500), vs random -0.422 [-0.442, -0.402] (n=500)
- GradExplainer, k=5: fidelity+ +0.486 [+0.465, +0.507] (n=500), vs random +0.426 [+0.406, +0.445] (n=500); fidelity- +0.051 [+0.042, +0.061] (n=500), vs random -0.387 [-0.407, -0.368] (n=500)
- GradExplainer, k=10: fidelity+ +0.505 [+0.484, +0.528] (n=494), vs random +0.385 [+0.366, +0.404] (n=494); fidelity- +0.041 [+0.034, +0.048] (n=494), vs random -0.309 [-0.327, -0.293] (n=494)
- IntegratedGradExplainer, k=1: fidelity+ +0.337 [+0.315, +0.359] (n=500), vs random +0.324 [+0.303, +0.346] (n=500); fidelity- +0.104 [+0.088, +0.122] (n=500), vs random -0.427 [-0.448, -0.405] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.427 [+0.407, +0.448] (n=500), vs random +0.402 [+0.383, +0.422] (n=500); fidelity- +0.049 [+0.036, +0.061] (n=500), vs random -0.456 [-0.476, -0.435] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.470 [+0.450, +0.491] (n=500), vs random +0.434 [+0.415, +0.453] (n=500); fidelity- +0.031 [+0.020, +0.042] (n=500), vs random -0.450 [-0.470, -0.430] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.498 [+0.477, +0.519] (n=500), vs random +0.438 [+0.419, +0.457] (n=500); fidelity- +0.038 [+0.028, +0.047] (n=500), vs random -0.400 [-0.420, -0.381] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.512 [+0.490, +0.534] (n=494), vs random +0.392 [+0.372, +0.410] (n=494); fidelity- +0.037 [+0.030, +0.045] (n=494), vs random -0.313 [-0.331, -0.296] (n=494)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top20/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.515 [+0.496, +0.535] (n=500)
- GNNExplainer: +0.427 [+0.406, +0.449] (n=500)
- GradExplainer: +0.483 [+0.462, +0.505] (n=500)
- IntegratedGradExplainer: +0.495 [+0.474, +0.516] (n=500)
- GNNExplainer - GradExplainer: -0.056 [-0.070, -0.043] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.068 [-0.082, -0.054] (n=500)
- GNNExplainer - builtin: -0.088 [-0.103, -0.074] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.011 [-0.017, -0.006] (n=500)
- GradExplainer - builtin: -0.032 [-0.039, -0.024] (n=500)
- IntegratedGradExplainer - builtin: -0.020 [-0.027, -0.014] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.002 [-0.009, +0.005] (n=500)
- GNNExplainer: +0.035 [+0.018, +0.053] (n=500)
- GradExplainer: +0.060 [+0.052, +0.069] (n=500)
- IntegratedGradExplainer: +0.051 [+0.042, +0.060] (n=500)
- GNNExplainer - GradExplainer: -0.025 [-0.040, -0.008] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.016 [-0.032, +0.002] (n=500)
- GNNExplainer - builtin: +0.037 [+0.021, +0.055] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.009 [+0.003, +0.015] (n=500)
- GradExplainer - builtin: +0.062 [+0.054, +0.071] (n=500)
- IntegratedGradExplainer - builtin: +0.053 [+0.046, +0.061] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.275 [+0.265, +0.285] (n=500)
- GNNExplainer: +0.222 [+0.212, +0.233] (n=500)
- GradExplainer: +0.252 [+0.241, +0.263] (n=500)
- IntegratedGradExplainer: +0.258 [+0.247, +0.268] (n=500)
- GNNExplainer - GradExplainer: -0.030 [-0.037, -0.024] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.036 [-0.043, -0.029] (n=500)
- GNNExplainer - builtin: -0.053 [-0.060, -0.046] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.006 [-0.008, -0.004] (n=500)
- GradExplainer - builtin: -0.023 [-0.026, -0.020] (n=500)
- IntegratedGradExplainer - builtin: -0.017 [-0.021, -0.014] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.218 [-0.227, -0.209] (n=500)
- GNNExplainer: -0.197 [-0.207, -0.188] (n=500)
- GradExplainer: -0.188 [-0.197, -0.179] (n=500)
- IntegratedGradExplainer: -0.190 [-0.200, -0.181] (n=500)
- GNNExplainer - GradExplainer: -0.010 [-0.016, -0.003] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.007 [-0.014, +0.000] (n=500)
- GNNExplainer - builtin: +0.020 [+0.013, +0.028] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.003 [+0.001, +0.005] (n=500)
- GradExplainer - builtin: +0.030 [+0.026, +0.034] (n=500)
- IntegratedGradExplainer - builtin: +0.027 [+0.024, +0.032] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.731 [+0.719, +0.743] (n=500)
- GNNExplainer: +0.102 [+0.101, +0.103] (n=500)
- GradExplainer: +0.381 [+0.372, +0.389] (n=500)
- IntegratedGradExplainer: +0.320 [+0.314, +0.326] (n=500)
- GNNExplainer - GradExplainer: -0.279 [-0.287, -0.270] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.218 [-0.224, -0.212] (n=500)
- GNNExplainer - builtin: -0.629 [-0.641, -0.617] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.061 [+0.055, +0.066] (n=500)
- GradExplainer - builtin: -0.351 [-0.363, -0.339] (n=500)
- IntegratedGradExplainer - builtin: -0.411 [-0.422, -0.401] (n=500)

