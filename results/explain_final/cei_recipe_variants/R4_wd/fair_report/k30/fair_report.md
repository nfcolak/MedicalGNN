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

- builtin, k=1: fidelity+ +0.376 [+0.356, +0.397] (n=500), vs random +0.364 [+0.344, +0.383] (n=500); fidelity- +0.036 [+0.021, +0.052] (n=500), vs random -0.488 [-0.509, -0.466] (n=500)
- builtin, k=2: fidelity+ +0.466 [+0.445, +0.485] (n=500), vs random +0.441 [+0.421, +0.459] (n=500); fidelity- -0.016 [-0.027, -0.004] (n=500), vs random -0.514 [-0.533, -0.492] (n=500)
- builtin, k=3: fidelity+ +0.499 [+0.477, +0.519] (n=500), vs random +0.462 [+0.442, +0.481] (n=500); fidelity- -0.023 [-0.032, -0.011] (n=500), vs random -0.498 [-0.518, -0.478] (n=500)
- builtin, k=5: fidelity+ +0.518 [+0.496, +0.539] (n=500), vs random +0.457 [+0.437, +0.476] (n=500); fidelity- -0.030 [-0.039, -0.021] (n=500), vs random -0.464 [-0.482, -0.446] (n=500)
- builtin, k=10: fidelity+ +0.535 [+0.514, +0.556] (n=492), vs random +0.413 [+0.395, +0.431] (n=492); fidelity- -0.033 [-0.040, -0.026] (n=492), vs random -0.381 [-0.398, -0.364] (n=492)
- GNNExplainer, k=1: fidelity+ +0.194 [+0.177, +0.212] (n=500), vs random +0.182 [+0.165, +0.200] (n=500); fidelity- +0.261 [+0.239, +0.284] (n=500), vs random -0.262 [-0.285, -0.240] (n=500)
- GNNExplainer, k=2: fidelity+ +0.296 [+0.276, +0.315] (n=500), vs random +0.270 [+0.251, +0.290] (n=500); fidelity- +0.143 [+0.123, +0.162] (n=500), vs random -0.355 [-0.378, -0.332] (n=500)
- GNNExplainer, k=3: fidelity+ +0.357 [+0.336, +0.377] (n=500), vs random +0.320 [+0.300, +0.341] (n=500); fidelity- +0.077 [+0.060, +0.095] (n=500), vs random -0.398 [-0.422, -0.374] (n=500)
- GNNExplainer, k=5: fidelity+ +0.420 [+0.400, +0.441] (n=500), vs random +0.360 [+0.340, +0.380] (n=500); fidelity- +0.016 [+0.002, +0.030] (n=500), vs random -0.417 [-0.439, -0.395] (n=500)
- GNNExplainer, k=10: fidelity+ +0.483 [+0.464, +0.503] (n=492), vs random +0.361 [+0.343, +0.378] (n=492); fidelity- -0.032 [-0.041, -0.023] (n=492), vs random -0.380 [-0.399, -0.362] (n=492)
- GradExplainer, k=1: fidelity+ +0.283 [+0.261, +0.303] (n=500), vs random +0.271 [+0.249, +0.290] (n=500); fidelity- +0.144 [+0.127, +0.162] (n=500), vs random -0.379 [-0.400, -0.355] (n=500)
- GradExplainer, k=2: fidelity+ +0.410 [+0.390, +0.430] (n=500), vs random +0.384 [+0.365, +0.403] (n=500); fidelity- +0.047 [+0.034, +0.061] (n=500), vs random -0.451 [-0.472, -0.429] (n=500)
- GradExplainer, k=3: fidelity+ +0.467 [+0.446, +0.487] (n=500), vs random +0.431 [+0.411, +0.448] (n=500); fidelity- +0.026 [+0.016, +0.037] (n=500), vs random -0.450 [-0.470, -0.429] (n=500)
- GradExplainer, k=5: fidelity+ +0.498 [+0.476, +0.519] (n=500), vs random +0.438 [+0.418, +0.457] (n=500); fidelity- +0.022 [+0.013, +0.032] (n=500), vs random -0.411 [-0.430, -0.391] (n=500)
- GradExplainer, k=10: fidelity+ +0.520 [+0.499, +0.540] (n=492), vs random +0.398 [+0.379, +0.416] (n=492); fidelity- +0.018 [+0.011, +0.025] (n=492), vs random -0.331 [-0.349, -0.312] (n=492)
- IntegratedGradExplainer, k=1: fidelity+ +0.350 [+0.328, +0.371] (n=500), vs random +0.337 [+0.316, +0.359] (n=500); fidelity- +0.064 [+0.048, +0.080] (n=500), vs random -0.459 [-0.482, -0.436] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.447 [+0.427, +0.467] (n=500), vs random +0.422 [+0.403, +0.441] (n=500); fidelity- +0.020 [+0.009, +0.032] (n=500), vs random -0.478 [-0.498, -0.456] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.487 [+0.465, +0.507] (n=500), vs random +0.450 [+0.430, +0.469] (n=500); fidelity- +0.010 [+0.000, +0.020] (n=500), vs random -0.466 [-0.486, -0.445] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.508 [+0.486, +0.528] (n=500), vs random +0.447 [+0.427, +0.466] (n=500); fidelity- +0.010 [+0.001, +0.021] (n=500), vs random -0.423 [-0.442, -0.403] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.521 [+0.500, +0.542] (n=492), vs random +0.399 [+0.380, +0.416] (n=492); fidelity- +0.019 [+0.012, +0.027] (n=492), vs random -0.329 [-0.347, -0.311] (n=492)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top30/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.522 [+0.501, +0.543] (n=500)
- GNNExplainer: +0.426 [+0.405, +0.447] (n=500)
- GradExplainer: +0.503 [+0.482, +0.524] (n=500)
- IntegratedGradExplainer: +0.511 [+0.490, +0.532] (n=500)
- GNNExplainer - GradExplainer: -0.077 [-0.092, -0.062] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.085 [-0.101, -0.070] (n=500)
- GNNExplainer - builtin: -0.096 [-0.112, -0.081] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.008 [-0.013, -0.004] (n=500)
- GradExplainer - builtin: -0.019 [-0.024, -0.014] (n=500)
- IntegratedGradExplainer - builtin: -0.011 [-0.015, -0.007] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.020 [-0.028, -0.012] (n=500)
- GNNExplainer: +0.024 [+0.007, +0.041] (n=500)
- GradExplainer: +0.032 [+0.024, +0.041] (n=500)
- IntegratedGradExplainer: +0.025 [+0.016, +0.033] (n=500)
- GNNExplainer - GradExplainer: -0.007 [-0.025, +0.009] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.000 [-0.017, +0.017] (n=500)
- GNNExplainer - builtin: +0.045 [+0.029, +0.061] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.007 [+0.003, +0.012] (n=500)
- GradExplainer - builtin: +0.052 [+0.045, +0.059] (n=500)
- IntegratedGradExplainer - builtin: +0.045 [+0.038, +0.052] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.277 [+0.266, +0.288] (n=500)
- GNNExplainer: +0.220 [+0.210, +0.230] (n=500)
- GradExplainer: +0.260 [+0.249, +0.270] (n=500)
- IntegratedGradExplainer: +0.263 [+0.252, +0.274] (n=500)
- GNNExplainer - GradExplainer: -0.040 [-0.047, -0.032] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.043 [-0.050, -0.035] (n=500)
- GNNExplainer - builtin: -0.057 [-0.064, -0.049] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.003 [-0.005, -0.002] (n=500)
- GradExplainer - builtin: -0.018 [-0.020, -0.015] (n=500)
- IntegratedGradExplainer - builtin: -0.014 [-0.017, -0.012] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.228 [-0.237, -0.219] (n=500)
- GNNExplainer: -0.204 [-0.213, -0.194] (n=500)
- GradExplainer: -0.197 [-0.206, -0.187] (n=500)
- IntegratedGradExplainer: -0.199 [-0.208, -0.189] (n=500)
- GNNExplainer - GradExplainer: -0.007 [-0.014, -0.000] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.005 [-0.012, +0.002] (n=500)
- GNNExplainer - builtin: +0.024 [+0.017, +0.031] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.002 [+0.000, +0.004] (n=500)
- GradExplainer - builtin: +0.031 [+0.028, +0.035] (n=500)
- IntegratedGradExplainer - builtin: +0.029 [+0.026, +0.033] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.712 [+0.698, +0.725] (n=500)
- GNNExplainer: +0.101 [+0.100, +0.102] (n=500)
- GradExplainer: +0.380 [+0.371, +0.388] (n=500)
- IntegratedGradExplainer: +0.340 [+0.334, +0.346] (n=500)
- GNNExplainer - GradExplainer: -0.278 [-0.286, -0.270] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.239 [-0.245, -0.233] (n=500)
- GNNExplainer - builtin: -0.611 [-0.623, -0.598] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.040 [+0.035, +0.045] (n=500)
- GradExplainer - builtin: -0.332 [-0.344, -0.320] (n=500)
- IntegratedGradExplainer - builtin: -0.372 [-0.383, -0.361] (n=500)

