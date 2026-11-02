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

- builtin, k=1: fidelity+ +0.339 [+0.319, +0.361] (n=500), vs random +0.327 [+0.307, +0.348] (n=500); fidelity- +0.055 [+0.041, +0.068] (n=500), vs random -0.467 [-0.488, -0.448] (n=500)
- builtin, k=2: fidelity+ +0.422 [+0.401, +0.443] (n=500), vs random +0.397 [+0.378, +0.417] (n=500); fidelity- +0.018 [+0.007, +0.029] (n=500), vs random -0.476 [-0.496, -0.457] (n=500)
- builtin, k=3: fidelity+ +0.454 [+0.435, +0.474] (n=500), vs random +0.419 [+0.400, +0.437] (n=500); fidelity- +0.009 [-0.001, +0.019] (n=500), vs random -0.460 [-0.479, -0.440] (n=500)
- builtin, k=5: fidelity+ +0.478 [+0.458, +0.498] (n=500), vs random +0.417 [+0.398, +0.435] (n=500); fidelity- -0.001 [-0.009, +0.008] (n=500), vs random -0.423 [-0.441, -0.406] (n=500)
- builtin, k=10: fidelity+ +0.500 [+0.479, +0.520] (n=493), vs random +0.375 [+0.356, +0.392] (n=493); fidelity- -0.001 [-0.008, +0.006] (n=493), vs random -0.335 [-0.352, -0.317] (n=493)
- GNNExplainer, k=1: fidelity+ +0.170 [+0.155, +0.186] (n=500), vs random +0.157 [+0.143, +0.173] (n=500); fidelity- +0.276 [+0.256, +0.298] (n=500), vs random -0.246 [-0.265, -0.226] (n=500)
- GNNExplainer, k=2: fidelity+ +0.260 [+0.243, +0.279] (n=500), vs random +0.236 [+0.219, +0.254] (n=500); fidelity- +0.161 [+0.143, +0.179] (n=500), vs random -0.333 [-0.354, -0.313] (n=500)
- GNNExplainer, k=3: fidelity+ +0.311 [+0.293, +0.329] (n=500), vs random +0.275 [+0.257, +0.293] (n=500); fidelity- +0.103 [+0.086, +0.120] (n=500), vs random -0.365 [-0.386, -0.345] (n=500)
- GNNExplainer, k=5: fidelity+ +0.385 [+0.367, +0.404] (n=500), vs random +0.324 [+0.306, +0.342] (n=500); fidelity- +0.038 [+0.024, +0.053] (n=500), vs random -0.384 [-0.403, -0.365] (n=500)
- GNNExplainer, k=10: fidelity+ +0.454 [+0.435, +0.473] (n=493), vs random +0.329 [+0.311, +0.346] (n=493); fidelity- -0.006 [-0.016, +0.005] (n=493), vs random -0.340 [-0.357, -0.323] (n=493)
- GradExplainer, k=1: fidelity+ +0.248 [+0.230, +0.268] (n=500), vs random +0.236 [+0.218, +0.256] (n=500); fidelity- +0.178 [+0.160, +0.195] (n=500), vs random -0.344 [-0.365, -0.326] (n=500)
- GradExplainer, k=2: fidelity+ +0.361 [+0.341, +0.382] (n=500), vs random +0.336 [+0.317, +0.357] (n=500); fidelity- +0.074 [+0.060, +0.087] (n=500), vs random -0.420 [-0.439, -0.401] (n=500)
- GradExplainer, k=3: fidelity+ +0.418 [+0.399, +0.439] (n=500), vs random +0.383 [+0.364, +0.402] (n=500); fidelity- +0.042 [+0.031, +0.053] (n=500), vs random -0.427 [-0.447, -0.409] (n=500)
- GradExplainer, k=5: fidelity+ +0.464 [+0.444, +0.485] (n=500), vs random +0.403 [+0.384, +0.422] (n=500); fidelity- +0.027 [+0.019, +0.036] (n=500), vs random -0.396 [-0.416, -0.377] (n=500)
- GradExplainer, k=10: fidelity+ +0.492 [+0.471, +0.513] (n=493), vs random +0.367 [+0.348, +0.384] (n=493); fidelity- +0.016 [+0.010, +0.021] (n=493), vs random -0.318 [-0.335, -0.300] (n=493)
- IntegratedGradExplainer, k=1: fidelity+ +0.334 [+0.313, +0.357] (n=500), vs random +0.322 [+0.301, +0.344] (n=500); fidelity- +0.070 [+0.055, +0.085] (n=500), vs random -0.452 [-0.473, -0.431] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.413 [+0.392, +0.434] (n=500), vs random +0.388 [+0.368, +0.409] (n=500); fidelity- +0.030 [+0.019, +0.042] (n=500), vs random -0.463 [-0.483, -0.443] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.450 [+0.430, +0.470] (n=500), vs random +0.414 [+0.395, +0.434] (n=500); fidelity- +0.016 [+0.007, +0.025] (n=500), vs random -0.453 [-0.472, -0.433] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.478 [+0.458, +0.498] (n=500), vs random +0.417 [+0.398, +0.435] (n=500); fidelity- +0.012 [+0.005, +0.020] (n=500), vs random -0.411 [-0.429, -0.393] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.501 [+0.480, +0.521] (n=493), vs random +0.376 [+0.357, +0.393] (n=493); fidelity- +0.007 [+0.001, +0.012] (n=493), vs random -0.327 [-0.343, -0.310] (n=493)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top10/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.484 [+0.464, +0.504] (n=500)
- GNNExplainer: +0.395 [+0.375, +0.414] (n=500)
- GradExplainer: +0.466 [+0.446, +0.487] (n=500)
- IntegratedGradExplainer: +0.487 [+0.467, +0.508] (n=500)
- GNNExplainer - GradExplainer: -0.072 [-0.086, -0.059] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.093 [-0.109, -0.079] (n=500)
- GNNExplainer - builtin: -0.090 [-0.104, -0.075] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.021 [-0.029, -0.014] (n=500)
- GradExplainer - builtin: -0.018 [-0.026, -0.010] (n=500)
- IntegratedGradExplainer - builtin: +0.003 [-0.004, +0.011] (n=500)

**fidelity-** (lower = more faithful)

- builtin: +0.003 [-0.004, +0.011] (n=500)
- GNNExplainer: +0.046 [+0.030, +0.063] (n=500)
- GradExplainer: +0.030 [+0.022, +0.037] (n=500)
- IntegratedGradExplainer: +0.012 [+0.006, +0.019] (n=500)
- GNNExplainer - GradExplainer: +0.016 [+0.002, +0.032] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.034 [+0.019, +0.050] (n=500)
- GNNExplainer - builtin: +0.043 [+0.028, +0.058] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.017 [+0.013, +0.022] (n=500)
- GradExplainer - builtin: +0.026 [+0.020, +0.033] (n=500)
- IntegratedGradExplainer - builtin: +0.009 [+0.002, +0.015] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.259 [+0.248, +0.270] (n=500)
- GNNExplainer: +0.205 [+0.195, +0.215] (n=500)
- GradExplainer: +0.245 [+0.234, +0.256] (n=500)
- IntegratedGradExplainer: +0.255 [+0.244, +0.266] (n=500)
- GNNExplainer - GradExplainer: -0.040 [-0.048, -0.033] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.050 [-0.058, -0.042] (n=500)
- GNNExplainer - builtin: -0.054 [-0.062, -0.045] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.009 [-0.013, -0.006] (n=500)
- GradExplainer - builtin: -0.014 [-0.019, -0.009] (n=500)
- IntegratedGradExplainer - builtin: -0.004 [-0.009, +0.001] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.202 [-0.211, -0.194] (n=500)
- GNNExplainer: -0.185 [-0.193, -0.177] (n=500)
- GradExplainer: -0.193 [-0.202, -0.185] (n=500)
- IntegratedGradExplainer: -0.199 [-0.207, -0.190] (n=500)
- GNNExplainer - GradExplainer: +0.008 [+0.002, +0.015] (n=500)
- GNNExplainer - IntegratedGradExplainer: +0.014 [+0.007, +0.021] (n=500)
- GNNExplainer - builtin: +0.017 [+0.010, +0.025] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.006 [+0.004, +0.008] (n=500)
- GradExplainer - builtin: +0.009 [+0.006, +0.013] (n=500)
- IntegratedGradExplainer - builtin: +0.004 [-0.000, +0.007] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.706 [+0.692, +0.721] (n=500)
- GNNExplainer: +0.101 [+0.100, +0.102] (n=500)
- GradExplainer: +0.411 [+0.403, +0.420] (n=500)
- IntegratedGradExplainer: +0.372 [+0.365, +0.379] (n=500)
- GNNExplainer - GradExplainer: -0.310 [-0.319, -0.302] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.271 [-0.278, -0.264] (n=500)
- GNNExplainer - builtin: -0.605 [-0.619, -0.590] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.039 [+0.034, +0.045] (n=500)
- GradExplainer - builtin: -0.295 [-0.309, -0.281] (n=500)
- IntegratedGradExplainer - builtin: -0.334 [-0.347, -0.321] (n=500)

