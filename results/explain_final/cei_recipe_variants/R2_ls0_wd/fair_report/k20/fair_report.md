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

- builtin, k=1: fidelity+ +0.392 [+0.369, +0.414] (n=500), vs random +0.379 [+0.358, +0.400] (n=500); fidelity- +0.031 [+0.014, +0.048] (n=500), vs random -0.574 [-0.598, -0.549] (n=500)
- builtin, k=2: fidelity+ +0.499 [+0.478, +0.520] (n=500), vs random +0.473 [+0.453, +0.493] (n=500); fidelity- -0.003 [-0.016, +0.011] (n=500), vs random -0.575 [-0.598, -0.553] (n=500)
- builtin, k=3: fidelity+ +0.539 [+0.518, +0.559] (n=500), vs random +0.501 [+0.480, +0.521] (n=500); fidelity- -0.007 [-0.017, +0.005] (n=500), vs random -0.550 [-0.571, -0.528] (n=500)
- builtin, k=5: fidelity+ +0.574 [+0.553, +0.595] (n=500), vs random +0.510 [+0.491, +0.530] (n=500); fidelity- -0.009 [-0.019, +0.002] (n=500), vs random -0.500 [-0.521, -0.480] (n=500)
- builtin, k=10: fidelity+ +0.607 [+0.586, +0.629] (n=494), vs random +0.478 [+0.459, +0.498] (n=494); fidelity- -0.014 [-0.022, -0.005] (n=494), vs random -0.403 [-0.422, -0.385] (n=494)
- GNNExplainer, k=1: fidelity+ +0.258 [+0.235, +0.279] (n=500), vs random +0.245 [+0.222, +0.266] (n=500); fidelity- +0.216 [+0.192, +0.241] (n=500), vs random -0.389 [-0.416, -0.362] (n=500)
- GNNExplainer, k=2: fidelity+ +0.361 [+0.337, +0.385] (n=500), vs random +0.335 [+0.312, +0.358] (n=500); fidelity- +0.108 [+0.087, +0.131] (n=500), vs random -0.465 [-0.491, -0.438] (n=500)
- GNNExplainer, k=3: fidelity+ +0.407 [+0.385, +0.431] (n=500), vs random +0.369 [+0.347, +0.393] (n=500); fidelity- +0.065 [+0.046, +0.084] (n=500), vs random -0.478 [-0.504, -0.454] (n=500)
- GNNExplainer, k=5: fidelity+ +0.465 [+0.443, +0.488] (n=500), vs random +0.401 [+0.379, +0.422] (n=500); fidelity- +0.018 [+0.002, +0.034] (n=500), vs random -0.473 [-0.497, -0.451] (n=500)
- GNNExplainer, k=10: fidelity+ +0.510 [+0.491, +0.533] (n=494), vs random +0.382 [+0.364, +0.402] (n=494); fidelity- -0.010 [-0.023, +0.003] (n=494), vs random -0.399 [-0.420, -0.379] (n=494)
- GradExplainer, k=1: fidelity+ +0.294 [+0.272, +0.317] (n=500), vs random +0.281 [+0.260, +0.304] (n=500); fidelity- +0.173 [+0.152, +0.195] (n=500), vs random -0.431 [-0.456, -0.406] (n=500)
- GradExplainer, k=2: fidelity+ +0.421 [+0.397, +0.445] (n=500), vs random +0.395 [+0.372, +0.418] (n=500); fidelity- +0.073 [+0.057, +0.089] (n=500), vs random -0.500 [-0.524, -0.476] (n=500)
- GradExplainer, k=3: fidelity+ +0.480 [+0.457, +0.505] (n=500), vs random +0.443 [+0.420, +0.465] (n=500); fidelity- +0.055 [+0.042, +0.068] (n=500), vs random -0.489 [-0.511, -0.465] (n=500)
- GradExplainer, k=5: fidelity+ +0.524 [+0.501, +0.547] (n=500), vs random +0.461 [+0.439, +0.482] (n=500); fidelity- +0.054 [+0.042, +0.066] (n=500), vs random -0.438 [-0.459, -0.416] (n=500)
- GradExplainer, k=10: fidelity+ +0.560 [+0.537, +0.583] (n=494), vs random +0.432 [+0.411, +0.452] (n=494); fidelity- +0.042 [+0.032, +0.051] (n=494), vs random -0.347 [-0.368, -0.328] (n=494)
- IntegratedGradExplainer, k=1: fidelity+ +0.362 [+0.338, +0.386] (n=500), vs random +0.350 [+0.326, +0.373] (n=500); fidelity- +0.079 [+0.061, +0.099] (n=500), vs random -0.526 [-0.551, -0.498] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.467 [+0.444, +0.491] (n=500), vs random +0.441 [+0.419, +0.463] (n=500); fidelity- +0.037 [+0.023, +0.052] (n=500), vs random -0.536 [-0.559, -0.511] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.506 [+0.483, +0.529] (n=500), vs random +0.468 [+0.446, +0.490] (n=500); fidelity- +0.036 [+0.023, +0.050] (n=500), vs random -0.507 [-0.530, -0.484] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.543 [+0.521, +0.566] (n=500), vs random +0.480 [+0.458, +0.500] (n=500); fidelity- +0.044 [+0.032, +0.056] (n=500), vs random -0.448 [-0.470, -0.426] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.563 [+0.541, +0.587] (n=494), vs random +0.435 [+0.414, +0.456] (n=494); fidelity- +0.044 [+0.034, +0.055] (n=494), vs random -0.345 [-0.365, -0.325] (n=494)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top20/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.584 [+0.563, +0.604] (n=500)
- GNNExplainer: +0.465 [+0.442, +0.487] (n=500)
- GradExplainer: +0.539 [+0.517, +0.562] (n=500)
- IntegratedGradExplainer: +0.554 [+0.532, +0.576] (n=500)
- GNNExplainer - GradExplainer: -0.075 [-0.093, -0.056] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.089 [-0.108, -0.070] (n=500)
- GNNExplainer - builtin: -0.119 [-0.136, -0.102] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.014 [-0.020, -0.009] (n=500)
- GradExplainer - builtin: -0.044 [-0.053, -0.036] (n=500)
- IntegratedGradExplainer - builtin: -0.030 [-0.037, -0.022] (n=500)

**fidelity-** (lower = more faithful)

- builtin: -0.007 [-0.017, +0.003] (n=500)
- GNNExplainer: +0.036 [+0.017, +0.054] (n=500)
- GradExplainer: +0.054 [+0.043, +0.066] (n=500)
- IntegratedGradExplainer: +0.045 [+0.034, +0.057] (n=500)
- GNNExplainer - GradExplainer: -0.018 [-0.038, +0.000] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.009 [-0.028, +0.009] (n=500)
- GNNExplainer - builtin: +0.044 [+0.026, +0.062] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.009 [+0.004, +0.015] (n=500)
- GradExplainer - builtin: +0.062 [+0.052, +0.072] (n=500)
- IntegratedGradExplainer - builtin: +0.053 [+0.043, +0.063] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.326 [+0.314, +0.336] (n=500)
- GNNExplainer: +0.239 [+0.228, +0.251] (n=500)
- GradExplainer: +0.283 [+0.271, +0.294] (n=500)
- IntegratedGradExplainer: +0.288 [+0.275, +0.300] (n=500)
- GNNExplainer - GradExplainer: -0.043 [-0.054, -0.033] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.049 [-0.059, -0.038] (n=500)
- GNNExplainer - builtin: -0.086 [-0.097, -0.076] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.005 [-0.008, -0.002] (n=500)
- GradExplainer - builtin: -0.043 [-0.048, -0.038] (n=500)
- IntegratedGradExplainer - builtin: -0.038 [-0.043, -0.032] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.241 [-0.251, -0.232] (n=500)
- GNNExplainer: -0.220 [-0.231, -0.210] (n=500)
- GradExplainer: -0.207 [-0.218, -0.197] (n=500)
- IntegratedGradExplainer: -0.210 [-0.220, -0.199] (n=500)
- GNNExplainer - GradExplainer: -0.013 [-0.021, -0.005] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.011 [-0.019, -0.002] (n=500)
- GNNExplainer - builtin: +0.021 [+0.013, +0.029] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.002 [-0.001, +0.005] (n=500)
- GradExplainer - builtin: +0.034 [+0.029, +0.039] (n=500)
- IntegratedGradExplainer - builtin: +0.032 [+0.027, +0.037] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.697 [+0.684, +0.709] (n=500)
- GNNExplainer: +0.101 [+0.100, +0.102] (n=500)
- GradExplainer: +0.389 [+0.381, +0.397] (n=500)
- IntegratedGradExplainer: +0.320 [+0.314, +0.326] (n=500)
- GNNExplainer - GradExplainer: -0.288 [-0.296, -0.281] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.219 [-0.225, -0.214] (n=500)
- GNNExplainer - builtin: -0.596 [-0.607, -0.584] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.069 [+0.064, +0.074] (n=500)
- GradExplainer - builtin: -0.308 [-0.319, -0.296] (n=500)
- IntegratedGradExplainer - builtin: -0.377 [-0.387, -0.366] (n=500)

