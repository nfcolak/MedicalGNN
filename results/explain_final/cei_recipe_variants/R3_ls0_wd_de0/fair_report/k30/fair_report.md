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

- builtin, k=1: fidelity+ +0.401 [+0.377, +0.423] (n=500), vs random +0.387 [+0.364, +0.409] (n=500); fidelity- +0.065 [+0.048, +0.084] (n=500), vs random -0.530 [-0.554, -0.505] (n=500)
- builtin, k=2: fidelity+ +0.512 [+0.489, +0.535] (n=500), vs random +0.485 [+0.462, +0.506] (n=500); fidelity- +0.017 [+0.004, +0.032] (n=500), vs random -0.547 [-0.568, -0.525] (n=500)
- builtin, k=3: fidelity+ +0.554 [+0.531, +0.576] (n=500), vs random +0.515 [+0.492, +0.535] (n=500); fidelity- +0.011 [-0.001, +0.024] (n=500), vs random -0.528 [-0.548, -0.507] (n=500)
- builtin, k=5: fidelity+ +0.585 [+0.562, +0.607] (n=500), vs random +0.519 [+0.497, +0.538] (n=500); fidelity- -0.003 [-0.014, +0.009] (n=500), vs random -0.493 [-0.512, -0.473] (n=500)
- builtin, k=10: fidelity+ +0.607 [+0.586, +0.629] (n=492), vs random +0.471 [+0.450, +0.491] (n=492); fidelity- -0.011 [-0.020, -0.001] (n=492), vs random -0.404 [-0.423, -0.385] (n=492)
- GNNExplainer, k=1: fidelity+ +0.254 [+0.232, +0.276] (n=500), vs random +0.240 [+0.218, +0.262] (n=500); fidelity- +0.252 [+0.228, +0.275] (n=500), vs random -0.343 [-0.370, -0.315] (n=500)
- GNNExplainer, k=2: fidelity+ +0.356 [+0.332, +0.380] (n=500), vs random +0.328 [+0.305, +0.352] (n=500); fidelity- +0.143 [+0.123, +0.164] (n=500), vs random -0.421 [-0.447, -0.394] (n=500)
- GNNExplainer, k=3: fidelity+ +0.412 [+0.388, +0.436] (n=500), vs random +0.372 [+0.349, +0.395] (n=500); fidelity- +0.084 [+0.066, +0.102] (n=500), vs random -0.455 [-0.480, -0.429] (n=500)
- GNNExplainer, k=5: fidelity+ +0.467 [+0.444, +0.490] (n=500), vs random +0.401 [+0.379, +0.422] (n=500); fidelity- +0.036 [+0.021, +0.052] (n=500), vs random -0.454 [-0.477, -0.430] (n=500)
- GNNExplainer, k=10: fidelity+ +0.523 [+0.502, +0.544] (n=492), vs random +0.387 [+0.367, +0.407] (n=492); fidelity- -0.014 [-0.026, -0.002] (n=492), vs random -0.407 [-0.427, -0.386] (n=492)
- GradExplainer, k=1: fidelity+ +0.297 [+0.273, +0.320] (n=500), vs random +0.283 [+0.260, +0.307] (n=500); fidelity- +0.197 [+0.176, +0.221] (n=500), vs random -0.398 [-0.422, -0.371] (n=500)
- GradExplainer, k=2: fidelity+ +0.434 [+0.410, +0.458] (n=500), vs random +0.407 [+0.383, +0.429] (n=500); fidelity- +0.087 [+0.072, +0.104] (n=500), vs random -0.478 [-0.502, -0.454] (n=500)
- GradExplainer, k=3: fidelity+ +0.501 [+0.477, +0.523] (n=500), vs random +0.461 [+0.438, +0.482] (n=500); fidelity- +0.062 [+0.050, +0.076] (n=500), vs random -0.477 [-0.499, -0.454] (n=500)
- GradExplainer, k=5: fidelity+ +0.543 [+0.520, +0.565] (n=500), vs random +0.476 [+0.455, +0.497] (n=500); fidelity- +0.055 [+0.044, +0.067] (n=500), vs random -0.435 [-0.456, -0.413] (n=500)
- GradExplainer, k=10: fidelity+ +0.572 [+0.550, +0.593] (n=492), vs random +0.436 [+0.415, +0.456] (n=492); fidelity- +0.046 [+0.035, +0.057] (n=492), vs random -0.347 [-0.368, -0.326] (n=492)
- IntegratedGradExplainer, k=1: fidelity+ +0.367 [+0.342, +0.391] (n=500), vs random +0.353 [+0.328, +0.376] (n=500); fidelity- +0.111 [+0.094, +0.130] (n=500), vs random -0.484 [-0.509, -0.458] (n=500)
- IntegratedGradExplainer, k=2: fidelity+ +0.483 [+0.460, +0.505] (n=500), vs random +0.455 [+0.434, +0.476] (n=500); fidelity- +0.051 [+0.037, +0.066] (n=500), vs random -0.514 [-0.536, -0.491] (n=500)
- IntegratedGradExplainer, k=3: fidelity+ +0.531 [+0.508, +0.553] (n=500), vs random +0.491 [+0.470, +0.512] (n=500); fidelity- +0.041 [+0.029, +0.054] (n=500), vs random -0.498 [-0.519, -0.477] (n=500)
- IntegratedGradExplainer, k=5: fidelity+ +0.559 [+0.536, +0.580] (n=500), vs random +0.492 [+0.470, +0.512] (n=500); fidelity- +0.042 [+0.030, +0.056] (n=500), vs random -0.447 [-0.469, -0.425] (n=500)
- IntegratedGradExplainer, k=10: fidelity+ +0.574 [+0.552, +0.596] (n=492), vs random +0.438 [+0.417, +0.458] (n=492); fidelity- +0.045 [+0.034, +0.056] (n=492), vs random -0.348 [-0.369, -0.328] (n=492)

## 3. ProtGNN: does the subgraph carry the evidence the prediction used?

## 4. Explainers compared on the same patients

Individual interval of each explainer, then every pair of explainers of one method as a paired difference (A - B, over the patients both have). Intervals are withheld below 10 patients. With few patients none of this is a ranking.

### top30/cei_mp2

**fidelity+** (higher = more faithful)

- builtin: +0.592 [+0.569, +0.613] (n=500)
- GNNExplainer: +0.472 [+0.449, +0.495] (n=500)
- GradExplainer: +0.556 [+0.533, +0.578] (n=500)
- IntegratedGradExplainer: +0.562 [+0.540, +0.584] (n=500)
- GNNExplainer - GradExplainer: -0.084 [-0.101, -0.066] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.090 [-0.108, -0.072] (n=500)
- GNNExplainer - builtin: -0.119 [-0.136, -0.102] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.006 [-0.011, -0.002] (n=500)
- GradExplainer - builtin: -0.036 [-0.042, -0.029] (n=500)
- IntegratedGradExplainer - builtin: -0.029 [-0.035, -0.023] (n=500)

**fidelity-** (lower = more faithful)

- builtin: +0.001 [-0.010, +0.011] (n=500)
- GNNExplainer: +0.038 [+0.021, +0.056] (n=500)
- GradExplainer: +0.055 [+0.043, +0.067] (n=500)
- IntegratedGradExplainer: +0.052 [+0.040, +0.065] (n=500)
- GNNExplainer - GradExplainer: -0.016 [-0.037, +0.003] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.014 [-0.034, +0.006] (n=500)
- GNNExplainer - builtin: +0.038 [+0.019, +0.057] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.003 [-0.003, +0.007] (n=500)
- GradExplainer - builtin: +0.054 [+0.046, +0.063] (n=500)
- IntegratedGradExplainer - builtin: +0.052 [+0.043, +0.060] (n=500)

**deletion vs random** (width-matched difference to random deletion; higher is better)

- builtin: +0.319 [+0.307, +0.330] (n=500)
- GNNExplainer: +0.238 [+0.226, +0.249] (n=500)
- GradExplainer: +0.284 [+0.271, +0.296] (n=500)
- IntegratedGradExplainer: +0.287 [+0.274, +0.299] (n=500)
- GNNExplainer - GradExplainer: -0.046 [-0.056, -0.036] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.049 [-0.060, -0.038] (n=500)
- GNNExplainer - builtin: -0.081 [-0.091, -0.071] (n=500)
- GradExplainer - IntegratedGradExplainer: -0.003 [-0.006, -0.001] (n=500)
- GradExplainer - builtin: -0.036 [-0.041, -0.031] (n=500)
- IntegratedGradExplainer - builtin: -0.032 [-0.038, -0.027] (n=500)

**insertion vs random** (width-matched difference to random insertion; lower is better)

- builtin: -0.244 [-0.253, -0.235] (n=500)
- GNNExplainer: -0.222 [-0.233, -0.211] (n=500)
- GradExplainer: -0.208 [-0.218, -0.198] (n=500)
- IntegratedGradExplainer: -0.209 [-0.220, -0.199] (n=500)
- GNNExplainer - GradExplainer: -0.014 [-0.023, -0.005] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.013 [-0.022, -0.004] (n=500)
- GNNExplainer - builtin: +0.021 [+0.013, +0.030] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.001 [-0.001, +0.004] (n=500)
- GradExplainer - builtin: +0.035 [+0.030, +0.041] (n=500)
- IntegratedGradExplainer - builtin: +0.034 [+0.029, +0.040] (n=500)

**sparsity** (share of eligible nodes not needed for 90% of the mass)

- builtin: +0.684 [+0.671, +0.696] (n=500)
- GNNExplainer: +0.101 [+0.100, +0.102] (n=500)
- GradExplainer: +0.395 [+0.387, +0.404] (n=500)
- IntegratedGradExplainer: +0.344 [+0.338, +0.350] (n=500)
- GNNExplainer - GradExplainer: -0.294 [-0.302, -0.286] (n=500)
- GNNExplainer - IntegratedGradExplainer: -0.242 [-0.248, -0.237] (n=500)
- GNNExplainer - builtin: -0.582 [-0.596, -0.570] (n=500)
- GradExplainer - IntegratedGradExplainer: +0.052 [+0.046, +0.057] (n=500)
- GradExplainer - builtin: -0.288 [-0.300, -0.277] (n=500)
- IntegratedGradExplainer - builtin: -0.340 [-0.350, -0.329] (n=500)

