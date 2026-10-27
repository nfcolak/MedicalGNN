# Explanation results, top-20, test fold: final methods and CEI-GNN training-recipe variants

Protocol fair_v10t, K = 20. Performance: all test visits (stored test predictions of each run). Explanations: the same 1,000 random test patients (1,345 visits) in every run, intervention contract v6. Mean over the three seeds (1234, 2025, 7); ± is the seed SD. Deletion vs random: how much more the predicted-class probability drops when the explanation's top-ranked evidence is deleted than when random evidence of the same size is deleted (higher is better). Fidelity+: drop when the top 20% are deleted (higher is better). Fidelity-: drop when only the top 20% are kept (lower is better). Sparsity: share of nodes that can be dropped while keeping 90% of the importance mass.

LS = label smoothing 0.1, WD = decoupled weight decay 0.01 on the weight matrices, DE = DropEdge 0.2 (1 = used as in the final recipe, 0 = removed). The recipe variants were explained after the test explanation results of the final runs had been seen; they are a diagnostic, not a model selection. The reported model stays CEI-GNN final.

## 1. Performance and explanation scores

| Model | LS | WD | DE | Macro-F1 | Accuracy | Balanced accuracy | Top-3 accuracy | Mean confidence | IG | Grad | GNNExplainer | Built-in | Built-in fidelity+ | Built-in fidelity- | Built-in sparsity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GSAT | own recipe |  |  | 0.6692 ± 0.0041 | 0.6848 | 0.6813 | 0.8815 | 0.697 | 0.310 ± 0.007 | 0.301 ± 0.001 | 0.264 ± 0.030 | 0.292 ± 0.011 | 0.525 | −0.057 | 0.099 |
| GraphCare-style | own recipe |  |  | 0.6714 ± 0.0044 | 0.6877 | 0.6812 | 0.8802 | 0.650 | 0.307 ± 0.009 | 0.333 ± 0.026 | 0.170 ± 0.047 | −0.012 ± 0.060 | 0.051 | 0.374 | 0.258 |
| ProtGNN | own recipe |  |  | 0.6655 ± 0.0023 | 0.6851 | 0.6754 | 0.8828 | 0.734 | 0.284 ± 0.004 | 0.261 ± 0.005 | 0.214 ± 0.014 | −0.009 ± 0.005 | 0.012 | 0.659 | 0.937 |
| CEI-GNN final | 1 | 1 | 1 | 0.6894 ± 0.0013 | 0.7043 | 0.7015 | 0.8975 | 0.646 | 0.259 ± 0.004 | 0.254 ± 0.002 | 0.234 ± 0.009 | 0.276 ± 0.008 | 0.512 | −0.008 | 0.749 |
| CEI-GNN, weight decay removed | 1 | 0 | 1 | 0.6896 ± 0.0041 | 0.7044 | 0.7022 | 0.8965 | 0.646 | 0.258 ± 0.003 | 0.253 ± 0.002 | 0.230 ± 0.011 | 0.277 ± 0.007 | 0.512 | −0.009 | 0.743 |
| CEI-GNN, DropEdge removed | 1 | 1 | 0 | 0.6904 ± 0.0061 | 0.7051 | 0.7024 | 0.8985 | 0.648 | 0.259 ± 0.008 | 0.254 ± 0.008 | 0.226 ± 0.008 | 0.274 ± 0.006 | 0.512 | −0.003 | 0.739 |
| CEI-GNN, label smoothing removed | 0 | 1 | 1 | 0.6805 ± 0.0035 | 0.6995 | 0.6982 | 0.8943 | 0.733 | 0.294 ± 0.005 | 0.286 ± 0.004 | 0.248 ± 0.010 | 0.322 ± 0.007 | 0.585 | 0.001 | 0.712 |
| CEI-GNN, label smoothing and weight decay removed | 0 | 0 | 1 | 0.6793 ± 0.0025 | 0.6974 | 0.6995 | 0.8947 | 0.728 | 0.290 ± 0.010 | 0.285 ± 0.004 | 0.242 ± 0.011 | 0.324 ± 0.003 | 0.583 | −0.005 | 0.705 |
| CEI-GNN, all three removed | 0 | 0 | 0 | 0.6816 ± 0.0035 | 0.6998 | 0.7021 | 0.8946 | 0.730 | 0.293 ± 0.008 | 0.286 ± 0.006 | 0.245 ± 0.006 | 0.322 ± 0.002 | 0.580 | 0.003 | 0.711 |

GraphCare-style and ProtGNN: the built-in explanation is not better than random, so its side metrics are not comparable. ProtGNN built-in: visit-runs whose prototype subgraph gives zero importance to every evidence node are left out of its mean (see visits explained in the summary).

## 2. Per explanation source: scores, ranks (1 = best of the rows) and mean rank

### IG

| Model | deletion vs random | fidelity+ | fidelity- | sparsity | Mean rank |
|---|---|---|---|---|---|
| GSAT | 0.310 ± 0.007 (1) | 0.556 ± 0.017 (4) | −0.077 ± 0.011 (2) | 0.407 ± 0.022 (3) | 2.50 |
| GraphCare-style | 0.307 ± 0.009 (2) | 0.531 ± 0.015 (6) | −0.143 ± 0.021 (1) | 0.537 ± 0.037 (2) | 2.75 |
| ProtGNN | 0.284 ± 0.004 (6) | 0.589 ± 0.012 (1) | 0.446 ± 0.019 (9) | 0.571 ± 0.143 (1) | 4.25 |
| CEI-GNN, label smoothing removed | 0.294 ± 0.005 (3) | 0.558 ± 0.007 (2) | 0.056 ± 0.007 (8) | 0.337 ± 0.009 (4) | 4.25 |
| CEI-GNN, all three removed | 0.293 ± 0.008 (4) | 0.557 ± 0.015 (3) | 0.055 ± 0.010 (7) | 0.322 ± 0.008 (9) | 5.75 |
| CEI-GNN final | 0.259 ± 0.004 (8) | 0.497 ± 0.005 (8) | 0.041 ± 0.014 (3) | 0.333 ± 0.012 (5) | 6.00 |
| CEI-GNN, label smoothing and weight decay removed | 0.290 ± 0.010 (5) | 0.555 ± 0.012 (5) | 0.054 ± 0.010 (6) | 0.327 ± 0.005 (8) | 6.00 |
| CEI-GNN, DropEdge removed | 0.259 ± 0.008 (7) | 0.498 ± 0.010 (7) | 0.052 ± 0.012 (5) | 0.328 ± 0.010 (6) | 6.25 |
| CEI-GNN, weight decay removed | 0.258 ± 0.003 (9) | 0.494 ± 0.007 (9) | 0.043 ± 0.013 (4) | 0.327 ± 0.011 (7) | 7.25 |

### Grad

| Model | deletion vs random | fidelity+ | fidelity- | sparsity | Mean rank |
|---|---|---|---|---|---|
| GraphCare-style | 0.333 ± 0.026 (1) | 0.506 ± 0.011 (6) | −0.125 ± 0.025 (1) | 0.747 ± 0.075 (1) | 2.25 |
| GSAT | 0.301 ± 0.001 (2) | 0.539 ± 0.009 (4) | −0.062 ± 0.022 (2) | 0.470 ± 0.010 (3) | 2.75 |
| CEI-GNN, label smoothing removed | 0.286 ± 0.004 (3) | 0.545 ± 0.010 (1) | 0.063 ± 0.006 (7) | 0.409 ± 0.006 (5) | 4.00 |
| CEI-GNN, label smoothing and weight decay removed | 0.285 ± 0.004 (5) | 0.543 ± 0.006 (2) | 0.060 ± 0.007 (6) | 0.405 ± 0.002 (6) | 4.75 |
| CEI-GNN, all three removed | 0.286 ± 0.006 (4) | 0.541 ± 0.016 (3) | 0.065 ± 0.010 (8) | 0.409 ± 0.012 (4) | 4.75 |
| ProtGNN | 0.261 ± 0.005 (6) | 0.525 ± 0.012 (5) | 0.449 ± 0.010 (9) | 0.621 ± 0.025 (2) | 5.50 |
| CEI-GNN final | 0.254 ± 0.002 (8) | 0.483 ± 0.005 (8) | 0.050 ± 0.009 (3) | 0.400 ± 0.008 (7) | 6.50 |
| CEI-GNN, DropEdge removed | 0.254 ± 0.008 (7) | 0.487 ± 0.009 (7) | 0.057 ± 0.010 (5) | 0.398 ± 0.015 (8) | 6.75 |
| CEI-GNN, weight decay removed | 0.253 ± 0.002 (9) | 0.480 ± 0.006 (9) | 0.052 ± 0.012 (4) | 0.395 ± 0.007 (9) | 7.75 |

### GNNExplainer

| Model | deletion vs random | fidelity+ | fidelity- | sparsity | Mean rank |
|---|---|---|---|---|---|
| GSAT | 0.264 ± 0.030 (1) | 0.402 ± 0.048 (8) | 0.030 ± 0.068 (6) | 0.113 ± 0.001 (1) | 4.00 |
| CEI-GNN final | 0.234 ± 0.009 (5) | 0.448 ± 0.012 (4) | 0.002 ± 0.017 (1) | 0.102 ± 0.000 (6) | 4.00 |
| CEI-GNN, label smoothing removed | 0.248 ± 0.010 (2) | 0.476 ± 0.017 (1) | 0.028 ± 0.012 (5) | 0.101 ± 0.000 (8) | 4.00 |
| CEI-GNN, label smoothing and weight decay removed | 0.242 ± 0.011 (4) | 0.467 ± 0.019 (2) | 0.027 ± 0.018 (4) | 0.101 ± 0.000 (7) | 4.25 |
| CEI-GNN, all three removed | 0.245 ± 0.006 (3) | 0.466 ± 0.010 (3) | 0.041 ± 0.001 (8) | 0.102 ± 0.000 (4) | 4.50 |
| CEI-GNN, weight decay removed | 0.230 ± 0.011 (6) | 0.437 ± 0.014 (6) | 0.012 ± 0.013 (2) | 0.102 ± 0.000 (5) | 4.75 |
| CEI-GNN, DropEdge removed | 0.226 ± 0.008 (7) | 0.430 ± 0.016 (7) | 0.027 ± 0.008 (3) | 0.102 ± 0.000 (3) | 5.00 |
| ProtGNN | 0.214 ± 0.014 (8) | 0.442 ± 0.020 (5) | 0.370 ± 0.025 (9) | 0.103 ± 0.001 (2) | 6.00 |
| GraphCare-style | 0.170 ± 0.047 (9) | 0.316 ± 0.079 (9) | 0.034 ± 0.077 (7) | 0.101 ± 0.001 (9) | 8.50 |

### Built-in

| Model | deletion vs random | fidelity+ | fidelity- | sparsity | Mean rank |
|---|---|---|---|---|---|
| CEI-GNN, label smoothing and weight decay removed | 0.324 ± 0.003 (1) | 0.583 ± 0.014 (2) | −0.005 ± 0.003 (4) | 0.705 ± 0.017 (7) | 3.50 |
| CEI-GNN, weight decay removed | 0.277 ± 0.007 (5) | 0.512 ± 0.021 (5) | −0.009 ± 0.012 (2) | 0.743 ± 0.019 (3) | 3.75 |
| CEI-GNN, label smoothing removed | 0.322 ± 0.007 (3) | 0.585 ± 0.020 (1) | 0.001 ± 0.003 (6) | 0.712 ± 0.017 (5) | 3.75 |
| CEI-GNN final | 0.276 ± 0.008 (6) | 0.512 ± 0.020 (6) | −0.008 ± 0.010 (3) | 0.749 ± 0.018 (2) | 4.25 |
| GSAT | 0.292 ± 0.011 (4) | 0.525 ± 0.013 (4) | −0.057 ± 0.018 (1) | 0.099 ± 0.006 (9) | 4.50 |
| CEI-GNN, all three removed | 0.322 ± 0.002 (2) | 0.580 ± 0.018 (3) | 0.003 ± 0.003 (7) | 0.711 ± 0.024 (6) | 4.50 |
| CEI-GNN, DropEdge removed | 0.274 ± 0.006 (7) | 0.512 ± 0.009 (7) | −0.003 ± 0.011 (5) | 0.739 ± 0.017 (4) | 5.75 |
| ProtGNN | −0.009 ± 0.005 (8) | 0.012 ± 0.002 (9) | 0.659 ± 0.012 (9) | 0.937 ± 0.021 (1) | 6.75 |
| GraphCare-style | −0.012 ± 0.060 (9) | 0.051 ± 0.044 (8) | 0.374 ± 0.056 (8) | 0.258 ± 0.038 (8) | 8.25 |

## 3. Per class: final CEI-GNN vs label smoothing removed (R1_ls0)

F1 on all test visits; explanation score (deletion vs random) by true class on the explained visits. F1 gain from LS = F1 final − F1 without LS; se = seed standard error of that difference.

| Class | Test visits | Explained visits | F1 final | F1 without LS | F1 gain from LS | se | IG final | IG without LS | Built-in final | Built-in without LS |
|---|---|---|---|---|---|---|---|---|---|---|
| Hypokalemia | 102 | 25 | 0.596 | 0.554 | +0.042 | 0.007 | 0.233 | 0.273 | 0.241 | 0.283 |
| Epilepsy | 99 | 33 | 0.845 | 0.810 | +0.035 | 0.008 | 0.358 | 0.427 | 0.333 | 0.381 |
| Sepsis | 71 | 24 | 0.402 | 0.380 | +0.021 | 0.013 | 0.171 | 0.224 | 0.200 | 0.237 |
| Acute pharyngitis | 72 | 28 | 0.750 | 0.730 | +0.021 | 0.016 | 0.251 | 0.290 | 0.265 | 0.339 |
| Lower respiratory disease | 379 | 127 | 0.682 | 0.664 | +0.018 | 0.010 | 0.241 | 0.274 | 0.262 | 0.300 |
| Heart failure | 141 | 46 | 0.627 | 0.612 | +0.015 | 0.006 | 0.264 | 0.297 | 0.278 | 0.300 |
| Alcohol abuse with intoxication | 97 | 34 | 0.729 | 0.718 | +0.012 | 0.008 | 0.289 | 0.334 | 0.286 | 0.352 |
| Unspecified atrial fibrillation | 115 | 45 | 0.682 | 0.673 | +0.009 | 0.019 | 0.250 | 0.276 | 0.274 | 0.312 |
| Major depressive disorder | 249 | 83 | 0.903 | 0.895 | +0.009 | 0.006 | 0.276 | 0.270 | 0.316 | 0.382 |
| Acute kidney failure | 115 | 29 | 0.441 | 0.434 | +0.007 | 0.013 | 0.237 | 0.275 | 0.248 | 0.295 |
| Anemia | 117 | 45 | 0.671 | 0.665 | +0.006 | 0.006 | 0.316 | 0.375 | 0.322 | 0.381 |
| GI bleed | 342 | 119 | 0.811 | 0.806 | +0.005 | 0.002 | 0.316 | 0.361 | 0.327 | 0.386 |
| Back or spine pain | 456 | 130 | 0.744 | 0.741 | +0.003 | 0.003 | 0.228 | 0.254 | 0.267 | 0.303 |
| Head injury | 289 | 84 | 0.772 | 0.770 | +0.002 | 0.002 | 0.266 | 0.291 | 0.279 | 0.307 |
| Diabetes mellitus | 155 | 42 | 0.717 | 0.715 | +0.002 | 0.008 | 0.272 | 0.306 | 0.304 | 0.366 |
| UTI or pyelonephritis | 385 | 145 | 0.718 | 0.718 | +0.001 | 0.006 | 0.315 | 0.360 | 0.291 | 0.330 |
| Limb injury or pain | 420 | 131 | 0.616 | 0.615 | +0.000 | 0.004 | 0.167 | 0.189 | 0.201 | 0.243 |
| Non-ST elevation (NSTEMI) myocardial infarction | 85 | 31 | 0.746 | 0.747 | −0.001 | 0.007 | 0.269 | 0.342 | 0.260 | 0.327 |
| Unsp intestnl obst | 126 | 45 | 0.673 | 0.686 | −0.013 | 0.010 | 0.332 | 0.390 | 0.349 | 0.407 |
| Skin or soft-tissue infection | 359 | 99 | 0.664 | 0.678 | −0.014 | 0.006 | 0.202 | 0.231 | 0.236 | 0.287 |

Spearman correlation across classes: F1 gain from LS vs IG cost of LS 0.12; vs built-in cost −0.17.
