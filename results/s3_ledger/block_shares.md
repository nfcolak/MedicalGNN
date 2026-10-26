# Evidence-ledger probes of the CEI-GNN mp2 final runs

Final fair_v10t runs, K = 10, 20, 30 x seeds 1234, 2025, 7 (`best.pt`), forward passes only. Test is the reported fold; validation (the decision fold, the earlier numbers) is kept below. Probes: `comparison/s3_ledger/probe.py`.

- Mass share (primary): per graph, mass_b = sum over all items of block b and all classes of |part|; denominator = sum of the five masses (bias excluded); averaged over graphs. Parts of opposite sign inside a block do not cancel: how much signed evidence a block puts on the table, not how much it moves the logit.
- Block-total variant: |block total T_c| summed over classes; cancellation inside a block allowed.
- Predicted-class contribution: mean block total at the argmax class (signed, logit units); with the bias the seven numbers add up to the mean predicted-class logit.
- Removal gap (node block): relative difference between the true effect of deleting a node (denominator renormalised) and the ledger-naive effect (its own part).
- Pair second difference: f(zi+zj) - f(zi) - f(zj) + f(0) of the pair vote; zero iff the vote is endpoint-additive.

## Test (reported fold)

9 runs, first 300 sample ids of the fold per K after sorting the id strings (identical for the three seeds; classes present: K=10: 10, K=20: 20, K=30: 27). Parity (reloaded softmax vs stored predictions of the fold): max abs diff 8.9e-08 (limit 1e-4), argmax agreement 100% in every run. Ledger exactness (max |logit - bias - sum of parts|, float64): max 5.1e-06, min 2.9e-06.

### Absolute evidence mass share, % (item-level |parts|; mean ± SD over 3 seeds, n-1)

| K | node | edge | pair | nbr | absence |
|---|---|---|---|---|---|
| 10 | 26.7 ± 1.6 | 33.3 ± 0.7 | 14.7 ± 0.8 | 18.2 ± 1.5 | 7.1 ± 0.9 |
| 20 | 26.6 ± 0.5 | 37.5 ± 1.1 | 14.4 ± 0.2 | 13.1 ± 1.3 | 8.5 ± 0.5 |
| 30 | 24.3 ± 2.2 | 37.2 ± 0.3 | 13.9 ± 0.6 | 17.0 ± 2.1 | 7.6 ± 0.2 |

### Same, block-total variant (cancellation inside a block allowed), %

| K | node | edge | pair | nbr | absence |
|---|---|---|---|---|---|
| 10 | 26.9 ± 2.7 | 26.6 ± 0.4 | 14.7 ± 0.9 | 27.5 ± 1.5 | 4.2 ± 0.4 |
| 20 | 33.5 ± 2.6 | 25.5 ± 2.4 | 12.3 ± 0.6 | 22.5 ± 0.9 | 6.2 ± 0.4 |
| 30 | 28.6 ± 2.9 | 26.7 ± 1.0 | 11.5 ± 0.6 | 27.6 ± 2.6 | 5.7 ± 0.4 |

### Contribution to the predicted-class logit (mean over graphs, logit units; mean ± SD over seeds)

| K | node | edge | pair | nbr | absence | bias | sum = predicted-class logit |
|---|---|---|---|---|---|---|---|
| 10 | 0.67 ± 0.13 | 0.64 ± 0.15 | 0.19 ± 0.06 | 1.82 ± 0.26 | 0.01 ± 0.01 | 0.00 ± 0.00 | 3.33 ± 0.10 |
| 20 | 0.70 ± 0.18 | 0.82 ± 0.11 | 0.16 ± 0.05 | 2.26 ± 0.15 | 0.02 ± 0.00 | 0.01 ± 0.00 | 3.97 ± 0.06 |
| 30 | 0.81 ± 0.06 | 1.07 ± 0.09 | 0.17 ± 0.07 | 2.18 ± 0.18 | 0.03 ± 0.00 | 0.01 ± 0.00 | 4.27 ± 0.14 |

### Per run

| K | seed | parity max diff | ledger max diff | node | edge | pair | nbr | absence (item-level mass %) |
|---|---|---|---|---|---|---|---|---|
| 10 | 7 | 6.0e-08 | 4.5e-06 | 25.1 | 34.0 | 14.8 | 19.1 | 7.0 |
| 10 | 1234 | 0.0e+00 | 4.4e-06 | 28.4 | 32.5 | 13.8 | 19.0 | 6.3 |
| 10 | 2025 | 7.5e-09 | 3.1e-06 | 26.8 | 33.3 | 15.4 | 16.5 | 8.1 |
| 20 | 7 | 9.3e-10 | 5.1e-06 | 26.6 | 38.6 | 14.6 | 11.7 | 8.4 |
| 20 | 1234 | 0.0e+00 | 4.8e-06 | 27.0 | 36.4 | 14.4 | 14.2 | 8.0 |
| 20 | 2025 | 0.0e+00 | 2.9e-06 | 26.0 | 37.4 | 14.1 | 13.4 | 9.0 |
| 30 | 7 | 5.6e-09 | 3.9e-06 | 25.3 | 37.5 | 13.3 | 16.2 | 7.7 |
| 30 | 1234 | 1.9e-09 | 4.0e-06 | 21.8 | 37.1 | 14.3 | 19.3 | 7.4 |
| 30 | 2025 | 8.9e-08 | 3.9e-06 | 25.9 | 37.0 | 14.0 | 15.5 | 7.6 |

### Node-block removal gap and nbr share (seed 1234)

| K | graphs | gap, all classes (mean) | gap, all classes (median) | gap, predicted class | nbr share item-level | nbr share block-total |
|---|---|---|---|---|---|---|
| 10 | 200 | 0.358 | 0.351 | 0.364 | 0.193 | 0.273 |
| 30 | 200 | 0.356 | 0.355 | 0.340 | 0.193 | 0.304 |

### Pair vote second difference

| K | seed | pairs sampled | max abs second difference | mean abs pair vote |
|---|---|---|---|---|
| 10 | 7 | 4000 | 4.47e-07 | 0.470 |
| 20 | 2025 | 4000 | 7.00e-07 | 0.528 |
| 30 | 1234 | 4000 | 8.79e-07 | 0.547 |

## Validation (decision fold; earlier numbers)

9 runs, first 300 sample ids of the fold per K after sorting the id strings (identical for the three seeds; classes present: K=10: 10, K=20: 20, K=30: 28). Parity (reloaded softmax vs stored predictions of the fold): max abs diff 1.5e-08 (limit 1e-4), argmax agreement 100% in every run. Ledger exactness (max |logit - bias - sum of parts|, float64): max 5.9e-06, min 2.6e-06.

### Absolute evidence mass share, % (item-level |parts|; mean ± SD over 3 seeds, n-1)

| K | node | edge | pair | nbr | absence |
|---|---|---|---|---|---|
| 10 | 26.7 ± 1.7 | 33.2 ± 0.9 | 14.1 ± 0.8 | 19.1 ± 1.4 | 6.9 ± 0.8 |
| 20 | 26.4 ± 0.3 | 37.9 ± 0.9 | 14.0 ± 0.2 | 13.4 ± 1.2 | 8.3 ± 0.5 |
| 30 | 24.1 ± 2.1 | 37.5 ± 0.3 | 13.7 ± 0.5 | 17.3 ± 1.9 | 7.4 ± 0.2 |

### Same, block-total variant (cancellation inside a block allowed), %

| K | node | edge | pair | nbr | absence |
|---|---|---|---|---|---|
| 10 | 26.8 ± 2.6 | 25.8 ± 0.4 | 14.3 ± 1.1 | 28.9 ± 1.1 | 4.2 ± 0.4 |
| 20 | 33.3 ± 2.8 | 25.2 ± 2.7 | 12.2 ± 0.4 | 23.2 ± 0.5 | 6.2 ± 0.4 |
| 30 | 28.4 ± 2.8 | 26.4 ± 1.0 | 11.6 ± 0.6 | 27.9 ± 2.6 | 5.6 ± 0.4 |

### Contribution to the predicted-class logit (mean over graphs, logit units; mean ± SD over seeds)

| K | node | edge | pair | nbr | absence | bias | sum = predicted-class logit |
|---|---|---|---|---|---|---|---|
| 10 | 0.69 ± 0.11 | 0.64 ± 0.10 | 0.17 ± 0.05 | 2.00 ± 0.23 | 0.01 ± 0.00 | 0.00 ± 0.00 | 3.51 ± 0.14 |
| 20 | 0.68 ± 0.16 | 0.78 ± 0.15 | 0.16 ± 0.04 | 2.36 ± 0.17 | 0.01 ± 0.00 | 0.01 ± 0.00 | 3.99 ± 0.05 |
| 30 | 0.82 ± 0.03 | 1.07 ± 0.16 | 0.18 ± 0.08 | 2.30 ± 0.26 | 0.01 ± 0.00 | 0.01 ± 0.00 | 4.39 ± 0.11 |

### Per run

| K | seed | parity max diff | ledger max diff | node | edge | pair | nbr | absence (item-level mass %) |
|---|---|---|---|---|---|---|---|---|
| 10 | 7 | 5.6e-09 | 3.3e-06 | 24.8 | 34.2 | 14.3 | 19.9 | 6.8 |
| 10 | 1234 | 4.7e-09 | 3.6e-06 | 28.2 | 32.5 | 13.2 | 19.9 | 6.2 |
| 10 | 2025 | 1.5e-08 | 3.1e-06 | 27.0 | 33.0 | 14.8 | 17.4 | 7.8 |
| 20 | 7 | 0.0e+00 | 5.9e-06 | 26.5 | 38.9 | 14.3 | 12.2 | 8.2 |
| 20 | 1234 | 0.0e+00 | 5.2e-06 | 26.7 | 37.1 | 14.0 | 14.5 | 7.8 |
| 20 | 2025 | 0.0e+00 | 2.6e-06 | 26.1 | 37.7 | 13.9 | 13.5 | 8.8 |
| 30 | 7 | 0.0e+00 | 3.0e-06 | 25.1 | 37.8 | 13.1 | 16.5 | 7.6 |
| 30 | 1234 | 0.0e+00 | 3.5e-06 | 21.7 | 37.4 | 14.1 | 19.5 | 7.3 |
| 30 | 2025 | 0.0e+00 | 5.1e-06 | 25.7 | 37.2 | 13.8 | 15.8 | 7.4 |

### Node-block removal gap and nbr share (seed 1234)

| K | graphs | gap, all classes (mean) | gap, all classes (median) | gap, predicted class | nbr share item-level | nbr share block-total |
|---|---|---|---|---|---|---|
| 10 | 200 | 0.358 | 0.354 | 0.370 | 0.203 | 0.290 |
| 30 | 200 | 0.357 | 0.356 | 0.340 | 0.193 | 0.307 |

### Pair vote second difference

| K | seed | pairs sampled | max abs second difference | mean abs pair vote |
|---|---|---|---|---|
| 10 | 7 | 4000 | 4.17e-07 | 0.469 |
| 20 | 2025 | 4000 | 6.85e-07 | 0.523 |
| 30 | 1234 | 4000 | 6.71e-07 | 0.554 |

## Validation reproduction against the stored validation files

Re-run with `--fold validation` against the stored validation files of the earlier probe (max abs difference of every shared number).

| stored file | rows | max abs diff | column of the max |
|---|---|---|---|
| block_shares.csv | 9 of 9 | 0.00e+00 | - |
| block_shares_blocktotal.csv | 9 of 9 | 0.00e+00 | - |
| block_shares_extra.csv | 9 of 9 | 0.00e+00 | - |
| ledger_maxdiff column (float32 round-off level, informational) | 9 of 9 | 0.00e+00 | ledger_maxdiff |
| removal_probe_K10_s1234.json | 1 | 0.00e+00 | - |
| removal_probe_K30_s1234.json | 1 | 0.00e+00 | - |
| pairsecond_K10_s7.json | 1 | 0.00e+00 | - |
| pairsecond_K20_s2025.json | 1 | 0.00e+00 | - |
| pairsecond_K30_s1234.json | 1 | 0.00e+00 | - |

