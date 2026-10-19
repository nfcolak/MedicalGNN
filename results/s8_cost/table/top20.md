# Model cost, K=20 (top-20 labels)

Task identity: K=20; validation fold (timings: 500 graphs, sample seed 0); checkpoints seed 1234 (parameters, bench); training statistics seeds 7, 1234, 2025, 40 epochs; rows train 33532, validation 4183; test not used.

| Method | Params total | Params active | Explanation params | Forward ms/graph | Tensorize ms/graph | Inference peak RSS MB | Median s/epoch (contended) | Total train min | Epochs to 95% | Epochs to 99% | Selected epoch (mean, min-max) | Explanation overhead ms/graph | Train batch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 223,240 | 223,240 | 0 | 0.41 | 0.97 | 800.5 | 80.0 | 55.2 | 4.7 | 19.0 | 36.0 (30-40) | 0.06 | 128 |
| GSAT | 222,520 | 222,520 | 66,049 | 0.28 | 0.99 | 756.7 | 66.5 | 46.5 | 7.7 | 20.7 | 29.7 (20-37) | 0.01 | 128 |
| GraphCare-style (adapted) | 145,340 | 145,340 | 0 | 0.19 | 0.97 | 740.6 | 26.9 | 22.4 | 6.7 | 20.3 | 31.7 (19-38) | n/a | 32 |
| ProtGNN | 431,728 | 431,728 | 14,800 | 0.33 | 0.97 | 757.5 | 65.1 | 46.1 | 4.3 | 12.3 | 23.0 (8-31) | 0.01 | 128 |

Notes: forward, tensorize and RSS are inference (bench, validation graphs, 2 threads). Median s/epoch, total minutes, epochs to 95%/99% and selected epoch are training runs (seeds as above). Explanation overhead: GSAT edge top-5 extraction, ProtGNN top-3 prototype similarities, CEI-GNN mp2 ledger extraction over logits-only; n/a = no built-in explanation.

## Caveats

- No inference measurement was flagged contended (1-minute load 1.20-1.83, limit 2).
- Inference bench sample: 500 validation graphs (sample seed 0, drawn from the K=10 validation fold of the seed-1234 GSAT run; the same graphs for every K), checkpoint seed 1234, 2 threads, batch 64, 3 round(s) x 3 timed pass(es) after 1 warm-up.
- Bench parity PASSED for all 12 records (max abs diff of softmax vs stored validation predictions 9.5e-07, tolerance 0.0001).
- Negative explanation overhead (K=10 protgnn, K=30 protgnn) is timing noise; read as about 0 ms/graph.
- Training runs were executed 6 concurrent runs x 3 CPU threads, nice 5 (wall times carry contention). Flagged contended epochs (epoch > 1.6 x run median) per K: K=10: 0, K=20: 0, K=30: 183.
- Training s/epoch for K=30 comes from contended runs (6 concurrent runs x 3 threads; overlapping runners); it is not an uncontended speed.
- Selected epoch at the budget boundary (38-40 of 40): K=10 cei_mp2 1/3, K=20 cei_mp2 2/3, K=20 graphcare 2/3, K=30 cei_mp2 1/3, K=30 gsat 1/3 runs.
- Training statistics: seeds 7, 1234, 2025, 40 epochs, selection by best validation macro-F1; epochs to 95%/99% and selected epoch are means over seeds.
- Training batch sizes differ, read from the run bindings: GraphCare-style (adapted) 32; CEI-GNN mp2, GSAT, ProtGNN 128. The inference bench uses batch 64 for every method.
- Training memory was not recorded in the final runs; the memory column is inference memory only.
- Parameter counts: seed-1234 models rebuilt and strict-loaded; recount matches the bindings for every row.
- Explanation columns: n/a means no built-in explanation head; explanation parameters are 0 for CEI-GNN mp2, GraphCare-style (adapted) (CEI-GNN mp2: its evidence heads are the prediction path).
- The timings, memory and parameter counts do not depend on the fold: their inputs are 500 validation graphs.
