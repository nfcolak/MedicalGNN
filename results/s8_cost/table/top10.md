# Model cost, K=10 (top-10 labels)

Task identity: K=10; validation fold (timings: 500 graphs, sample seed 0); checkpoints seed 1234 (parameters, bench); training statistics seeds 7, 1234, 2025, 40 epochs; rows train 25170, validation 3165; test not used.

| Method | Params total | Params active | Explanation params | Forward ms/graph | Tensorize ms/graph | Inference peak RSS MB | Median s/epoch (contended) | Total train min | Epochs to 95% | Epochs to 99% | Selected epoch (mean, min-max) | Explanation overhead ms/graph | Train batch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 206,736 | 206,736 | 0 | 0.38 | 0.97 | 799.8 | 53.9 | 37.2 | 3.3 | 16.3 | 29.0 (16-40) | 0.06 | 128 |
| GSAT | 217,966 | 217,966 | 66,049 | 0.28 | 0.97 | 756.3 | 54.1 | 37.9 | 4.3 | 12.7 | 27.7 (21-33) | 0.01 | 128 |
| GraphCare-style (adapted) | 138,226 | 138,226 | 0 | 0.19 | 0.97 | 744.1 | 20.8 | 16.5 | 2.3 | 14.7 | 30.7 (28-34) | n/a | 32 |
| ProtGNN | 420,564 | 420,564 | 6,900 | 0.37 | 0.98 | 756.8 | 47.2 | 34.0 | 4.7 | 16.7 | 21.7 (17-27) | -0.00 | 128 |

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
- Test fold: not used in any number of this table.
