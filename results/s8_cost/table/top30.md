# Model cost, K=30 (top-30 labels)

Task identity: K=30; validation fold (timings: 500 graphs, sample seed 0); checkpoints seed 1234 (parameters, bench); training statistics seeds 7, 1234, 2025, 40 epochs; rows train 36592, validation 4534; test not used.

| Method | Params total | Params active | Explanation params | Forward ms/graph | Tensorize ms/graph | Inference peak RSS MB | Median s/epoch (contended) | Total train min | Epochs to 95% | Epochs to 99% | Selected epoch (mean, min-max) | Explanation overhead ms/graph | Train batch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 237,112 | 237,112 | 0 | 0.45 | 0.96 | 868.2 | 188.4 | 110.0 | 5.7 | 14.3 | 30.0 (25-38) | 0.11 | 128 |
| GSAT | 224,642 | 224,642 | 66,049 | 0.28 | 0.98 | 758.0 | 77.7 | 54.7 | 11.7 | 22.7 | 33.7 (29-38) | 0.01 | 128 |
| GraphCare-style (adapted) | 150,022 | 150,022 | 0 | 0.19 | 0.97 | 744.6 | 42.6 | 30.7 | 7.0 | 19.0 | 30.3 (20-37) | n/a | 32 |
| ProtGNN | 441,460 | 441,460 | 23,700 | 0.36 | 0.97 | 757.0 | 160.9 | 99.0 | 6.7 | 21.7 | 22.7 (14-30) | -0.00 | 128 |

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
