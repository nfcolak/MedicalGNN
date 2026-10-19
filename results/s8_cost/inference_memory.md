# Inference memory per model (peak resident set size, MB)

Measured 2026-10-18 with `python3 -m comparison.s8_cost.bench --execute` on an otherwise idle machine (1-minute load 1.20-1.83, no record flagged contended): stored final checkpoints (seed 1234), one process per method and K, the same 500 validation graphs for every method and K (sample seed 0, from the K=10 validation fold, a subset of the K=20 and K=30 folds), batch 64, 2 CPU threads, 3 rounds x 3 passes after a warm-up. Values are the whole process (ru_maxrss): Python and libraries (about 320 MB at start), the 500 parsed and encoded graphs, the model and the forward passes. Reloaded checkpoints reproduce the stored validation probabilities (all 12 parity checks passed). Training memory was not recorded in the final runs. Test fold not used.

| K | Method | Process start | After graphs and model | Peak, inference | Peak, with explanation extraction |
|---|---|---|---|---|---|
| 10 | CEI-GNN mp2 | 320.3 | 528.4 | 765.1 | 799.8 |
| 10 | GSAT | 321.3 | 490.8 | 756.1 | 756.3 |
| 10 | GraphCare-style (adapted) | 321.5 | 491.0 | 744.1 | no built-in explanation |
| 10 | ProtGNN | 320.2 | 492.4 | 756.6 | 756.8 |
| 20 | CEI-GNN mp2 | 321.3 | 528.8 | 761.4 | 800.5 |
| 20 | GSAT | 321.9 | 492.0 | 756.7 | 756.7 |
| 20 | GraphCare-style (adapted) | 321.0 | 490.1 | 740.6 | no built-in explanation |
| 20 | ProtGNN | 320.2 | 491.7 | 757.5 | 757.5 |
| 30 | CEI-GNN mp2 | 320.6 | 526.8 | 802.8 | 868.2 |
| 30 | GSAT | 321.6 | 492.0 | 757.9 | 758.0 |
| 30 | GraphCare-style (adapted) | 321.8 | 491.4 | 744.6 | no built-in explanation |
| 30 | ProtGNN | 321.8 | 494.4 | 756.9 | 757.0 |

Explanation extraction: CEI-GNN mp2 evidence ledger, GSAT top-5 edges, ProtGNN top-3 prototype similarities.
