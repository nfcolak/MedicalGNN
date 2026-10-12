# Test results, top-30 diagnoses

K = 30 | test rows = 4,566 | test subjects = 3,404 | protocol `fair_v10t` | 3 seeds (1234, 2025, 7) | 1,000 patient-cluster bootstrap resamples

Methods are in validation-ranking order (mean selected validation macro-F1 over seeds). Test macro-F1 is mean ± SD (sample SD, n = 3 seeds). The test fold was scored once per run.

| Rank | Method | Selected validation macro-F1 (mean) | Test macro-F1 (mean ± SD) | Test accuracy (mean) | Test top-3 accuracy (mean) | Seeds |
|---|---|---|---|---|---|---|
| 1 | CEI-GNN mp2 | 0.5708 | 0.5781 ± 0.0035 | 0.6662 | 0.8631 | 1234, 2025, 7 |
| 2 | ProtGNN | 0.5632 | 0.5567 ± 0.0023 | 0.6472 | 0.8485 | 1234, 2025, 7 |
| 3 | GCHM-PNA | 0.5602 | 0.5556 ± 0.0087 | 0.6479 | 0.8498 | 1234, 2025, 7 |
| 4 | GSAT | 0.5548 | 0.5570 ± 0.0075 | 0.6483 | 0.8469 | 1234, 2025, 7 |
| 5 | GraphCare | 0.5491 | 0.5467 ± 0.0020 | 0.6447 | 0.8437 | 1234, 2025, 7 |

## Paired comparison (test macro-F1, delta = A − B)

Patient-cluster bootstrap, 1,000 resamples, 95% percentile CI; P(delta > 0) is the share of resamples with delta > 0.

| A | B | Delta | 95% CI | P(delta > 0) |
|---|---|---|---|---|
| CEI-GNN mp2 | ProtGNN | +0.0215 | [+0.0111, +0.0318] | 1.000 |
| CEI-GNN mp2 | GCHM-PNA | +0.0225 | [+0.0091, +0.0362] | 1.000 |
| CEI-GNN mp2 | GSAT | +0.0212 | [+0.0098, +0.0347] | 1.000 |
| CEI-GNN mp2 | GraphCare | +0.0314 | [+0.0213, +0.0429] | 1.000 |
| ProtGNN | GCHM-PNA | +0.0010 | [-0.0092, +0.0124] | 0.614 |
| ProtGNN | GSAT | -0.0003 | [-0.0097, +0.0104] | 0.493 |
| ProtGNN | GraphCare | +0.0099 | [-0.0010, +0.0200] | 0.958 |
| GCHM-PNA | GSAT | -0.0013 | [-0.0116, +0.0081] | 0.401 |
| GCHM-PNA | GraphCare | +0.0089 | [-0.0030, +0.0203] | 0.917 |
| GSAT | GraphCare | +0.0103 | [-0.0013, +0.0212] | 0.961 |

Note: Hypothyroidism has 0 test rows at K = 30, so its F1 is 0 for every method; End stage renal disease has 5 test rows.
