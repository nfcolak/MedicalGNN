# Test results, top-20 diagnoses

K = 20 | test rows = 4,174 | test subjects = 3,151 | protocol `fair_v10t` | 3 seeds (1234, 2025, 7) | 1,000 patient-cluster bootstrap resamples

Methods are in test-ranking order (mean test macro-F1 over seeds). Test macro-F1 is mean ± SD (sample SD, n = 3 seeds). The test fold was scored once per run, at the checkpoint selected on validation. The validation columns are the decision fold (checkpoint selection), not a result; the validation rank is the order in which the methods were selected.

| Test rank | Method | Test macro-F1 (mean ± SD) | Test accuracy (mean) | Test top-3 accuracy (mean) | Selected validation macro-F1 (mean; decision fold) | Validation rank (decision) | Seeds |
|---|---|---|---|---|---|---|---|
| 1 | CEI-GNN mp2 | 0.6894 ± 0.0013 | 0.7043 | 0.8975 | 0.6859 | 1 | 1234, 2025, 7 |
| 2 | GraphCare | 0.6714 ± 0.0044 | 0.6877 | 0.8802 | 0.6742 | 3 | 1234, 2025, 7 |
| 3 | GSAT | 0.6692 ± 0.0041 | 0.6848 | 0.8815 | 0.6718 | 4 | 1234, 2025, 7 |
| 4 | ProtGNN | 0.6655 ± 0.0023 | 0.6851 | 0.8828 | 0.6743 | 2 | 1234, 2025, 7 |

## Paired comparison (test macro-F1, delta = A − B)

Pairs are ordered by test rank (A ranks above B on test). Patient-cluster bootstrap, 1,000 resamples, 95% percentile CI; P(delta > 0) is the share of resamples with delta > 0.

| A | B | Delta | 95% CI | P(delta > 0) |
|---|---|---|---|---|
| CEI-GNN mp2 | GraphCare | +0.0180 | [+0.0102, +0.0258] | 1.000 |
| CEI-GNN mp2 | GSAT | +0.0202 | [+0.0117, +0.0290] | 1.000 |
| CEI-GNN mp2 | ProtGNN | +0.0239 | [+0.0155, +0.0324] | 1.000 |
| GraphCare | GSAT | +0.0021 | [-0.0065, +0.0103] | 0.683 |
| GraphCare | ProtGNN | +0.0059 | [-0.0023, +0.0137] | 0.913 |
| GSAT | ProtGNN | +0.0037 | [-0.0044, +0.0116] | 0.835 |
