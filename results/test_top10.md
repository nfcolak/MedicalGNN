# Test results, top-10 diagnoses

K = 10 | test rows = 3,175 | test subjects = 2,497 | protocol `fair_v10t` | 3 seeds (1234, 2025, 7) | 1,000 patient-cluster bootstrap resamples

Methods are in validation-ranking order (mean selected validation macro-F1 over seeds). Test macro-F1 is mean ± SD (sample SD, n = 3 seeds). The test fold was scored once per run.

| Rank | Method | Selected validation macro-F1 (mean) | Test macro-F1 (mean ± SD) | Test accuracy (mean) | Test top-3 accuracy (mean) | Seeds |
|---|---|---|---|---|---|---|
| 1 | CEI-GNN mp2 | 0.7451 | 0.7612 ± 0.0090 | 0.7543 | 0.9347 | 1234, 2025, 7 |
| 2 | ProtGNN | 0.7338 | 0.7510 ± 0.0062 | 0.7451 | 0.9302 | 1234, 2025, 7 |
| 3 | GCHM-PNA | 0.7336 | 0.7559 ± 0.0054 | 0.7506 | 0.9326 | 1234, 2025, 7 |
| 4 | GraphCare | 0.7332 | 0.7629 ± 0.0032 | 0.7543 | 0.9302 | 1234, 2025, 7 |
| 5 | GSAT | 0.7262 | 0.7509 ± 0.0042 | 0.7465 | 0.9283 | 1234, 2025, 7 |

## Paired comparison (test macro-F1, delta = A − B)

Patient-cluster bootstrap, 1,000 resamples, 95% percentile CI; P(delta > 0) is the share of resamples with delta > 0.

| A | B | Delta | 95% CI | P(delta > 0) |
|---|---|---|---|---|
| CEI-GNN mp2 | ProtGNN | +0.0102 | [+0.0024, +0.0178] | 0.997 |
| CEI-GNN mp2 | GCHM-PNA | +0.0053 | [-0.0019, +0.0117] | 0.916 |
| CEI-GNN mp2 | GraphCare | -0.0017 | [-0.0095, +0.0066] | 0.340 |
| CEI-GNN mp2 | GSAT | +0.0103 | [+0.0031, +0.0178] | 0.998 |
| ProtGNN | GCHM-PNA | -0.0050 | [-0.0122, +0.0021] | 0.073 |
| ProtGNN | GraphCare | -0.0119 | [-0.0196, -0.0038] | 0.001 |
| ProtGNN | GSAT | +0.0001 | [-0.0080, +0.0079] | 0.492 |
| GCHM-PNA | GraphCare | -0.0069 | [-0.0143, +0.0015] | 0.046 |
| GCHM-PNA | GSAT | +0.0050 | [-0.0025, +0.0126] | 0.897 |
| GraphCare | GSAT | +0.0119 | [+0.0045, +0.0198] | 0.997 |
