# Summary over seeds (mean and SD, ddof=1)

Final runs, protocol fair_v10t; seeds 1234, 2025, 7; methods CEI-GNN mp2, GraphCare-style (adapted), GSAT, ProtGNN. Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run at the validation-selected checkpoint).

## K=10, test fold (reported result): 3175 rows, 2497 patients, seeds 1234, 2025, 7

| method | macro-F1 | SD | accuracy | SD | bal. acc | SD | rank in fold | best baseline (chosen on validation) |
|---|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 0.7612 | 0.0090 | 0.7543 | 0.0098 | 0.7638 | 0.0101 | 2 |  |
| GraphCare-style (adapted) | 0.7629 | 0.0032 | 0.7543 | 0.0025 | 0.7658 | 0.0025 | 1 |  |
| GSAT | 0.7509 | 0.0042 | 0.7465 | 0.0016 | 0.7558 | 0.0033 | 4 |  |
| ProtGNN | 0.7510 | 0.0062 | 0.7451 | 0.0058 | 0.7603 | 0.0064 | 3 | yes |

## K=10, validation fold (validation: decision fold): 3165 rows, 2473 patients, seeds 1234, 2025, 7

| method | macro-F1 | SD | accuracy | SD | bal. acc | SD | rank in fold | best baseline (chosen on validation) |
|---|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 0.7451 | 0.0033 | 0.7410 | 0.0026 | 0.7484 | 0.0030 | 1 |  |
| GraphCare-style (adapted) | 0.7332 | 0.0033 | 0.7315 | 0.0030 | 0.7370 | 0.0038 | 3 |  |
| GSAT | 0.7262 | 0.0015 | 0.7258 | 0.0011 | 0.7308 | 0.0016 | 4 |  |
| ProtGNN | 0.7338 | 0.0035 | 0.7329 | 0.0040 | 0.7418 | 0.0048 | 2 | yes |

## K=20, test fold (reported result): 4174 rows, 3151 patients, seeds 1234, 2025, 7

| method | macro-F1 | SD | accuracy | SD | bal. acc | SD | rank in fold | best baseline (chosen on validation) |
|---|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 0.6894 | 0.0013 | 0.7043 | 0.0011 | 0.7015 | 0.0036 | 1 |  |
| GraphCare-style (adapted) | 0.6714 | 0.0044 | 0.6877 | 0.0051 | 0.6812 | 0.0053 | 2 |  |
| GSAT | 0.6692 | 0.0041 | 0.6848 | 0.0045 | 0.6813 | 0.0091 | 3 |  |
| ProtGNN | 0.6655 | 0.0023 | 0.6851 | 0.0032 | 0.6754 | 0.0051 | 4 | yes |

## K=20, validation fold (validation: decision fold): 4183 rows, 3142 patients, seeds 1234, 2025, 7

| method | macro-F1 | SD | accuracy | SD | bal. acc | SD | rank in fold | best baseline (chosen on validation) |
|---|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 0.6859 | 0.0040 | 0.6978 | 0.0034 | 0.7015 | 0.0049 | 1 |  |
| GraphCare-style (adapted) | 0.6742 | 0.0017 | 0.6866 | 0.0026 | 0.6857 | 0.0011 | 3 |  |
| GSAT | 0.6718 | 0.0029 | 0.6842 | 0.0009 | 0.6861 | 0.0057 | 4 |  |
| ProtGNN | 0.6743 | 0.0041 | 0.6850 | 0.0024 | 0.6849 | 0.0015 | 2 | yes |

## K=30, test fold (reported result): 4566 rows, 3404 patients, seeds 1234, 2025, 7

| method | macro-F1 | SD | accuracy | SD | bal. acc | SD | rank in fold | best baseline (chosen on validation) |
|---|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 0.5781 | 0.0035 | 0.6662 | 0.0012 | 0.6261 | 0.0049 | 1 |  |
| GraphCare-style (adapted) | 0.5467 | 0.0020 | 0.6447 | 0.0041 | 0.5750 | 0.0063 | 4 |  |
| GSAT | 0.5570 | 0.0075 | 0.6483 | 0.0070 | 0.5964 | 0.0050 | 2 |  |
| ProtGNN | 0.5567 | 0.0023 | 0.6472 | 0.0066 | 0.5955 | 0.0052 | 3 | yes |

## K=30, validation fold (validation: decision fold): 4534 rows, 3388 patients, seeds 1234, 2025, 7

| method | macro-F1 | SD | accuracy | SD | bal. acc | SD | rank in fold | best baseline (chosen on validation) |
|---|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 0.5708 | 0.0037 | 0.6576 | 0.0027 | 0.6188 | 0.0037 | 1 |  |
| GraphCare-style (adapted) | 0.5491 | 0.0019 | 0.6452 | 0.0003 | 0.5624 | 0.0065 | 4 |  |
| GSAT | 0.5548 | 0.0023 | 0.6445 | 0.0012 | 0.5808 | 0.0082 | 3 |  |
| ProtGNN | 0.5632 | 0.0089 | 0.6471 | 0.0033 | 0.5941 | 0.0097 | 2 | yes |

