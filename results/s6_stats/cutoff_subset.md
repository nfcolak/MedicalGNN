# Cut-off = ED discharge subset (test fold)

Reported fold: test. Methods CEI-GNN mp2, GSAT, GraphCare-style (adapted), ProtGNN, GCHM-PNA; K 10, 20, 30; seeds 1234, 2025, 7 (45 stored runs). Subset = the cut-off of the stay equals the ED discharge time; rest = all other stays. Gap = subset minus rest in points (seed mean), interval = patient bootstrap (B=1000, shared subject draw, percentile 95%).

## Subset size

| K | fold | rows scored | rows in subset | share |
|---|---|---|---|---|
| 10 | test | 3175 | 179 | 5.6% |
| 10 | validation | 3165 | 197 | 6.2% |
| 20 | test | 4174 | 213 | 5.1% |
| 20 | validation | 4183 | 240 | 5.7% |
| 30 | test | 4566 | 232 | 5.1% |
| 30 | validation | 4534 | 261 | 5.8% |

All stays of the graph artifact (before the class filter): test 432/10071 = 4.29%; train 3466/80246 = 4.32%; validation 462/10038 = 4.60%.

## Accuracy: subset, rest and gap (test)

| K | method | subset | rest = overall without the subset | overall | gap subset - rest, points [95% CI] | overall - rest, points |
|---|---|---|---|---|---|---|
| 10 | CEI-GNN mp2 | 0.7467 | 0.7548 | 0.7543 | -0.8 [-6.9, +5.0] | -0.0 |
| 10 | GSAT | 0.7542 | 0.7460 | 0.7465 | +0.8 [-5.3, +6.5] | +0.0 |
| 10 | GraphCare-style (adapted) | 0.7598 | 0.7540 | 0.7543 | +0.6 [-5.2, +6.2] | +0.0 |
| 10 | ProtGNN | 0.7505 | 0.7448 | 0.7451 | +0.6 [-5.2, +6.4] | +0.0 |
| 10 | GCHM-PNA | 0.7542 | 0.7503 | 0.7506 | +0.4 [-5.4, +6.1] | +0.0 |
| 20 | CEI-GNN mp2 | 0.7136 | 0.7038 | 0.7043 | +1.0 [-4.4, +6.7] | +0.1 |
| 20 | GSAT | 0.7230 | 0.6827 | 0.6848 | +4.0 [-1.6, +9.5] | +0.2 |
| 20 | GraphCare-style (adapted) | 0.7121 | 0.6864 | 0.6877 | +2.6 [-3.4, +7.9] | +0.1 |
| 20 | ProtGNN | 0.7246 | 0.6830 | 0.6851 | +4.2 [-1.6, +9.5] | +0.2 |
| 20 | GCHM-PNA | 0.7293 | 0.6856 | 0.6878 | +4.4 [-1.4, +9.6] | +0.2 |
| 30 | CEI-GNN mp2 | 0.7112 | 0.6637 | 0.6662 | +4.7 [-1.2, +10.3] | +0.2 |
| 30 | GSAT | 0.7155 | 0.6447 | 0.6483 | +7.1 [+1.5, +12.6] | +0.4 |
| 30 | GraphCare-style (adapted) | 0.6940 | 0.6421 | 0.6447 | +5.2 [-0.7, +10.9] | +0.3 |
| 30 | ProtGNN | 0.6983 | 0.6444 | 0.6472 | +5.4 [-0.1, +11.1] | +0.3 |
| 30 | GCHM-PNA | 0.7055 | 0.6448 | 0.6479 | +6.1 [+0.7, +11.6] | +0.3 |

## Macro-F1: subset, rest and gap (test)

| K | method | subset | rest = overall without the subset | overall | gap subset - rest, points [95% CI] | overall - rest, points |
|---|---|---|---|---|---|---|
| 10 | CEI-GNN mp2 | 0.7203 | 0.7615 | 0.7612 | -4.1 [-13.1, +3.6] | -0.0 |
| 10 | GSAT | 0.7241 | 0.7505 | 0.7509 | -2.6 [-11.5, +4.5] | +0.0 |
| 10 | GraphCare-style (adapted) | 0.7198 | 0.7628 | 0.7629 | -4.3 [-13.8, +3.2] | +0.0 |
| 10 | ProtGNN | 0.7097 | 0.7510 | 0.7510 | -4.1 [-12.6, +3.3] | -0.0 |
| 10 | GCHM-PNA | 0.7218 | 0.7558 | 0.7559 | -3.4 [-12.1, +3.7] | +0.0 |
| 20 | CEI-GNN mp2 | 0.6564 | 0.6897 | 0.6894 | -3.3 [-13.6, +3.9] | -0.0 |
| 20 | GSAT | 0.6869 | 0.6676 | 0.6692 | +1.9 [-8.4, +8.8] | +0.2 |
| 20 | GraphCare-style (adapted) | 0.6539 | 0.6712 | 0.6714 | -1.7 [-12.1, +5.5] | +0.0 |
| 20 | ProtGNN | 0.6623 | 0.6644 | 0.6655 | -0.2 [-9.2, +7.0] | +0.1 |
| 20 | GCHM-PNA | 0.6734 | 0.6698 | 0.6712 | +0.4 [-9.4, +7.3] | +0.1 |
| 30 | CEI-GNN mp2 | 0.6569 | 0.5951 | 0.5981 | +6.2 [-1.8, +14.3] | +0.3 |
| 30 | GSAT | 0.6428 | 0.5733 | 0.5762 | +6.9 [-1.7, +16.1] | +0.3 |
| 30 | GraphCare-style (adapted) | 0.6409 | 0.5626 | 0.5656 | +7.8 [-0.7, +16.0] | +0.3 |
| 30 | ProtGNN | 0.6000 | 0.5741 | 0.5758 | +2.6 [-3.9, +11.8] | +0.2 |
| 30 | GCHM-PNA | 0.6267 | 0.5722 | 0.5748 | +5.4 [-2.9, +14.8] | +0.3 |

## Reproduction of the stored validation statement

Stored statement: +8.5 to +11 validation accuracy points on the subset for every method (about +0.6 overall), made for the K = 20 task. Accuracy gap per K over the methods:

| K | subset - rest accuracy gap, points (min to max over methods) | overall - rest accuracy, points |
|---|---|---|
| 10 | -0.8 to +0.8 | -0.05 to +0.05 |
| 20 | +1.0 to +4.4 | +0.05 to +0.22 |
| 30 | +4.7 to +7.1 | +0.24 to +0.36 |

K = 20, test fold: +1.0 to +4.4 points; inside the stored range (+8.5 to +11): no (the stored statement is a validation statement; this is the test fold).

Parity with the stored result.json (whole fold, 45 runs): max abs diff accuracy 4.98e-07, macro-F1 4.99e-07.

---

# Validation (decision fold): same analysis

Reported fold: validation (decision fold; labelled "validation"). Methods CEI-GNN mp2, GSAT, GraphCare-style (adapted), ProtGNN, GCHM-PNA; K 10, 20, 30; seeds 1234, 2025, 7 (45 stored runs). Subset = the cut-off of the stay equals the ED discharge time; rest = all other stays. Gap = subset minus rest in points (seed mean), interval = patient bootstrap (B=1000, shared subject draw, percentile 95%).

## Subset size

| K | fold | rows scored | rows in subset | share |
|---|---|---|---|---|
| 10 | test | 3175 | 179 | 5.6% |
| 10 | validation | 3165 | 197 | 6.2% |
| 20 | test | 4174 | 213 | 5.1% |
| 20 | validation | 4183 | 240 | 5.7% |
| 30 | test | 4566 | 232 | 5.1% |
| 30 | validation | 4534 | 261 | 5.8% |

All stays of the graph artifact (before the class filter): test 432/10071 = 4.29%; train 3466/80246 = 4.32%; validation 462/10038 = 4.60%.

## Accuracy: subset, rest and gap (validation)

| K | method | subset | rest = overall without the subset | overall | gap subset - rest, points [95% CI] | overall - rest, points |
|---|---|---|---|---|---|---|
| 10 | CEI-GNN mp2 | 0.8156 | 0.7361 | 0.7410 | +7.9 [+2.2, +13.1] | +0.5 |
| 10 | GSAT | 0.8122 | 0.7200 | 0.7258 | +9.2 [+3.9, +14.4] | +0.6 |
| 10 | GraphCare-style (adapted) | 0.7970 | 0.7272 | 0.7315 | +7.0 [+1.5, +12.3] | +0.4 |
| 10 | ProtGNN | 0.7953 | 0.7288 | 0.7329 | +6.6 [+1.0, +11.8] | +0.4 |
| 10 | GCHM-PNA | 0.8206 | 0.7266 | 0.7325 | +9.4 [+4.0, +14.7] | +0.6 |
| 20 | CEI-GNN mp2 | 0.8000 | 0.6916 | 0.6978 | +10.8 [+6.1, +15.9] | +0.6 |
| 20 | GSAT | 0.7653 | 0.6793 | 0.6842 | +8.6 [+4.1, +14.0] | +0.5 |
| 20 | GraphCare-style (adapted) | 0.7750 | 0.6812 | 0.6866 | +9.4 [+4.7, +14.3] | +0.5 |
| 20 | ProtGNN | 0.7653 | 0.6801 | 0.6850 | +8.5 [+3.5, +13.5] | +0.5 |
| 20 | GCHM-PNA | 0.7903 | 0.6804 | 0.6867 | +11.0 [+6.5, +15.8] | +0.6 |
| 30 | CEI-GNN mp2 | 0.7484 | 0.6521 | 0.6576 | +9.6 [+4.5, +14.5] | +0.6 |
| 30 | GSAT | 0.7254 | 0.6396 | 0.6445 | +8.6 [+3.6, +13.7] | +0.5 |
| 30 | GraphCare-style (adapted) | 0.7356 | 0.6397 | 0.6452 | +9.6 [+4.6, +14.6] | +0.6 |
| 30 | ProtGNN | 0.7331 | 0.6419 | 0.6471 | +9.1 [+3.8, +14.3] | +0.5 |
| 30 | GCHM-PNA | 0.7382 | 0.6429 | 0.6484 | +9.5 [+4.5, +14.4] | +0.5 |

## Macro-F1: subset, rest and gap (validation)

| K | method | subset | rest = overall without the subset | overall | gap subset - rest, points [95% CI] | overall - rest, points |
|---|---|---|---|---|---|---|
| 10 | CEI-GNN mp2 | 0.7593 | 0.7417 | 0.7451 | +1.8 [-5.3, +9.5] | +0.3 |
| 10 | GSAT | 0.7592 | 0.7221 | 0.7262 | +3.7 [-3.1, +10.7] | +0.4 |
| 10 | GraphCare-style (adapted) | 0.7418 | 0.7304 | 0.7332 | +1.1 [-6.6, +8.9] | +0.3 |
| 10 | ProtGNN | 0.7313 | 0.7314 | 0.7338 | -0.0 [-7.5, +7.5] | +0.2 |
| 10 | GCHM-PNA | 0.7710 | 0.7294 | 0.7336 | +4.2 [-2.6, +11.3] | +0.4 |
| 20 | CEI-GNN mp2 | 0.6946 | 0.6818 | 0.6859 | +1.3 [-4.5, +9.7] | +0.4 |
| 20 | GSAT | 0.6781 | 0.6692 | 0.6718 | +0.9 [-5.4, +9.0] | +0.3 |
| 20 | GraphCare-style (adapted) | 0.7107 | 0.6707 | 0.6742 | +4.0 [-1.8, +11.0] | +0.4 |
| 20 | ProtGNN | 0.6659 | 0.6712 | 0.6743 | -0.5 [-6.3, +7.0] | +0.3 |
| 20 | GCHM-PNA | 0.7075 | 0.6729 | 0.6769 | +3.5 [-2.5, +10.5] | +0.4 |
| 30 | CEI-GNN mp2 | 0.6047 | 0.5687 | 0.5708 | +3.6 [-4.1, +11.6] | +0.2 |
| 30 | GSAT | 0.5801 | 0.5523 | 0.5548 | +2.8 [-4.2, +11.3] | +0.2 |
| 30 | GraphCare-style (adapted) | 0.6056 | 0.5466 | 0.5491 | +5.9 [-1.2, +14.5] | +0.3 |
| 30 | ProtGNN | 0.5908 | 0.5612 | 0.5632 | +3.0 [-4.1, +10.7] | +0.2 |
| 30 | GCHM-PNA | 0.6125 | 0.5556 | 0.5602 | +5.7 [-1.9, +13.1] | +0.5 |

## Reproduction of the stored validation statement

Stored statement: +8.5 to +11 validation accuracy points on the subset for every method (about +0.6 overall), made for the K = 20 task. Accuracy gap per K over the methods:

| K | subset - rest accuracy gap, points (min to max over methods) | overall - rest accuracy, points |
|---|---|---|
| 10 | +6.6 to +9.4 | +0.41 to +0.59 |
| 20 | +8.5 to +11.0 | +0.49 to +0.63 |
| 30 | +8.6 to +9.6 | +0.49 to +0.55 |

K = 20, validation fold: +8.5 to +11.0 points; inside the stored range (+8.5 to +11): yes.

Parity with the stored result.json (whole fold, 45 runs): max abs diff accuracy 4.98e-07, macro-F1 4.83e-07.
