# Class spread of the CEI-GNN mp2 gain

Final runs, protocol fair_v10t; seeds 1234, 2025, 7; methods CEI-GNN mp2, GraphCare-style (adapted), GSAT, ProtGNN. Validation decides; test fold: reported only, not used for decisions.

Best baseline per K = highest mean VALIDATION macro-F1 (applied to both folds). Classes with support < 30 in the fold are not evaluable. Top-3 share = share of the macro-F1 gap carried by the 3 classes with the largest CEI minus best-baseline F1 delta.

| K | fold | rows / patients | best baseline (validation) | macro gap vs best | macro gap vs mean of baselines | classes CEI > best | classes CEI > every baseline | not evaluable | top-3 share of gap |
|---|---|---|---|---|---|---|---|---|---|
| 10 | validation | 3165 / 2473 | ProtGNN | 0.0113 | 0.0140 | 8 of 10 | 7 of 10 | 0 | 0.6807 |
| 10 | test | 3175 / 2497 | ProtGNN | 0.0102 | 0.0063 | 9 of 10 | 5 of 10 | 0 | 0.5364 |
| 20 | validation | 4183 / 3142 | ProtGNN | 0.0115 | 0.0124 | 16 of 20 | 13 of 20 | 0 | 0.4610 |
| 20 | test | 4174 / 3151 | ProtGNN | 0.0239 | 0.0207 | 18 of 20 | 15 of 20 | 0 | 0.3208 |
| 30 | validation | 4534 / 3388 | ProtGNN | 0.0076 | 0.0151 | 18 of 30 | 11 of 30 | 3 | 0.7801 |
| 30 | test | 4566 / 3404 | ProtGNN | 0.0215 | 0.0247 | 23 of 30 | 19 of 30 | 3 | 0.3401 |

## K=10, validation fold (decision fold): top-3 classes and support terciles

| class | support | CEI F1 | best baseline F1 | delta |
|---|---|---|---|---|
| Heart failure | 163 | 0.6801 | 0.6438 | 0.0362 |
| Skin or soft-tissue infection | 325 | 0.6528 | 0.6318 | 0.0210 |
| Head injury | 245 | 0.7406 | 0.7211 | 0.0195 |

| support tercile | support range | classes | CEI mean F1 | best baseline mean F1 | mean delta |
|---|---|---|---|---|---|
| 1 | 158-293 | 4 | 0.7802 | 0.7622 | 0.0181 |
| 2 | 325-367 | 3 | 0.7352 | 0.7269 | 0.0083 |
| 3 | 395-481 | 3 | 0.7082 | 0.7030 | 0.0052 |

## K=10, test fold (test fold: reported only, not used for decisions): top-3 classes and support terciles

| class | support | CEI F1 | best baseline F1 | delta |
|---|---|---|---|---|
| Lower respiratory disease | 379 | 0.7299 | 0.7072 | 0.0227 |
| Head injury | 289 | 0.7871 | 0.7704 | 0.0167 |
| UTI or pyelonephritis | 385 | 0.7616 | 0.7460 | 0.0155 |

| support tercile | support range | classes | CEI mean F1 | best baseline mean F1 | mean delta |
|---|---|---|---|---|---|
| 1 | 141-289 | 4 | 0.7972 | 0.7856 | 0.0116 |
| 2 | 342-379 | 3 | 0.7576 | 0.7520 | 0.0057 |
| 3 | 385-456 | 3 | 0.7168 | 0.7039 | 0.0129 |

## K=20, validation fold (decision fold): top-3 classes and support terciles

| class | support | CEI F1 | best baseline F1 | delta |
|---|---|---|---|---|
| Alcohol abuse with intoxication | 102 | 0.7622 | 0.7173 | 0.0449 |
| Hypokalemia | 96 | 0.6086 | 0.5767 | 0.0319 |
| Anemia | 89 | 0.6460 | 0.6164 | 0.0295 |

| support tercile | support range | classes | CEI mean F1 | best baseline mean F1 | mean delta |
|---|---|---|---|---|---|
| 1 | 69-105 | 7 | 0.6775 | 0.6641 | 0.0134 |
| 2 | 111-293 | 7 | 0.6905 | 0.6843 | 0.0062 |
| 3 | 325-481 | 6 | 0.6902 | 0.6747 | 0.0155 |

## K=20, test fold (test fold: reported only, not used for decisions): top-3 classes and support terciles

| class | support | CEI F1 | best baseline F1 | delta |
|---|---|---|---|---|
| Non-ST elevation (NSTEMI) myocardial infarction | 85 | 0.7460 | 0.6786 | 0.0674 |
| Alcohol abuse with intoxication | 97 | 0.7292 | 0.6812 | 0.0479 |
| Heart failure | 141 | 0.6265 | 0.5886 | 0.0379 |

| support tercile | support range | classes | CEI mean F1 | best baseline mean F1 | mean delta |
|---|---|---|---|---|---|
| 1 | 71-115 | 7 | 0.6441 | 0.6070 | 0.0371 |
| 2 | 115-289 | 7 | 0.7207 | 0.7032 | 0.0174 |
| 3 | 342-456 | 6 | 0.7057 | 0.6897 | 0.0159 |

## K=30, validation fold (decision fold): top-3 classes and support terciles

| class | support | CEI F1 | best baseline F1 | delta |
|---|---|---|---|---|
| Acute upper respiratory infection | 36 | 0.3673 | 0.2846 | 0.0827 |
| Anemia | 89 | 0.6614 | 0.6132 | 0.0482 |
| Dehydration | 45 | 0.2939 | 0.2466 | 0.0474 |

| support tercile | support range | classes | CEI mean F1 | best baseline mean F1 | mean delta |
|---|---|---|---|---|---|
| 1 | 1-60 | 10 | 0.3814 | 0.3762 | 0.0052 |
| 2 | 69-145 | 10 | 0.6454 | 0.6397 | 0.0058 |
| 3 | 158-481 | 10 | 0.6855 | 0.6736 | 0.0119 |

## K=30, test fold (test fold: reported only, not used for decisions): top-3 classes and support terciles

| class | support | CEI F1 | best baseline F1 | delta |
|---|---|---|---|---|
| Acute kidney failure | 115 | 0.4486 | 0.3528 | 0.0957 |
| Hypokalemia | 102 | 0.5998 | 0.5296 | 0.0702 |
| Dehydration | 62 | 0.3226 | 0.2693 | 0.0533 |

| support tercile | support range | classes | CEI mean F1 | best baseline mean F1 | mean delta |
|---|---|---|---|---|---|
| 1 | 0-71 | 10 | 0.3657 | 0.3475 | 0.0182 |
| 2 | 72-126 | 10 | 0.6667 | 0.6370 | 0.0296 |
| 3 | 141-456 | 10 | 0.7020 | 0.6854 | 0.0166 |

