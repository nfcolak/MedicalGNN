# CEI-GNN mp2 against each baseline: patient-only and two-level intervals

Final runs, protocol fair_v10t; seeds 1234, 2025, 7; methods CEI-GNN mp2, GraphCare-style (adapted), GSAT, ProtGNN. Validation decides; test fold: reported only, not used for decisions.

Delta = CEI-GNN mp2 minus baseline in macro-F1 (mean over seeds). Patient-only: the interval of comparison/top3/compare.py (B=1000, rng 2026, resampling patients, seeds fixed). Two-level: seeds resampled per method (level 1) and the same patient draw (level 2), B=2000. SE ratio = two-level SE / patient-only SE. Bonferroni: two-level interval at level 1 - 0.05/9 (9 = 3 baselines x 3 K), labelled as such.

## K=10, validation fold (decision fold): 3165 rows, 2473 patients, seeds 1234, 2025, 7

| baseline | delta | patient-only 95% CI | P(d>0) | two-level 95% CI | P(d>0) | SE ratio | Bonferroni CI (1-0.05/9) |
|---|---|---|---|---|---|---|---|
| GraphCare-style (adapted) | 0.0119 | [+0.0038, +0.0202] | 0.9960 | [+0.0017, +0.0216] | 0.9865 | 1.2381 | [-0.0026, +0.0251] |
| GSAT | 0.0189 | [+0.0112, +0.0265] | 1.0000 | [+0.0095, +0.0286] | 0.9995 | 1.2505 | [+0.0052, +0.0331] |
| ProtGNN (best by validation) | 0.0113 | [+0.0039, +0.0186] | 0.9990 | [+0.0009, +0.0212] | 0.9840 | 1.3493 | [-0.0026, +0.0252] |
| mean of the baselines | 0.0140 | [+0.0079, +0.0203] | 1.0000 | [+0.0060, +0.0217] | 0.9990 | 1.2965 | [+0.0026, +0.0258] |

## K=10, test fold (test fold: reported only, not used for decisions): 3175 rows, 2497 patients, seeds 1234, 2025, 7

| baseline | delta | patient-only 95% CI | P(d>0) | two-level 95% CI | P(d>0) | SE ratio | Bonferroni CI (1-0.05/9) |
|---|---|---|---|---|---|---|---|
| GraphCare-style (adapted) | -0.0017 | [-0.0095, +0.0066] | 0.3400 | [-0.0145, +0.0118] | 0.3945 | 1.5849 | [-0.0196, +0.0160] |
| GSAT | 0.0103 | [+0.0031, +0.0178] | 0.9980 | [-0.0025, +0.0234] | 0.9455 | 1.6751 | [-0.0080, +0.0292] |
| ProtGNN (best by validation) | 0.0102 | [+0.0024, +0.0178] | 0.9970 | [-0.0029, +0.0238] | 0.9305 | 1.8053 | [-0.0072, +0.0302] |
| mean of the baselines | 0.0063 | [+0.0001, +0.0125] | 0.9760 | [-0.0051, +0.0177] | 0.8660 | 1.8302 | [-0.0084, +0.0230] |

## K=20, validation fold (decision fold): 4183 rows, 3142 patients, seeds 1234, 2025, 7

| baseline | delta | patient-only 95% CI | P(d>0) | two-level 95% CI | P(d>0) | SE ratio | Bonferroni CI (1-0.05/9) |
|---|---|---|---|---|---|---|---|
| GraphCare-style (adapted) | 0.0116 | [+0.0034, +0.0198] | 0.9950 | [+0.0017, +0.0218] | 0.9880 | 1.2001 | [-0.0025, +0.0255] |
| GSAT | 0.0141 | [+0.0059, +0.0225] | 1.0000 | [+0.0040, +0.0241] | 0.9955 | 1.2224 | [-0.0009, +0.0291] |
| ProtGNN (best by validation) | 0.0115 | [+0.0039, +0.0195] | 0.9990 | [+0.0011, +0.0226] | 0.9855 | 1.3473 | [-0.0028, +0.0264] |
| mean of the baselines | 0.0124 | [+0.0059, +0.0192] | 1.0000 | [+0.0036, +0.0210] | 0.9985 | 1.2387 | [+0.0006, +0.0243] |

## K=20, test fold (test fold: reported only, not used for decisions): 4174 rows, 3151 patients, seeds 1234, 2025, 7

| baseline | delta | patient-only 95% CI | P(d>0) | two-level 95% CI | P(d>0) | SE ratio | Bonferroni CI (1-0.05/9) |
|---|---|---|---|---|---|---|---|
| GraphCare-style (adapted) | 0.0180 | [+0.0102, +0.0258] | 1.0000 | [+0.0077, +0.0286] | 0.9995 | 1.3095 | [+0.0036, +0.0332] |
| GSAT | 0.0202 | [+0.0117, +0.0290] | 1.0000 | [+0.0090, +0.0302] | 0.9990 | 1.2401 | [+0.0039, +0.0344] |
| ProtGNN (best by validation) | 0.0239 | [+0.0155, +0.0324] | 1.0000 | [+0.0137, +0.0340] | 1.0000 | 1.2260 | [+0.0084, +0.0381] |
| mean of the baselines | 0.0207 | [+0.0136, +0.0273] | 1.0000 | [+0.0123, +0.0289] | 1.0000 | 1.2164 | [+0.0082, +0.0326] |

## K=30, validation fold (decision fold): 4534 rows, 3388 patients, seeds 1234, 2025, 7

| baseline | delta | patient-only 95% CI | P(d>0) | two-level 95% CI | P(d>0) | SE ratio | Bonferroni CI (1-0.05/9) |
|---|---|---|---|---|---|---|---|
| GraphCare-style (adapted) | 0.0217 | [+0.0096, +0.0341] | 1.0000 | [+0.0080, +0.0361] | 0.9990 | 1.1318 | [+0.0026, +0.0432] |
| GSAT | 0.0160 | [+0.0064, +0.0254] | 1.0000 | [+0.0044, +0.0288] | 0.9965 | 1.2460 | [-0.0003, +0.0352] |
| ProtGNN (best by validation) | 0.0076 | [-0.0029, +0.0188] | 0.9280 | [-0.0091, +0.0230] | 0.8680 | 1.4525 | [-0.0197, +0.0291] |
| mean of the baselines | 0.0151 | [+0.0069, +0.0242] | 1.0000 | [+0.0045, +0.0265] | 0.9960 | 1.2160 | [-0.0004, +0.0326] |

## K=30, test fold (test fold: reported only, not used for decisions): 4566 rows, 3404 patients, seeds 1234, 2025, 7

| baseline | delta | patient-only 95% CI | P(d>0) | two-level 95% CI | P(d>0) | SE ratio | Bonferroni CI (1-0.05/9) |
|---|---|---|---|---|---|---|---|
| GraphCare-style (adapted) | 0.0314 | [+0.0213, +0.0429] | 1.0000 | [+0.0187, +0.0444] | 1.0000 | 1.2087 | [+0.0131, +0.0508] |
| GSAT | 0.0212 | [+0.0098, +0.0347] | 1.0000 | [+0.0067, +0.0379] | 0.9965 | 1.2390 | [-0.0002, +0.0450] |
| ProtGNN (best by validation) | 0.0215 | [+0.0111, +0.0318] | 1.0000 | [+0.0087, +0.0337] | 0.9995 | 1.2176 | [+0.0042, +0.0387] |
| mean of the baselines | 0.0247 | [+0.0158, +0.0347] | 1.0000 | [+0.0139, +0.0360] | 1.0000 | 1.1858 | [+0.0098, +0.0409] |

