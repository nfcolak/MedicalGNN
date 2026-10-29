# Section-2 analysis

Protocol fair_v10t. Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run at the validation-selected checkpoint). K=20; reference `final`; arms: R1, R2, R3, R4, R5; methods: cei_mp2.
Test 4174 rows / 3151 patients; validation (decision fold) 4183 rows / 3142 patients.
Bootstrap B=2000: seeds (matched, with replacement), then patients (by subjects, rows paired by sample_ids; one patient resample per replicate shared by seeds and arms); mean over drawn seeds of macro-F1(arm) - macro-F1(reference); 95% percentile.
Reading rule (both folds): matters = drop (reference - arm) >= 0.005 and the seeds+patients CI excludes 0; no measurable effect = CI includes 0; small effect, not resolved = CI excludes 0 and drop < 0.005.

## Overview (delta = arm - reference, matched seeds)

| arm | method | seeds | test macro-F1 | test reading | d test [CI seeds+patients] | P(d>0) | CI patients only | validation macro-F1 (decision fold) | validation reading | d val [CI seeds+patients] | P(d>0) | CI patients only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final (ref) | cei_mp2 | 7,1234,2025 | 0.6894 ± 0.0013 | - | - | - | - | 0.6859 ± 0.0040 | - | - | - | - |
| R1 | cei_mp2 | 7,1234,2025 | 0.6805 ± 0.0035 | matters | -0.0089 [-0.0165, -0.0009] | 0.014 | [-0.0146, -0.0032] | 0.6829 ± 0.0044 | no measurable effect | -0.0030 [-0.0108, +0.0047] | 0.211 | [-0.0083, +0.0023] |
| R2 | cei_mp2 | 7,1234,2025 | 0.6793 ± 0.0025 | matters | -0.0100 [-0.0175, -0.0028] | 0.006 | [-0.0155, -0.0047] | 0.6820 ± 0.0028 | no measurable effect | -0.0039 [-0.0108, +0.0029] | 0.128 | [-0.0098, +0.0017] |
| R3 | cei_mp2 | 7,1234,2025 | 0.6816 ± 0.0035 | no measurable effect (not resolved at this precision) | -0.0078 [-0.0159, +0.0010] | 0.035 | [-0.0138, -0.0016] | 0.6840 ± 0.0024 | no measurable effect | -0.0018 [-0.0090, +0.0047] | 0.299 | [-0.0073, +0.0037] |
| R4 | cei_mp2 | 7,1234,2025 | 0.6896 ± 0.0041 | no measurable effect | +0.0002 [-0.0048, +0.0059] | 0.534 | [-0.0027, +0.0031] | 0.6863 ± 0.0050 | no measurable effect | +0.0004 [-0.0039, +0.0044] | 0.583 | [-0.0026, +0.0034] |
| R5 | cei_mp2 | 7,1234,2025 | 0.6904 ± 0.0061 | no measurable effect | +0.0010 [-0.0080, +0.0099] | 0.611 | [-0.0036, +0.0057] | 0.6860 ± 0.0046 | no measurable effect | +0.0001 [-0.0060, +0.0066] | 0.525 | [-0.0045, +0.0052] |

Intervals are conditional on the fixed test fold; the seed level rests on the matched seeds only (3 per arm here), so a CI that excludes 0 is not seed-robust proof: check per-seed deltas.

## final (reference) / cei_mp2

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6909 | 0.7056 | 0.6904 | 0.7016 | 38 |
| 1234 | 0.6889 | 0.7036 | 0.6843 | 0.6950 | 30 |
| 2025 | 0.6883 | 0.7036 | 0.6829 | 0.6969 | 40 |
| mean ± SD | 0.6894 ± 0.0013 | 0.7043 ± 0.0011 | 0.6859 ± 0.0040 | 0.6978 ± 0.0034 | 36.0 ± 5.3 |

## R1 / cei_mp2

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6783 | 0.6984 | 0.6856 | 0.6978 | 36 |
| 1234 | 0.6846 | 0.7017 | 0.6778 | 0.6942 | 40 |
| 2025 | 0.6786 | 0.6984 | 0.6852 | 0.6983 | 25 |
| mean ± SD | 0.6805 ± 0.0035 | 0.6995 ± 0.0019 | 0.6829 ± 0.0044 | 0.6968 ± 0.0022 | 33.7 ± 7.8 |

vs reference `final`, matched seeds [7, 1234, 2025]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 7 | -0.0126 | -0.0072 |
| test | 1234 | -0.0043 | -0.0019 |
| test | 2025 | -0.0097 | -0.0053 |
| test | mean | -0.0089 ± 0.0042 | -0.0048 ± 0.0027 |
| test | CI seeds+patients | [-0.0165, -0.0009] (P>0 0.014) | patients only [-0.0146, -0.0032] |
| validation (decision fold) | 7 | -0.0048 | -0.0038 |
| validation (decision fold) | 1234 | -0.0065 | -0.0007 |
| validation (decision fold) | 2025 | +0.0023 | +0.0014 |
| validation (decision fold) | mean | -0.0030 ± 0.0047 | -0.0010 ± 0.0026 |
| validation (decision fold) | CI seeds+patients | [-0.0108, +0.0047] (P>0 0.211) | patients only [-0.0083, +0.0023] |

## R2 / cei_mp2

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6769 | 0.6957 | 0.6853 | 0.6973 | 27 |
| 1234 | 0.6792 | 0.6960 | 0.6803 | 0.6959 | 30 |
| 2025 | 0.6819 | 0.7005 | 0.6805 | 0.6966 | 28 |
| mean ± SD | 0.6793 ± 0.0025 | 0.6974 ± 0.0027 | 0.6820 ± 0.0028 | 0.6966 ± 0.0007 | 28.3 ± 1.5 |

vs reference `final`, matched seeds [7, 1234, 2025]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 7 | -0.0140 | -0.0098 |
| test | 1234 | -0.0097 | -0.0077 |
| test | 2025 | -0.0064 | -0.0031 |
| test | mean | -0.0100 ± 0.0038 | -0.0069 ± 0.0034 |
| test | CI seeds+patients | [-0.0175, -0.0028] (P>0 0.006) | patients only [-0.0155, -0.0047] |
| validation (decision fold) | 7 | -0.0051 | -0.0043 |
| validation (decision fold) | 1234 | -0.0040 | +0.0010 |
| validation (decision fold) | 2025 | -0.0024 | -0.0002 |
| validation (decision fold) | mean | -0.0039 ± 0.0013 | -0.0012 ± 0.0028 |
| validation (decision fold) | CI seeds+patients | [-0.0108, +0.0029] (P>0 0.128) | patients only [-0.0098, +0.0017] |

## R3 / cei_mp2

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6776 | 0.6972 | 0.6867 | 0.6981 | 27 |
| 1234 | 0.6840 | 0.7010 | 0.6824 | 0.6959 | 40 |
| 2025 | 0.6833 | 0.7012 | 0.6830 | 0.6976 | 24 |
| mean ± SD | 0.6816 ± 0.0035 | 0.6998 ± 0.0023 | 0.6840 ± 0.0024 | 0.6972 ± 0.0011 | 30.3 ± 8.5 |

vs reference `final`, matched seeds [7, 1234, 2025]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 7 | -0.0133 | -0.0084 |
| test | 1234 | -0.0049 | -0.0026 |
| test | 2025 | -0.0050 | -0.0024 |
| test | mean | -0.0078 ± 0.0048 | -0.0045 ± 0.0034 |
| test | CI seeds+patients | [-0.0159, +0.0010] (P>0 0.035) | patients only [-0.0138, -0.0016] |
| validation (decision fold) | 7 | -0.0036 | -0.0036 |
| validation (decision fold) | 1234 | -0.0020 | +0.0010 |
| validation (decision fold) | 2025 | +0.0001 | +0.0007 |
| validation (decision fold) | mean | -0.0018 ± 0.0019 | -0.0006 ± 0.0026 |
| validation (decision fold) | CI seeds+patients | [-0.0090, +0.0047] (P>0 0.299) | patients only [-0.0073, +0.0037] |

## R4 / cei_mp2

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6920 | 0.7058 | 0.6920 | 0.7024 | 38 |
| 1234 | 0.6920 | 0.7048 | 0.6842 | 0.6950 | 36 |
| 2025 | 0.6848 | 0.7024 | 0.6826 | 0.6959 | 39 |
| mean ± SD | 0.6896 ± 0.0041 | 0.7044 ± 0.0017 | 0.6863 ± 0.0050 | 0.6977 ± 0.0040 | 37.7 ± 1.5 |

vs reference `final`, matched seeds [7, 1234, 2025]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 7 | +0.0012 | +0.0002 |
| test | 1234 | +0.0031 | +0.0012 |
| test | 2025 | -0.0035 | -0.0012 |
| test | mean | 0.0002 ± 0.0034 | 0.0001 ± 0.0012 |
| test | CI seeds+patients | [-0.0048, +0.0059] (P>0 0.534) | patients only [-0.0027, +0.0031] |
| validation (decision fold) | 7 | +0.0016 | +0.0007 |
| validation (decision fold) | 1234 | -0.0001 | +0.0000 |
| validation (decision fold) | 2025 | -0.0003 | -0.0010 |
| validation (decision fold) | mean | 0.0004 ± 0.0011 | -0.0001 ± 0.0008 |
| validation (decision fold) | CI seeds+patients | [-0.0039, +0.0044] (P>0 0.583) | patients only [-0.0026, +0.0034] |

## R5 / cei_mp2

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6834 | 0.7005 | 0.6911 | 0.7016 | 32 |
| 1234 | 0.6941 | 0.7070 | 0.6819 | 0.6957 | 34 |
| 2025 | 0.6937 | 0.7077 | 0.6851 | 0.6981 | 40 |
| mean ± SD | 0.6904 ± 0.0061 | 0.7051 ± 0.0040 | 0.6860 ± 0.0046 | 0.6985 ± 0.0030 | 35.3 ± 4.2 |

vs reference `final`, matched seeds [7, 1234, 2025]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 7 | -0.0075 | -0.0050 |
| test | 1234 | +0.0052 | +0.0034 |
| test | 2025 | +0.0053 | +0.0041 |
| test | mean | 0.0010 ± 0.0074 | 0.0008 ± 0.0051 |
| test | CI seeds+patients | [-0.0080, +0.0099] (P>0 0.611) | patients only [-0.0036, +0.0057] |
| validation (decision fold) | 7 | +0.0007 | +0.0000 |
| validation (decision fold) | 1234 | -0.0024 | +0.0007 |
| validation (decision fold) | 2025 | +0.0022 | +0.0012 |
| validation (decision fold) | mean | 0.0001 ± 0.0023 | 0.0006 ± 0.0006 |
| validation (decision fold) | CI seeds+patients | [-0.0060, +0.0066] (P>0 0.525) | patients only [-0.0045, +0.0052] |

