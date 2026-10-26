# Section-2 analysis

Protocol fair_v10t. Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run at the validation-selected checkpoint). K=20; reference `final`; arms: G_ib1.0_r0.5, G_ib1.0_r0.7, G_ib0.1_r0.5, G_ib0.1_r0.7, G_ib0.01_r0.5, G_ib0.01_r0.7; methods: gsat.
Test 4174 rows / 3151 patients; validation (decision fold) 4183 rows / 3142 patients.
Bootstrap B=2000: seeds (matched, with replacement), then patients (by subjects, rows paired by sample_ids; one patient resample per replicate shared by seeds and arms); mean over drawn seeds of macro-F1(arm) - macro-F1(reference); 95% percentile.
Reading rule (both folds): matters = drop (reference - arm) >= 0.005 and the seeds+patients CI excludes 0; no measurable effect = CI includes 0; small effect, not resolved = CI excludes 0 and drop < 0.005.

## Overview (delta = arm - reference, matched seeds)

| arm | method | seeds | test macro-F1 | test reading | d test [CI seeds+patients] | P(d>0) | CI patients only | validation macro-F1 (decision fold) | validation reading | d val [CI seeds+patients] | P(d>0) | CI patients only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final (ref) | gsat | 7,1234,2025 | 0.6692 ± 0.0041 | - | - | - | - | 0.6718 ± 0.0029 | - | - | - | - |
| G_ib1.0_r0.5 | gsat | 1234 | 0.6655 | no measurable effect (not resolved at this precision) | -0.0082 [-0.0200, +0.0030] | 0.076 | [-0.0200, +0.0030] | 0.6705 | no measurable effect | -0.0039 [-0.0154, +0.0070] | 0.253 | [-0.0154, +0.0070] |
| G_ib1.0_r0.7 | gsat | 1234 | 0.6714 | no measurable effect | -0.0023 [-0.0133, +0.0084] | 0.353 | [-0.0133, +0.0084] | 0.6727 | no measurable effect | -0.0017 [-0.0125, +0.0091] | 0.373 | [-0.0125, +0.0091] |
| G_ib0.1_r0.5 | gsat | 1234 | 0.6634 | no measurable effect (not resolved at this precision) | -0.0103 [-0.0213, +0.0002] | 0.028 | [-0.0213, +0.0002] | 0.6647 | no measurable effect (not resolved at this precision) | -0.0097 [-0.0206, +0.0005] | 0.030 | [-0.0206, +0.0005] |
| G_ib0.1_r0.7 | gsat | 1234 | 0.6597 | matters | -0.0140 [-0.0255, -0.0017] | 0.013 | [-0.0255, -0.0017] | 0.6675 | no measurable effect (not resolved at this precision) | -0.0069 [-0.0179, +0.0043] | 0.117 | [-0.0179, +0.0043] |
| G_ib0.01_r0.5 | gsat | 1234 | 0.4674 | matters | -0.2062 [-0.2238, -0.1890] | 0.000 | [-0.2238, -0.1890] | 0.4630 | matters | -0.2114 [-0.2286, -0.1950] | 0.000 | [-0.2286, -0.1950] |
| G_ib0.01_r0.7 | gsat | 1234 | 0.5813 | matters | -0.0924 [-0.1075, -0.0770] | 0.000 | [-0.1075, -0.0770] | 0.5597 | matters | -0.1146 [-0.1310, -0.0995] | 0.000 | [-0.1310, -0.0995] |

Intervals are conditional on the fixed test fold; the seed level rests on the matched seeds only (1 per arm here), so a CI that excludes 0 is not seed-robust proof: check per-seed deltas.

## final (reference) / gsat

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6683 | 0.6838 | 0.6687 | 0.6840 | 37 |
| 1234 | 0.6737 | 0.6897 | 0.6744 | 0.6852 | 32 |
| 2025 | 0.6657 | 0.6809 | 0.6723 | 0.6835 | 20 |
| mean ± SD | 0.6692 ± 0.0041 | 0.6848 ± 0.0045 | 0.6718 ± 0.0029 | 0.6842 ± 0.0009 | 29.7 ± 8.7 |

## G_ib1.0_r0.5 / gsat

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 1234 | 0.6655 | 0.6761 | 0.6705 | 0.6787 | 24 |
| mean ± SD | 0.6655 | 0.6761 | 0.6705 | 0.6787 | 24.0 |

vs reference `final`, matched seeds [1234]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 1234 | -0.0082 | -0.0137 |
| test | mean | -0.0082 | -0.0137 |
| test | CI seeds+patients | [-0.0200, +0.0030] (P>0 0.076) | patients only [-0.0200, +0.0030] |
| validation (decision fold) | 1234 | -0.0039 | -0.0065 |
| validation (decision fold) | mean | -0.0039 | -0.0065 |
| validation (decision fold) | CI seeds+patients | [-0.0154, +0.0070] (P>0 0.253) | patients only [-0.0154, +0.0070] |

## G_ib1.0_r0.7 / gsat

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 1234 | 0.6714 | 0.6806 | 0.6727 | 0.6761 | 24 |
| mean ± SD | 0.6714 | 0.6806 | 0.6727 | 0.6761 | 24.0 |

vs reference `final`, matched seeds [1234]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 1234 | -0.0023 | -0.0091 |
| test | mean | -0.0023 | -0.0091 |
| test | CI seeds+patients | [-0.0133, +0.0084] (P>0 0.353) | patients only [-0.0133, +0.0084] |
| validation (decision fold) | 1234 | -0.0017 | -0.0091 |
| validation (decision fold) | mean | -0.0017 | -0.0091 |
| validation (decision fold) | CI seeds+patients | [-0.0125, +0.0091] (P>0 0.373) | patients only [-0.0125, +0.0091] |

## G_ib0.1_r0.5 / gsat

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 1234 | 0.6634 | 0.6778 | 0.6647 | 0.6765 | 32 |
| mean ± SD | 0.6634 | 0.6778 | 0.6647 | 0.6765 | 32.0 |

vs reference `final`, matched seeds [1234]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 1234 | -0.0103 | -0.0120 |
| test | mean | -0.0103 | -0.0120 |
| test | CI seeds+patients | [-0.0213, +0.0002] (P>0 0.028) | patients only [-0.0213, +0.0002] |
| validation (decision fold) | 1234 | -0.0097 | -0.0086 |
| validation (decision fold) | mean | -0.0097 | -0.0086 |
| validation (decision fold) | CI seeds+patients | [-0.0206, +0.0005] (P>0 0.030) | patients only [-0.0206, +0.0005] |

## G_ib0.1_r0.7 / gsat

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 1234 | 0.6597 | 0.6775 | 0.6675 | 0.6777 | 22 |
| mean ± SD | 0.6597 | 0.6775 | 0.6675 | 0.6777 | 22.0 |

vs reference `final`, matched seeds [1234]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 1234 | -0.0140 | -0.0122 |
| test | mean | -0.0140 | -0.0122 |
| test | CI seeds+patients | [-0.0255, -0.0017] (P>0 0.013) | patients only [-0.0255, -0.0017] |
| validation (decision fold) | 1234 | -0.0069 | -0.0074 |
| validation (decision fold) | mean | -0.0069 | -0.0074 |
| validation (decision fold) | CI seeds+patients | [-0.0179, +0.0043] (P>0 0.117) | patients only [-0.0179, +0.0043] |

## G_ib0.01_r0.5 / gsat

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 1234 | 0.4674 | 0.4832 | 0.4630 | 0.4772 | 6 |
| mean ± SD | 0.4674 | 0.4832 | 0.4630 | 0.4772 | 6.0 |

vs reference `final`, matched seeds [1234]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 1234 | -0.2062 | -0.2065 |
| test | mean | -0.2062 | -0.2065 |
| test | CI seeds+patients | [-0.2238, -0.1890] (P>0 0.000) | patients only [-0.2238, -0.1890] |
| validation (decision fold) | 1234 | -0.2114 | -0.2080 |
| validation (decision fold) | mean | -0.2114 | -0.2080 |
| validation (decision fold) | CI seeds+patients | [-0.2286, -0.1950] (P>0 0.000) | patients only [-0.2286, -0.1950] |

## G_ib0.01_r0.7 / gsat

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 1234 | 0.5813 | 0.6018 | 0.5597 | 0.5855 | 6 |
| mean ± SD | 0.5813 | 0.6018 | 0.5597 | 0.5855 | 6.0 |

vs reference `final`, matched seeds [1234]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 1234 | -0.0924 | -0.0879 |
| test | mean | -0.0924 | -0.0879 |
| test | CI seeds+patients | [-0.1075, -0.0770] (P>0 0.000) | patients only [-0.1075, -0.0770] |
| validation (decision fold) | 1234 | -0.1146 | -0.0997 |
| validation (decision fold) | mean | -0.1146 | -0.0997 |
| validation (decision fold) | CI seeds+patients | [-0.1310, -0.0995] (P>0 0.000) | patients only [-0.1310, -0.0995] |

