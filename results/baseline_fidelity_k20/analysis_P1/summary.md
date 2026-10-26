# Section-2 analysis

Protocol fair_v10t. Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run at the validation-selected checkpoint). K=20; reference `final`; arms: P1; methods: protgnn.
Test 4174 rows / 3151 patients; validation (decision fold) 4183 rows / 3142 patients.
Bootstrap B=2000: seeds (matched, with replacement), then patients (by subjects, rows paired by sample_ids; one patient resample per replicate shared by seeds and arms); mean over drawn seeds of macro-F1(arm) - macro-F1(reference); 95% percentile.
Reading rule (both folds): matters = drop (reference - arm) >= 0.005 and the seeds+patients CI excludes 0; no measurable effect = CI includes 0; small effect, not resolved = CI excludes 0 and drop < 0.005.

## Overview (delta = arm - reference, matched seeds)

| arm | method | seeds | test macro-F1 | test reading | d test [CI seeds+patients] | P(d>0) | CI patients only | validation macro-F1 (decision fold) | validation reading | d val [CI seeds+patients] | P(d>0) | CI patients only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final (ref) | protgnn | 7,1234,2025 | 0.6655 ± 0.0023 | - | - | - | - | 0.6743 ± 0.0041 | - | - | - | - |
| P1 | protgnn | 7,1234,2025 | 0.6721 ± 0.0016 | no measurable effect | +0.0066 [-0.0018, +0.0150] | 0.942 | [+0.0001, +0.0131] | 0.6748 ± 0.0088 | no measurable effect | +0.0005 [-0.0083, +0.0100] | 0.552 | [-0.0056, +0.0068] |

Intervals are conditional on the fixed test fold; the seed level rests on the matched seeds only (3 per arm here), so a CI that excludes 0 is not seed-robust proof: check per-seed deltas.

## final (reference) / protgnn

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6637 | 0.6818 | 0.6774 | 0.6875 | 31 |
| 1234 | 0.6646 | 0.6852 | 0.6759 | 0.6828 | 8 |
| 2025 | 0.6681 | 0.6883 | 0.6697 | 0.6847 | 30 |
| mean ± SD | 0.6655 ± 0.0023 | 0.6851 ± 0.0032 | 0.6743 ± 0.0041 | 0.6850 ± 0.0024 | 23.0 ± 13.0 |

## P1 / protgnn

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6703 | 0.6902 | 0.6825 | 0.6954 | 25 |
| 1234 | 0.6727 | 0.6866 | 0.6767 | 0.6830 | 23 |
| 2025 | 0.6734 | 0.6895 | 0.6653 | 0.6804 | 28 |
| mean ± SD | 0.6721 ± 0.0016 | 0.6888 ± 0.0019 | 0.6748 ± 0.0088 | 0.6863 ± 0.0080 | 25.3 ± 2.5 |

vs reference `final`, matched seeds [7, 1234, 2025]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 7 | +0.0066 | +0.0084 |
| test | 1234 | +0.0081 | +0.0014 |
| test | 2025 | +0.0053 | +0.0012 |
| test | mean | 0.0066 ± 0.0014 | 0.0037 ± 0.0041 |
| test | CI seeds+patients | [-0.0018, +0.0150] (P>0 0.942) | patients only [+0.0001, +0.0131] |
| validation (decision fold) | 7 | +0.0051 | +0.0079 |
| validation (decision fold) | 1234 | +0.0008 | +0.0002 |
| validation (decision fold) | 2025 | -0.0044 | -0.0043 |
| validation (decision fold) | mean | 0.0005 ± 0.0048 | 0.0013 ± 0.0062 |
| validation (decision fold) | CI seeds+patients | [-0.0083, +0.0100] (P>0 0.552) | patients only [-0.0056, +0.0068] |

