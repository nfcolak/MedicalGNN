# Section-2 analysis

Protocol fair_v10t. Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run at the validation-selected checkpoint). K=20; reference `final`; arms: C1; methods: gchm_pna.
Test 4174 rows / 3151 patients; validation (decision fold) 4183 rows / 3142 patients.
Bootstrap B=2000: seeds (matched, with replacement), then patients (by subjects, rows paired by sample_ids; one patient resample per replicate shared by seeds and arms); mean over drawn seeds of macro-F1(arm) - macro-F1(reference); 95% percentile.
Reading rule (both folds): matters = drop (reference - arm) >= 0.005 and the seeds+patients CI excludes 0; no measurable effect = CI includes 0; small effect, not resolved = CI excludes 0 and drop < 0.005.

## Overview (delta = arm - reference, matched seeds)

| arm | method | seeds | test macro-F1 | test reading | d test [CI seeds+patients] | P(d>0) | CI patients only | validation macro-F1 (decision fold) | validation reading | d val [CI seeds+patients] | P(d>0) | CI patients only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final (ref) | gchm_pna | 7,1234,2025 | 0.6712 ± 0.0099 | - | - | - | - | 0.6769 ± 0.0026 | - | - | - | - |
| C1 | gchm_pna | 7,1234,2025 | 0.6757 ± 0.0047 | no measurable effect | +0.0045 [-0.0069, +0.0164] | 0.798 | [-0.0022, +0.0114] | 0.6785 ± 0.0021 | no measurable effect | +0.0016 [-0.0070, +0.0104] | 0.657 | [-0.0046, +0.0082] |

Intervals are conditional on the fixed test fold; the seed level rests on the matched seeds only (3 per arm here), so a CI that excludes 0 is not seed-robust proof: check per-seed deltas.

## final (reference) / gchm_pna

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6762 | 0.6895 | 0.6758 | 0.6844 | 22 |
| 1234 | 0.6598 | 0.6794 | 0.6799 | 0.6904 | 35 |
| 2025 | 0.6777 | 0.6945 | 0.6750 | 0.6852 | 18 |
| mean ± SD | 0.6712 ± 0.0099 | 0.6878 ± 0.0077 | 0.6769 ± 0.0026 | 0.6867 ± 0.0033 | 25.0 ± 8.9 |

## C1 / gchm_pna

| seed | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy | selected epoch |
|---|---|---|---|---|---|
| 7 | 0.6812 | 0.6950 | 0.6764 | 0.6899 | 28 |
| 1234 | 0.6729 | 0.6826 | 0.6785 | 0.6840 | 23 |
| 2025 | 0.6732 | 0.6838 | 0.6805 | 0.6856 | 17 |
| mean ± SD | 0.6757 ± 0.0047 | 0.6871 ± 0.0069 | 0.6785 ± 0.0021 | 0.6865 ± 0.0031 | 22.7 ± 5.5 |

vs reference `final`, matched seeds [7, 1234, 2025]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| test | 7 | +0.0050 | +0.0055 |
| test | 1234 | +0.0130 | +0.0031 |
| test | 2025 | -0.0045 | -0.0108 |
| test | mean | 0.0045 ± 0.0088 | -0.0007 ± 0.0088 |
| test | CI seeds+patients | [-0.0069, +0.0164] (P>0 0.798) | patients only [-0.0022, +0.0114] |
| validation (decision fold) | 7 | +0.0006 | +0.0055 |
| validation (decision fold) | 1234 | -0.0014 | -0.0065 |
| validation (decision fold) | 2025 | +0.0055 | +0.0005 |
| validation (decision fold) | mean | 0.0016 ± 0.0036 | -0.0002 ± 0.0060 |
| validation (decision fold) | CI seeds+patients | [-0.0070, +0.0104] (P>0 0.657) | patients only [-0.0046, +0.0082] |

