# Split replication: arms vs Full

Protocol fair_v10t, K=20, reference `a1_full`, arms a5_nomp, a5b_nomp_matched, a6_pool, seeds 7,42,1234,2025,3407, splits r0, r1, r2 (canonical layout, no sidecar: r0). Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run at the validation-selected checkpoint).
Bootstrap B=2000. Per split: arms_vs_ref two-level: matched seeds, then patients (rows paired by sample_ids). Combined: per split independent seed picks and decoupled patient resamples, then the mean over splits; 95% percentile. Split level: SD of the per-split mean deltas (ddof=1) and t interval, df = n_splits - 1.
Reading rule: applied on the test fold and, as the decision column, on validation. matters: drop (Full - arm) >= 0.005 and CI excludes 0; no measurable effect: CI includes 0 (note when drop >= 0.005); small effect, not resolved: CI excludes 0 and drop < 0.005.


## Kept classes per split (top-K by TRAIN counts)

- r0: [0, 1, 3, 4, 6, 9, 11, 12, 13, 14, 15, 19, 20, 21, 23, 24, 25, 26, 27, 29]
- r1: [0, 1, 3, 4, 6, 9, 11, 12, 13, 14, 15, 19, 20, 21, 23, 24, 25, 26, 27, 29]
- r2: [0, 1, 3, 4, 6, 9, 11, 12, 13, 14, 15, 19, 20, 21, 23, 24, 25, 26, 27, 29]

## Full macro-F1 per split (mean ± SD over seeds)

| split | test | validation (decision fold) |
|---|---|---|
| r0 | 0.6891 ± 0.0011 | 0.6861 ± 0.0031 |
| r1 | 0.6916 ± 0.0050 | 0.6989 ± 0.0014 |
| r2 | 0.6857 ± 0.0025 | 0.6966 ± 0.0008 |

## Per split, test; delta = arm - Full, matched seeds

| split | arm | reading | note | d test macro-F1 [CI seeds+patients] | P(d>0) | CI patients only |
|---|---|---|---|---|---|---|
| r0 | a5_nomp | no measurable effect |  | -0.0001 [-0.0067, +0.0064] | 0.476 | [-0.0053, +0.0054] |
| r0 | a5b_nomp_matched | no measurable effect |  | -0.0049 [-0.0120, +0.0017] | 0.081 | [-0.0109, +0.0007] |
| r0 | a6_pool | no measurable effect | not resolved at this precision | -0.0059 [-0.0136, +0.0017] | 0.069 | [-0.0123, +0.0007] |
| r1 | a5_nomp | no measurable effect |  | +0.0011 [-0.0062, +0.0082] | 0.609 | [-0.0036, +0.0058] |
| r1 | a5b_nomp_matched | no measurable effect | not resolved at this precision | -0.0066 [-0.0147, +0.0020] | 0.065 | [-0.0117, -0.0015] |
| r1 | a6_pool | no measurable effect | not resolved at this precision | -0.0082 [-0.0173, +0.0008] | 0.035 | [-0.0145, -0.0018] |
| r2 | a5_nomp | no measurable effect | not resolved at this precision | -0.0051 [-0.0111, +0.0014] | 0.063 | [-0.0094, -0.0005] |
| r2 | a5b_nomp_matched | matters |  | -0.0147 [-0.0222, -0.0072] | 0.001 | [-0.0202, -0.0093] |
| r2 | a6_pool | matters |  | -0.0082 [-0.0158, -0.0003] | 0.022 | [-0.0143, -0.0019] |

## Per split, validation (decision fold); delta = arm - Full, matched seeds

| split | arm | reading | note | d val macro-F1 [CI seeds+patients] | P(d>0) | CI patients only |
|---|---|---|---|---|---|---|
| r0 | a5_nomp | no measurable effect |  | -0.0047 [-0.0114, +0.0017] | 0.085 | [-0.0096, +0.0002] |
| r0 | a5b_nomp_matched | matters |  | -0.0101 [-0.0177, -0.0027] | 0.004 | [-0.0153, -0.0051] |
| r0 | a6_pool | no measurable effect | not resolved at this precision | -0.0055 [-0.0139, +0.0032] | 0.101 | [-0.0118, +0.0012] |
| r1 | a5_nomp | no measurable effect |  | -0.0021 [-0.0081, +0.0041] | 0.262 | [-0.0068, +0.0025] |
| r1 | a5b_nomp_matched | matters |  | -0.0099 [-0.0171, -0.0032] | 0.004 | [-0.0151, -0.0047] |
| r1 | a6_pool | no measurable effect |  | -0.0031 [-0.0106, +0.0049] | 0.215 | [-0.0096, +0.0038] |
| r2 | a5_nomp | no measurable effect |  | -0.0019 [-0.0080, +0.0042] | 0.248 | [-0.0065, +0.0028] |
| r2 | a5b_nomp_matched | matters |  | -0.0078 [-0.0143, -0.0016] | 0.005 | [-0.0127, -0.0029] |
| r2 | a6_pool | no measurable effect |  | -0.0049 [-0.0126, +0.0032] | 0.114 | [-0.0113, +0.0017] |

## Combined: primary r1+r2, test

| arm | reading | note | mean d test | CI seeds+patients (stratified) | P(d>0) | CI patients only | between-split SD | t interval over splits |
|---|---|---|---|---|---|---|---|---|
| a5_nomp | no measurable effect |  | -0.0020 | [-0.0066, +0.0032] | 0.213 | [-0.0051, +0.0014] | 0.0043 | [-0.0408, +0.0368] (df 1) |
| a5b_nomp_matched | matters |  | -0.0107 | [-0.0161, -0.0050] | 0.000 | [-0.0145, -0.0070] | 0.0057 | [-0.0620, +0.0407] (df 1) |
| a6_pool | matters |  | -0.0082 | [-0.0140, -0.0024] | 0.004 | [-0.0125, -0.0038] | 0.0001 | [-0.0088, -0.0076] (df 1) |

Per-split mean d test: a5_nomp: r1 +0.0011, r2 -0.0051; a5b_nomp_matched: r1 -0.0066, r2 -0.0147; a6_pool: r1 -0.0082, r2 -0.0082

## Combined: primary r1+r2, validation (decision fold)

| arm | reading | note | mean d val | CI seeds+patients (stratified) | P(d>0) | CI patients only | between-split SD | t interval over splits |
|---|---|---|---|---|---|---|---|---|
| a5_nomp | no measurable effect |  | -0.0020 | [-0.0062, +0.0023] | 0.185 | [-0.0053, +0.0015] | 0.0001 | [-0.0029, -0.0010] (df 1) |
| a5b_nomp_matched | matters |  | -0.0088 | [-0.0136, -0.0041] | 0.001 | [-0.0124, -0.0053] | 0.0015 | [-0.0221, +0.0044] (df 1) |
| a6_pool | no measurable effect |  | -0.0040 | [-0.0094, +0.0018] | 0.081 | [-0.0086, +0.0007] | 0.0013 | [-0.0155, +0.0075] (df 1) |

Per-split mean d val: a5_nomp: r1 -0.0021, r2 -0.0019; a5b_nomp_matched: r1 -0.0099, r2 -0.0078; a6_pool: r1 -0.0031, r2 -0.0049

## Combined: secondary r0+r1+r2, test

| arm | reading | note | mean d test | CI seeds+patients (stratified) | P(d>0) | CI patients only | between-split SD | t interval over splits |
|---|---|---|---|---|---|---|---|---|
| a5_nomp | no measurable effect |  | -0.0014 | [-0.0051, +0.0025] | 0.233 | [-0.0042, +0.0014] | 0.0032 | [-0.0094, +0.0067] (df 2) |
| a5b_nomp_matched | matters |  | -0.0088 | [-0.0133, -0.0041] | 0.000 | [-0.0120, -0.0057] | 0.0052 | [-0.0217, +0.0042] (df 2) |
| a6_pool | matters |  | -0.0074 | [-0.0123, -0.0025] | 0.001 | [-0.0110, -0.0038] | 0.0013 | [-0.0108, -0.0041] (df 2) |

Per-split mean d test: a5_nomp: r0 -0.0001, r1 +0.0011, r2 -0.0051; a5b_nomp_matched: r0 -0.0049, r1 -0.0066, r2 -0.0147; a6_pool: r0 -0.0059, r1 -0.0082, r2 -0.0082

## Combined: secondary r0+r1+r2, validation (decision fold)

| arm | reading | note | mean d val | CI seeds+patients (stratified) | P(d>0) | CI patients only | between-split SD | t interval over splits |
|---|---|---|---|---|---|---|---|---|
| a5_nomp | no measurable effect |  | -0.0029 | [-0.0064, +0.0007] | 0.053 | [-0.0057, -0.0002] | 0.0016 | [-0.0068, +0.0010] (df 2) |
| a5b_nomp_matched | matters |  | -0.0093 | [-0.0133, -0.0052] | 0.000 | [-0.0124, -0.0063] | 0.0013 | [-0.0124, -0.0061] (df 2) |
| a6_pool | no measurable effect |  | -0.0045 | [-0.0093, +0.0003] | 0.033 | [-0.0085, -0.0006] | 0.0012 | [-0.0076, -0.0014] (df 2) |

Per-split mean d val: a5_nomp: r0 -0.0047, r1 -0.0021, r2 -0.0019; a5b_nomp_matched: r0 -0.0101, r1 -0.0099, r2 -0.0078; a6_pool: r0 -0.0055, r1 -0.0031, r2 -0.0049

