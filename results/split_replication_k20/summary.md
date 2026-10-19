# Split replication: arms vs Full

Protocol fair_v10t, K=20, reference `a1_full`, arms a5_nomp, a5b_nomp_matched, a6_pool, seeds 7,42,1234,2025,3407, splits r0, r1, r2 (canonical layout, no sidecar: r0). Decisions use VALIDATION; test reported only, not used for decisions.
Bootstrap B=2000. Per split: arms_vs_ref two-level: matched seeds, then patients (rows paired by sample_ids). Combined: per split independent seed picks and decoupled patient resamples, then the mean over splits; 95% percentile. Split level: SD of the per-split mean deltas (ddof=1) and t interval, df = n_splits - 1.
Reading rule: matters: drop (Full - arm) >= 0.005 and CI excludes 0; no measurable effect: CI includes 0 (note when drop >= 0.005); small effect, not resolved: CI excludes 0 and drop < 0.005 (PREREGISTRATION.md).


## Kept classes per split (top-K by TRAIN counts)

- r0: [0, 1, 3, 4, 6, 9, 11, 12, 13, 14, 15, 19, 20, 21, 23, 24, 25, 26, 27, 29]
- r1: [0, 1, 3, 4, 6, 9, 11, 12, 13, 14, 15, 19, 20, 21, 23, 24, 25, 26, 27, 29]
- r2: [0, 1, 3, 4, 6, 9, 11, 12, 13, 14, 15, 19, 20, 21, 23, 24, 25, 26, 27, 29]

## Per split (delta = arm - Full, matched seeds)

| split | arm | d val macro-F1 [CI seeds+patients] | P(d>0) | CI patients only | reading | note | d test [CI] (reported only) |
|---|---|---|---|---|---|---|---|
| r0 | a5_nomp | -0.0047 [-0.0114, +0.0017] | 0.085 | [-0.0096, +0.0002] | no measurable effect |  | -0.0001 [-0.0067, +0.0064] |
| r0 | a5b_nomp_matched | -0.0101 [-0.0177, -0.0027] | 0.004 | [-0.0153, -0.0051] | matters |  | -0.0049 [-0.0120, +0.0017] |
| r0 | a6_pool | -0.0055 [-0.0139, +0.0032] | 0.101 | [-0.0118, +0.0012] | no measurable effect | not resolved at this precision | -0.0059 [-0.0136, +0.0017] |
| r1 | a5_nomp | -0.0021 [-0.0081, +0.0041] | 0.262 | [-0.0068, +0.0025] | no measurable effect |  | +0.0011 [-0.0062, +0.0082] |
| r1 | a5b_nomp_matched | -0.0099 [-0.0171, -0.0032] | 0.004 | [-0.0151, -0.0047] | matters |  | -0.0066 [-0.0147, +0.0020] |
| r1 | a6_pool | -0.0031 [-0.0106, +0.0049] | 0.215 | [-0.0096, +0.0038] | no measurable effect |  | -0.0082 [-0.0173, +0.0008] |
| r2 | a5_nomp | -0.0019 [-0.0080, +0.0042] | 0.248 | [-0.0065, +0.0028] | no measurable effect |  | -0.0051 [-0.0111, +0.0014] |
| r2 | a5b_nomp_matched | -0.0078 [-0.0143, -0.0016] | 0.005 | [-0.0127, -0.0029] | matters |  | -0.0147 [-0.0222, -0.0072] |
| r2 | a6_pool | -0.0049 [-0.0126, +0.0032] | 0.114 | [-0.0113, +0.0017] | no measurable effect |  | -0.0082 [-0.0158, -0.0003] |

## Combined: primary r1+r2

| arm | mean d val | CI seeds+patients (stratified) | P(d>0) | CI patients only | between-split SD | t interval over splits | reading | note | mean d test (reported only) |
|---|---|---|---|---|---|---|---|---|---|
| a5_nomp | -0.0020 | [-0.0062, +0.0023] | 0.185 | [-0.0053, +0.0015] | 0.0001 | [-0.0029, -0.0010] (df 1) | no measurable effect |  | -0.0020 |
| a5b_nomp_matched | -0.0088 | [-0.0136, -0.0041] | 0.001 | [-0.0124, -0.0053] | 0.0015 | [-0.0221, +0.0044] (df 1) | matters |  | -0.0107 |
| a6_pool | -0.0040 | [-0.0094, +0.0018] | 0.081 | [-0.0086, +0.0007] | 0.0013 | [-0.0155, +0.0075] (df 1) | no measurable effect |  | -0.0082 |

Per-split mean d val: a5_nomp: r1 -0.0021, r2 -0.0019; a5b_nomp_matched: r1 -0.0099, r2 -0.0078; a6_pool: r1 -0.0031, r2 -0.0049

## Combined: secondary r0+r1+r2

| arm | mean d val | CI seeds+patients (stratified) | P(d>0) | CI patients only | between-split SD | t interval over splits | reading | note | mean d test (reported only) |
|---|---|---|---|---|---|---|---|---|---|
| a5_nomp | -0.0029 | [-0.0064, +0.0007] | 0.053 | [-0.0057, -0.0002] | 0.0016 | [-0.0068, +0.0010] (df 2) | no measurable effect |  | -0.0014 |
| a5b_nomp_matched | -0.0093 | [-0.0133, -0.0052] | 0.000 | [-0.0124, -0.0063] | 0.0013 | [-0.0124, -0.0061] (df 2) | matters |  | -0.0088 |
| a6_pool | -0.0045 | [-0.0093, +0.0003] | 0.033 | [-0.0085, -0.0006] | 0.0012 | [-0.0076, -0.0014] (df 2) | no measurable effect |  | -0.0074 |

Per-split mean d val: a5_nomp: r0 -0.0047, r1 -0.0021, r2 -0.0019; a5b_nomp_matched: r0 -0.0101, r1 -0.0099, r2 -0.0078; a6_pool: r0 -0.0055, r1 -0.0031, r2 -0.0049

