# Arms vs reference

Protocol fair_v10t (validation selection; test scored once, reported only, not used for decisions). K=20; reference `a1_full`; arms: a2_mlp, a3_noabsence, a4_nopair, a4b_nopair_noedge, a5_nomp, a5b_nomp_matched, a6_pool; methods: cei_mp2.
Validation 4183 rows / 3142 patients; test 4174 rows / 3151 patients.
Bootstrap B=2000: seeds (matched, with replacement), then patients (by subjects, rows paired by sample_ids; one patient resample per replicate shared by seeds and arms); mean over drawn seeds of macro-F1(arm) - macro-F1(reference); 95% percentile.

## Overview (delta = arm - reference, matched seeds)

| arm | method | seeds | val macro-F1 | d val [CI seeds+patients] | P(d>0) | CI patients only | test macro-F1 (reported only) | d test [CI] (reported only) |
|---|---|---|---|---|---|---|---|---|
| a1_full (ref) | cei_mp2 | 7,42,1234,2025,3407 | 0.6861 ± 0.0031 | - | - | - | 0.6891 ± 0.0011 | - |
| a2_mlp | cei_mp2 | 7,42,1234,2025,3407 | 0.6844 ± 0.0031 | -0.0017 [-0.0088, +0.0055] | 0.308 | [-0.0071, +0.0035] | 0.6883 ± 0.0038 | -0.0008 [-0.0080, +0.0064] |
| a3_noabsence | cei_mp2 | 7,42,1234,2025,3407 | 0.6863 ± 0.0046 | +0.0001 [-0.0046, +0.0041] | 0.526 | [-0.0029, +0.0032] | 0.6864 ± 0.0024 | -0.0027 [-0.0074, +0.0020] |
| a4_nopair | cei_mp2 | 7,42,1234,2025,3407 | 0.6856 ± 0.0012 | -0.0006 [-0.0066, +0.0051] | 0.433 | [-0.0044, +0.0032] | 0.6923 ± 0.0029 | +0.0032 [-0.0029, +0.0093] |
| a4b_nopair_noedge | cei_mp2 | 7,42,1234,2025,3407 | 0.6847 ± 0.0006 | -0.0015 [-0.0081, +0.0056] | 0.349 | [-0.0067, +0.0041] | 0.6862 ± 0.0050 | -0.0029 [-0.0107, +0.0055] |
| a5_nomp | cei_mp2 | 7,42,1234,2025,3407 | 0.6815 ± 0.0025 | -0.0047 [-0.0114, +0.0017] | 0.085 | [-0.0096, +0.0002] | 0.6890 ± 0.0025 | -0.0001 [-0.0067, +0.0064] |
| a5b_nomp_matched | cei_mp2 | 7,42,1234,2025,3407 | 0.6760 ± 0.0057 | -0.0101 [-0.0177, -0.0027] | 0.004 | [-0.0153, -0.0051] | 0.6841 ± 0.0025 | -0.0049 [-0.0120, +0.0017] |
| a6_pool | cei_mp2 | 7,42,1234,2025,3407 | 0.6807 ± 0.0018 | -0.0055 [-0.0139, +0.0032] | 0.101 | [-0.0118, +0.0012] | 0.6832 ± 0.0028 | -0.0059 [-0.0136, +0.0017] |

Intervals are conditional on the fixed evaluation fold; the seed level rests on the matched seeds only (3 for final-style arms), so a CI that excludes 0 is not seed-robust proof: check per-seed deltas.

## a1_full (reference) / cei_mp2

| seed | val macro-F1 | val accuracy | selected epoch | test macro-F1 (reported only) | test accuracy (reported only) |
|---|---|---|---|---|---|
| 7 | 0.6904 | 0.7016 | 38 | 0.6909 | 0.7056 |
| 42 | 0.6885 | 0.6969 | 22 | 0.6881 | 0.7020 |
| 1234 | 0.6843 | 0.6950 | 30 | 0.6889 | 0.7036 |
| 2025 | 0.6829 | 0.6969 | 40 | 0.6883 | 0.7036 |
| 3407 | 0.6846 | 0.6966 | 40 | 0.6891 | 0.7032 |
| mean ± SD | 0.6861 ± 0.0031 | 0.6974 ± 0.0025 | 34.0 ± 7.9 | 0.6891 ± 0.0011 | 0.7036 ± 0.0013 |

## a2_mlp / cei_mp2

| seed | val macro-F1 | val accuracy | selected epoch | test macro-F1 (reported only) | test accuracy (reported only) |
|---|---|---|---|---|---|
| 7 | 0.6842 | 0.6973 | 40 | 0.6831 | 0.6977 |
| 42 | 0.6879 | 0.6997 | 37 | 0.6860 | 0.6989 |
| 1234 | 0.6796 | 0.6957 | 37 | 0.6921 | 0.7063 |
| 2025 | 0.6860 | 0.6971 | 35 | 0.6886 | 0.7017 |
| 3407 | 0.6845 | 0.6957 | 33 | 0.6916 | 0.7027 |
| mean ± SD | 0.6844 ± 0.0031 | 0.6971 ± 0.0017 | 36.4 ± 2.6 | 0.6883 ± 0.0038 | 0.7014 ± 0.0034 |

vs reference `a1_full`, matched seeds [7, 42, 1234, 2025, 3407]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| validation | 7 | -0.0062 | -0.0043 |
| validation | 42 | -0.0006 | +0.0029 |
| validation | 1234 | -0.0047 | +0.0007 |
| validation | 2025 | +0.0031 | +0.0002 |
| validation | 3407 | -0.0001 | -0.0010 |
| validation | mean | -0.0017 ± 0.0037 | -0.0003 ± 0.0026 |
| validation | CI seeds+patients | [-0.0088, +0.0055] (P>0 0.308) | patients only [-0.0071, +0.0035] |
| test (reported only, not used for decisions) | 7 | -0.0077 | -0.0079 |
| test (reported only, not used for decisions) | 42 | -0.0021 | -0.0031 |
| test (reported only, not used for decisions) | 1234 | +0.0032 | +0.0026 |
| test (reported only, not used for decisions) | 2025 | +0.0003 | -0.0019 |
| test (reported only, not used for decisions) | 3407 | +0.0025 | -0.0005 |
| test (reported only, not used for decisions) | mean | -0.0008 ± 0.0044 | -0.0022 ± 0.0039 |
| test (reported only, not used for decisions) | CI seeds+patients | [-0.0080, +0.0064] (P>0 0.428) | patients only [-0.0061, +0.0044] |

## a3_noabsence / cei_mp2

| seed | val macro-F1 | val accuracy | selected epoch | test macro-F1 (reported only) | test accuracy (reported only) |
|---|---|---|---|---|---|
| 7 | 0.6921 | 0.7028 | 40 | 0.6835 | 0.7010 |
| 42 | 0.6895 | 0.6985 | 22 | 0.6857 | 0.7012 |
| 1234 | 0.6858 | 0.6973 | 36 | 0.6879 | 0.7017 |
| 2025 | 0.6807 | 0.6935 | 23 | 0.6852 | 0.7022 |
| 3407 | 0.6833 | 0.6971 | 36 | 0.6895 | 0.7034 |
| mean ± SD | 0.6863 ± 0.0046 | 0.6979 ± 0.0034 | 31.4 ± 8.3 | 0.6864 ± 0.0024 | 0.7019 ± 0.0009 |

vs reference `a1_full`, matched seeds [7, 42, 1234, 2025, 3407]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| validation | 7 | +0.0017 | +0.0012 |
| validation | 42 | +0.0010 | +0.0017 |
| validation | 1234 | +0.0015 | +0.0024 |
| validation | 2025 | -0.0022 | -0.0033 |
| validation | 3407 | -0.0013 | +0.0005 |
| validation | mean | 0.0001 ± 0.0018 | 0.0005 ± 0.0022 |
| validation | CI seeds+patients | [-0.0046, +0.0041] (P>0 0.526) | patients only [-0.0029, +0.0032] |
| test (reported only, not used for decisions) | 7 | -0.0074 | -0.0046 |
| test (reported only, not used for decisions) | 42 | -0.0024 | -0.0007 |
| test (reported only, not used for decisions) | 1234 | -0.0010 | -0.0019 |
| test (reported only, not used for decisions) | 2025 | -0.0032 | -0.0014 |
| test (reported only, not used for decisions) | 3407 | +0.0004 | +0.0002 |
| test (reported only, not used for decisions) | mean | -0.0027 ± 0.0029 | -0.0017 ± 0.0018 |
| test (reported only, not used for decisions) | CI seeds+patients | [-0.0074, +0.0020] (P>0 0.111) | patients only [-0.0057, +0.0001] |

## a4_nopair / cei_mp2

| seed | val macro-F1 | val accuracy | selected epoch | test macro-F1 (reported only) | test accuracy (reported only) |
|---|---|---|---|---|---|
| 7 | 0.6859 | 0.6993 | 34 | 0.6908 | 0.7020 |
| 42 | 0.6850 | 0.6959 | 38 | 0.6954 | 0.7082 |
| 1234 | 0.6857 | 0.6957 | 37 | 0.6937 | 0.7060 |
| 2025 | 0.6872 | 0.6959 | 37 | 0.6879 | 0.7053 |
| 3407 | 0.6840 | 0.6957 | 30 | 0.6937 | 0.7046 |
| mean ± SD | 0.6856 ± 0.0012 | 0.6965 ± 0.0016 | 35.2 ± 3.3 | 0.6923 ± 0.0029 | 0.7052 ± 0.0023 |

vs reference `a1_full`, matched seeds [7, 42, 1234, 2025, 3407]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| validation | 7 | -0.0045 | -0.0024 |
| validation | 42 | -0.0035 | -0.0010 |
| validation | 1234 | +0.0014 | +0.0007 |
| validation | 2025 | +0.0043 | -0.0010 |
| validation | 3407 | -0.0006 | -0.0010 |
| validation | mean | -0.0006 ± 0.0036 | -0.0009 ± 0.0011 |
| validation | CI seeds+patients | [-0.0066, +0.0051] (P>0 0.433) | patients only [-0.0044, +0.0032] |
| test (reported only, not used for decisions) | 7 | -0.0001 | -0.0036 |
| test (reported only, not used for decisions) | 42 | +0.0073 | +0.0062 |
| test (reported only, not used for decisions) | 1234 | +0.0048 | +0.0024 |
| test (reported only, not used for decisions) | 2025 | -0.0004 | +0.0017 |
| test (reported only, not used for decisions) | 3407 | +0.0046 | +0.0014 |
| test (reported only, not used for decisions) | mean | 0.0032 ± 0.0033 | 0.0016 ± 0.0035 |
| test (reported only, not used for decisions) | CI seeds+patients | [-0.0029, +0.0093] (P>0 0.857) | patients only [-0.0010, +0.0073] |

## a4b_nopair_noedge / cei_mp2

| seed | val macro-F1 | val accuracy | selected epoch | test macro-F1 (reported only) | test accuracy (reported only) |
|---|---|---|---|---|---|
| 7 | 0.6854 | 0.6983 | 35 | 0.6834 | 0.7010 |
| 42 | 0.6852 | 0.6976 | 40 | 0.6874 | 0.7010 |
| 1234 | 0.6842 | 0.6973 | 35 | 0.6824 | 0.6981 |
| 2025 | 0.6839 | 0.6969 | 40 | 0.6944 | 0.7103 |
| 3407 | 0.6845 | 0.6973 | 23 | 0.6833 | 0.6998 |
| mean ± SD | 0.6847 ± 0.0006 | 0.6975 ± 0.0005 | 34.6 ± 6.9 | 0.6862 ± 0.0050 | 0.7021 ± 0.0048 |

vs reference `a1_full`, matched seeds [7, 42, 1234, 2025, 3407]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| validation | 7 | -0.0050 | -0.0033 |
| validation | 42 | -0.0032 | +0.0007 |
| validation | 1234 | -0.0001 | +0.0024 |
| validation | 2025 | +0.0010 | +0.0000 |
| validation | 3407 | -0.0001 | +0.0007 |
| validation | mean | -0.0015 ± 0.0025 | 0.0001 ± 0.0021 |
| validation | CI seeds+patients | [-0.0081, +0.0056] (P>0 0.349) | patients only [-0.0067, +0.0041] |
| test (reported only, not used for decisions) | 7 | -0.0075 | -0.0046 |
| test (reported only, not used for decisions) | 42 | -0.0007 | -0.0010 |
| test (reported only, not used for decisions) | 1234 | -0.0065 | -0.0055 |
| test (reported only, not used for decisions) | 2025 | +0.0061 | +0.0067 |
| test (reported only, not used for decisions) | 3407 | -0.0059 | -0.0034 |
| test (reported only, not used for decisions) | mean | -0.0029 ± 0.0057 | -0.0015 ± 0.0049 |
| test (reported only, not used for decisions) | CI seeds+patients | [-0.0107, +0.0055] (P>0 0.242) | patients only [-0.0086, +0.0029] |

## a5_nomp / cei_mp2

| seed | val macro-F1 | val accuracy | selected epoch | test macro-F1 (reported only) | test accuracy (reported only) |
|---|---|---|---|---|---|
| 7 | 0.6803 | 0.6907 | 22 | 0.6856 | 0.6967 |
| 42 | 0.6841 | 0.6971 | 40 | 0.6874 | 0.7039 |
| 1234 | 0.6780 | 0.6911 | 36 | 0.6920 | 0.7058 |
| 2025 | 0.6835 | 0.6957 | 35 | 0.6899 | 0.7022 |
| 3407 | 0.6813 | 0.6921 | 40 | 0.6900 | 0.7034 |
| mean ± SD | 0.6815 ± 0.0025 | 0.6933 ± 0.0029 | 34.6 ± 7.4 | 0.6890 ± 0.0025 | 0.7024 ± 0.0034 |

vs reference `a1_full`, matched seeds [7, 42, 1234, 2025, 3407]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| validation | 7 | -0.0101 | -0.0110 |
| validation | 42 | -0.0043 | +0.0002 |
| validation | 1234 | -0.0063 | -0.0038 |
| validation | 2025 | +0.0006 | -0.0012 |
| validation | 3407 | -0.0033 | -0.0045 |
| validation | mean | -0.0047 ± 0.0039 | -0.0041 ± 0.0043 |
| validation | CI seeds+patients | [-0.0114, +0.0017] (P>0 0.085) | patients only [-0.0096, +0.0002] |
| test (reported only, not used for decisions) | 7 | -0.0053 | -0.0089 |
| test (reported only, not used for decisions) | 42 | -0.0007 | +0.0019 |
| test (reported only, not used for decisions) | 1234 | +0.0031 | +0.0022 |
| test (reported only, not used for decisions) | 2025 | +0.0016 | -0.0014 |
| test (reported only, not used for decisions) | 3407 | +0.0009 | +0.0002 |
| test (reported only, not used for decisions) | mean | -0.0001 ± 0.0032 | -0.0012 ± 0.0045 |
| test (reported only, not used for decisions) | CI seeds+patients | [-0.0067, +0.0064] (P>0 0.476) | patients only [-0.0053, +0.0054] |

## a5b_nomp_matched / cei_mp2

| seed | val macro-F1 | val accuracy | selected epoch | test macro-F1 (reported only) | test accuracy (reported only) |
|---|---|---|---|---|---|
| 7 | 0.6847 | 0.6954 | 40 | 0.6870 | 0.6998 |
| 42 | 0.6710 | 0.6849 | 40 | 0.6866 | 0.7024 |
| 1234 | 0.6747 | 0.6883 | 33 | 0.6829 | 0.6986 |
| 2025 | 0.6783 | 0.6895 | 36 | 0.6821 | 0.6943 |
| 3407 | 0.6714 | 0.6842 | 40 | 0.6821 | 0.6957 |
| mean ± SD | 0.6760 ± 0.0057 | 0.6885 ± 0.0045 | 37.8 ± 3.2 | 0.6841 ± 0.0025 | 0.6982 ± 0.0032 |

vs reference `a1_full`, matched seeds [7, 42, 1234, 2025, 3407]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| validation | 7 | -0.0057 | -0.0062 |
| validation | 42 | -0.0175 | -0.0120 |
| validation | 1234 | -0.0096 | -0.0067 |
| validation | 2025 | -0.0046 | -0.0074 |
| validation | 3407 | -0.0132 | -0.0124 |
| validation | mean | -0.0101 ± 0.0054 | -0.0089 ± 0.0030 |
| validation | CI seeds+patients | [-0.0177, -0.0027] (P>0 0.004) | patients only [-0.0153, -0.0051] |
| test (reported only, not used for decisions) | 7 | -0.0039 | -0.0057 |
| test (reported only, not used for decisions) | 42 | -0.0015 | +0.0005 |
| test (reported only, not used for decisions) | 1234 | -0.0060 | -0.0050 |
| test (reported only, not used for decisions) | 2025 | -0.0062 | -0.0093 |
| test (reported only, not used for decisions) | 3407 | -0.0071 | -0.0074 |
| test (reported only, not used for decisions) | mean | -0.0049 ± 0.0023 | -0.0054 ± 0.0037 |
| test (reported only, not used for decisions) | CI seeds+patients | [-0.0120, +0.0017] (P>0 0.081) | patients only [-0.0109, +0.0007] |

## a6_pool / cei_mp2

| seed | val macro-F1 | val accuracy | selected epoch | test macro-F1 (reported only) | test accuracy (reported only) |
|---|---|---|---|---|---|
| 7 | 0.6794 | 0.6902 | 38 | 0.6841 | 0.6977 |
| 42 | 0.6812 | 0.6942 | 33 | 0.6793 | 0.6967 |
| 1234 | 0.6820 | 0.6928 | 37 | 0.6871 | 0.6974 |
| 2025 | 0.6826 | 0.6911 | 19 | 0.6825 | 0.6921 |
| 3407 | 0.6782 | 0.6923 | 31 | 0.6831 | 0.6986 |
| mean ± SD | 0.6807 ± 0.0018 | 0.6921 ± 0.0016 | 31.6 ± 7.6 | 0.6832 ± 0.0028 | 0.6965 ± 0.0025 |

vs reference `a1_full`, matched seeds [7, 42, 1234, 2025, 3407]:

| fold | seed | d macro-F1 | d accuracy |
|---|---|---|---|
| validation | 7 | -0.0110 | -0.0115 |
| validation | 42 | -0.0073 | -0.0026 |
| validation | 1234 | -0.0024 | -0.0022 |
| validation | 2025 | -0.0003 | -0.0057 |
| validation | 3407 | -0.0064 | -0.0043 |
| validation | mean | -0.0055 ± 0.0042 | -0.0053 ± 0.0038 |
| validation | CI seeds+patients | [-0.0139, +0.0032] (P>0 0.101) | patients only [-0.0118, +0.0012] |
| test (reported only, not used for decisions) | 7 | -0.0068 | -0.0079 |
| test (reported only, not used for decisions) | 42 | -0.0089 | -0.0053 |
| test (reported only, not used for decisions) | 1234 | -0.0018 | -0.0062 |
| test (reported only, not used for decisions) | 2025 | -0.0058 | -0.0115 |
| test (reported only, not used for decisions) | 3407 | -0.0060 | -0.0046 |
| test (reported only, not used for decisions) | mean | -0.0059 ± 0.0026 | -0.0071 ± 0.0028 |
| test (reported only, not used for decisions) | CI seeds+patients | [-0.0136, +0.0017] (P>0 0.069) | patients only [-0.0123, +0.0007] |

