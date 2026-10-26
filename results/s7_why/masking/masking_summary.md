# Source masking (scored fold: test; baselines chosen on validation)

Seeds [1234, 2025, 7], limit None, threads 2. Drop = metric(none) - metric(mask). Difference = CEI drop - baseline drop (negative: CEI drops less); 95% CI from a patient bootstrap (B=1000, rng 20261017, shared draw, seed-mean statistic). Baselines chosen on validation macro-F1 (decision); every number below is on the test fold.

CEI-GNN reads an absence block ("items not observed"); removing labs therefore also changes its absence evidence (intended).

## K=10 (10 classes, 3175 test rows, 2497 patients, seeds [1234, 2025, 7])

Classes: Back or spine pain, Diabetes mellitus, GI bleed, Head injury, Heart failure, Limb injury or pain, Lower respiratory disease, Major depressive disorder, Skin or soft-tissue infection, UTI or pyelonephritis

| method | mask | macro-F1 | accuracy | macro-F1 drop | accuracy drop | seeds |
|---|---|---|---|---|---|---|
| CEI-GNN mp2 | none | 0.7612 | 0.7543 | +0.0000 | +0.0000 | 3 |
| CEI-GNN mp2 | history | 0.7566 | 0.7499 | +0.0046 | +0.0044 | 3 |
| CEI-GNN mp2 | labs | 0.6942 | 0.6896 | +0.0670 | +0.0648 | 3 |
| CEI-GNN mp2 | triage | 0.4732 | 0.4542 | +0.2880 | +0.3002 | 3 |
| ProtGNN | none | 0.7510 | 0.7451 | +0.0000 | +0.0000 | 3 |
| ProtGNN | history | 0.7495 | 0.7447 | +0.0014 | +0.0004 | 3 |
| ProtGNN | labs | 0.6773 | 0.6818 | +0.0737 | +0.0633 | 3 |
| ProtGNN | triage | 0.4608 | 0.4497 | +0.2902 | +0.2954 | 3 |
| GraphCare-style (adapted) | none | 0.7629 | 0.7543 | +0.0000 | +0.0000 | 3 |
| GraphCare-style (adapted) | history | 0.7569 | 0.7492 | +0.0060 | +0.0051 | 3 |
| GraphCare-style (adapted) | labs | 0.6686 | 0.6656 | +0.0943 | +0.0887 | 3 |
| GraphCare-style (adapted) | triage | 0.4648 | 0.4614 | +0.2981 | +0.2929 | 3 |

| baseline | mask | macro-F1 drop difference (CEI - baseline) | 95% CI | accuracy drop difference | 95% CI |
|---|---|---|---|---|---|
| ProtGNN | history | +0.0032 | [-0.0030, +0.0094] | +0.0040 | [-0.0017, +0.0096] |
| ProtGNN | labs | -0.0067 | [-0.0173, +0.0038] | +0.0015 | [-0.0076, +0.0096] |
| ProtGNN | triage | -0.0021 | [-0.0135, +0.0110] | +0.0047 | [-0.0072, +0.0176] |
| GraphCare-style (adapted) | history | -0.0014 | [-0.0076, +0.0052] | -0.0007 | [-0.0063, +0.0050] |
| GraphCare-style (adapted) | labs | -0.0273 | [-0.0385, -0.0158] | -0.0239 | [-0.0339, -0.0143] |
| GraphCare-style (adapted) | triage | -0.0100 | [-0.0225, +0.0046] | +0.0072 | [-0.0055, +0.0224] |

## K=20 (20 classes, 4174 test rows, 3151 patients, seeds [1234, 2025, 7])

Classes: Acute kidney failure, Acute pharyngitis, Alcohol abuse with intoxication, Anemia, Back or spine pain, Diabetes mellitus, Epilepsy, GI bleed, Head injury, Heart failure, Hypokalemia, Limb injury or pain, Lower respiratory disease, Major depressive disorder, Non-ST elevation (NSTEMI) myocardial infarction, Sepsis, Skin or soft-tissue infection, UTI or pyelonephritis, Unsp intestnl obst, Unspecified atrial fibrillation

| method | mask | macro-F1 | accuracy | macro-F1 drop | accuracy drop | seeds |
|---|---|---|---|---|---|---|
| CEI-GNN mp2 | none | 0.6894 | 0.7043 | +0.0000 | +0.0000 | 3 |
| CEI-GNN mp2 | history | 0.6849 | 0.6999 | +0.0044 | +0.0044 | 3 |
| CEI-GNN mp2 | labs | 0.6023 | 0.6322 | +0.0871 | +0.0720 | 3 |
| CEI-GNN mp2 | triage | 0.3882 | 0.3939 | +0.3012 | +0.3104 | 3 |
| ProtGNN | none | 0.6655 | 0.6851 | +0.0000 | +0.0000 | 3 |
| ProtGNN | history | 0.6667 | 0.6857 | -0.0012 | -0.0006 | 3 |
| ProtGNN | labs | 0.5693 | 0.6122 | +0.0961 | +0.0729 | 3 |
| ProtGNN | triage | 0.3824 | 0.3844 | +0.2831 | +0.3007 | 3 |
| GraphCare-style (adapted) | none | 0.6714 | 0.6877 | +0.0000 | +0.0000 | 3 |
| GraphCare-style (adapted) | history | 0.6733 | 0.6888 | -0.0020 | -0.0011 | 3 |
| GraphCare-style (adapted) | labs | 0.5699 | 0.5989 | +0.1015 | +0.0887 | 3 |
| GraphCare-style (adapted) | triage | 0.3431 | 0.3526 | +0.3283 | +0.3351 | 3 |

| baseline | mask | macro-F1 drop difference (CEI - baseline) | 95% CI | accuracy drop difference | 95% CI |
|---|---|---|---|---|---|
| ProtGNN | history | +0.0056 | [-0.0005, +0.0120] | +0.0050 | [+0.0000, +0.0102] |
| ProtGNN | labs | -0.0090 | [-0.0198, +0.0022] | -0.0009 | [-0.0090, +0.0071] |
| ProtGNN | triage | +0.0181 | [+0.0066, +0.0298] | +0.0097 | [-0.0014, +0.0204] |
| GraphCare-style (adapted) | history | +0.0064 | [+0.0003, +0.0127] | +0.0055 | [+0.0006, +0.0108] |
| GraphCare-style (adapted) | labs | -0.0144 | [-0.0251, -0.0036] | -0.0167 | [-0.0259, -0.0087] |
| GraphCare-style (adapted) | triage | -0.0270 | [-0.0397, -0.0145] | -0.0247 | [-0.0371, -0.0129] |

## K=30 (30 classes, 4566 test rows, 3404 patients, seeds [1234, 2025, 7])

Classes: Acute kidney failure, Acute pharyngitis, Acute upper respiratory infection, Alcohol abuse with intoxication, Anemia, Anxiety disorder, Back or spine pain, Cardiovascular risk factor, Dehydration, Diabetes mellitus, End stage renal disease, Epilepsy, GI bleed, Head injury, Heart failure, Hypokalemia, Hypotension, Hypothyroidism, Laceration w/o fb of l idx fngr w/o damage to nail, Limb injury or pain, Lower respiratory disease, Major depressive disorder, Multiple fractures of ribs, Non-ST elevation (NSTEMI) myocardial infarction, Sepsis, Skin or soft-tissue infection, UTI or pyelonephritis, Unsp intestnl obst, Unspecified asthma with (acute) exacerbation, Unspecified atrial fibrillation

| method | mask | macro-F1 | accuracy | macro-F1 drop | accuracy drop | seeds |
|---|---|---|---|---|---|---|
| CEI-GNN mp2 | none | 0.5781 | 0.6662 | +0.0000 | +0.0000 | 3 |
| CEI-GNN mp2 | history | 0.5729 | 0.6625 | +0.0052 | +0.0037 | 3 |
| CEI-GNN mp2 | labs | 0.5011 | 0.5874 | +0.0771 | +0.0788 | 3 |
| CEI-GNN mp2 | triage | 0.2582 | 0.3311 | +0.3199 | +0.3350 | 3 |
| ProtGNN | none | 0.5567 | 0.6472 | +0.0000 | +0.0000 | 3 |
| ProtGNN | history | 0.5553 | 0.6445 | +0.0014 | +0.0026 | 3 |
| ProtGNN | labs | 0.4688 | 0.5772 | +0.0878 | +0.0699 | 3 |
| ProtGNN | triage | 0.2602 | 0.3398 | +0.2964 | +0.3074 | 3 |
| GSAT | none | 0.5570 | 0.6483 | +0.0000 | +0.0000 | 3 |
| GSAT | history | 0.5610 | 0.6493 | -0.0041 | -0.0010 | 3 |
| GSAT | labs | 0.4746 | 0.5572 | +0.0824 | +0.0910 | 3 |
| GSAT | triage | 0.2291 | 0.2886 | +0.3279 | +0.3597 | 3 |

| baseline | mask | macro-F1 drop difference (CEI - baseline) | 95% CI | accuracy drop difference | 95% CI |
|---|---|---|---|---|---|
| ProtGNN | history | +0.0038 | [-0.0045, +0.0110] | +0.0010 | [-0.0039, +0.0058] |
| ProtGNN | labs | -0.0107 | [-0.0219, +0.0017] | +0.0088 | [-0.0000, +0.0172] |
| ProtGNN | triage | +0.0235 | [+0.0116, +0.0353] | +0.0276 | [+0.0162, +0.0384] |
| GSAT | history | +0.0093 | [+0.0005, +0.0168] | +0.0047 | [-0.0004, +0.0094] |
| GSAT | labs | -0.0053 | [-0.0183, +0.0069] | -0.0123 | [-0.0210, -0.0039] |
| GSAT | triage | -0.0080 | [-0.0209, +0.0050] | -0.0247 | [-0.0359, -0.0144] |

Parity gate (unmasked vs stored test probabilities, tolerance 0.0001): 27/27 runs passed, max abs diff 1.430511474609375e-06.
