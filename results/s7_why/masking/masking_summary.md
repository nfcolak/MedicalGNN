# Source masking (fold: validation; test never touched)

Seeds [1234, 2025, 7], limit None, threads 2. Drop = metric(none) - metric(mask). Difference = CEI drop - baseline drop (negative: CEI drops less); 95% CI from a patient bootstrap (B=1000, rng 20261017, shared draw, seed-mean statistic). Baselines chosen on validation macro-F1.

CEI-GNN reads an absence block ("items not observed"); removing labs therefore also changes its absence evidence (intended).

## K=10 (10 classes, 3165 validation rows, 2473 patients, seeds [1234, 2025, 7])

Classes: Back or spine pain, Diabetes mellitus, GI bleed, Head injury, Heart failure, Limb injury or pain, Lower respiratory disease, Major depressive disorder, Skin or soft-tissue infection, UTI or pyelonephritis

| method | mask | macro-F1 | accuracy | macro-F1 drop | accuracy drop | seeds |
|---|---|---|---|---|---|---|
| CEI-GNN mp2 | none | 0.7451 | 0.7410 | +0.0000 | +0.0000 | 3 |
| CEI-GNN mp2 | history | 0.7402 | 0.7366 | +0.0049 | +0.0044 | 3 |
| CEI-GNN mp2 | labs | 0.6688 | 0.6737 | +0.0763 | +0.0673 | 3 |
| CEI-GNN mp2 | triage | 0.4551 | 0.4362 | +0.2900 | +0.3048 | 3 |
| ProtGNN | none | 0.7338 | 0.7329 | +0.0000 | +0.0000 | 3 |
| ProtGNN | history | 0.7328 | 0.7324 | +0.0010 | +0.0005 | 3 |
| ProtGNN | labs | 0.6544 | 0.6659 | +0.0795 | +0.0670 | 3 |
| ProtGNN | triage | 0.4505 | 0.4439 | +0.2833 | +0.2890 | 3 |
| GraphCare-style (adapted) | none | 0.7332 | 0.7315 | +0.0000 | +0.0000 | 3 |
| GraphCare-style (adapted) | history | 0.7363 | 0.7340 | -0.0031 | -0.0024 | 3 |
| GraphCare-style (adapted) | labs | 0.6434 | 0.6476 | +0.0898 | +0.0839 | 3 |
| GraphCare-style (adapted) | triage | 0.4499 | 0.4489 | +0.2833 | +0.2827 | 3 |

| baseline | mask | macro-F1 drop difference (CEI - baseline) | 95% CI | accuracy drop difference | 95% CI |
|---|---|---|---|---|---|
| ProtGNN | history | +0.0039 | [-0.0018, +0.0099] | +0.0039 | [-0.0017, +0.0095] |
| ProtGNN | labs | -0.0031 | [-0.0130, +0.0068] | +0.0003 | [-0.0087, +0.0092] |
| ProtGNN | triage | +0.0067 | [-0.0051, +0.0193] | +0.0158 | [+0.0041, +0.0272] |
| GraphCare-style (adapted) | history | +0.0080 | [+0.0022, +0.0133] | +0.0068 | [+0.0009, +0.0121] |
| GraphCare-style (adapted) | labs | -0.0134 | [-0.0249, -0.0028] | -0.0166 | [-0.0271, -0.0066] |
| GraphCare-style (adapted) | triage | +0.0067 | [-0.0059, +0.0210] | +0.0221 | [+0.0091, +0.0370] |

## K=20 (20 classes, 4183 validation rows, 3142 patients, seeds [1234, 2025, 7])

Classes: Acute kidney failure, Acute pharyngitis, Alcohol abuse with intoxication, Anemia, Back or spine pain, Diabetes mellitus, Epilepsy, GI bleed, Head injury, Heart failure, Hypokalemia, Limb injury or pain, Lower respiratory disease, Major depressive disorder, Non-ST elevation (NSTEMI) myocardial infarction, Sepsis, Skin or soft-tissue infection, UTI or pyelonephritis, Unsp intestnl obst, Unspecified atrial fibrillation

| method | mask | macro-F1 | accuracy | macro-F1 drop | accuracy drop | seeds |
|---|---|---|---|---|---|---|
| CEI-GNN mp2 | none | 0.6859 | 0.6978 | +0.0000 | +0.0000 | 3 |
| CEI-GNN mp2 | history | 0.6815 | 0.6926 | +0.0044 | +0.0052 | 3 |
| CEI-GNN mp2 | labs | 0.5929 | 0.6255 | +0.0930 | +0.0724 | 3 |
| CEI-GNN mp2 | triage | 0.3927 | 0.3915 | +0.2932 | +0.3063 | 3 |
| ProtGNN | none | 0.6743 | 0.6850 | +0.0000 | +0.0000 | 3 |
| ProtGNN | history | 0.6705 | 0.6832 | +0.0038 | +0.0018 | 3 |
| ProtGNN | labs | 0.5572 | 0.6005 | +0.1171 | +0.0845 | 3 |
| ProtGNN | triage | 0.3887 | 0.3863 | +0.2856 | +0.2987 | 3 |
| GraphCare-style (adapted) | none | 0.6742 | 0.6866 | +0.0000 | +0.0000 | 3 |
| GraphCare-style (adapted) | history | 0.6680 | 0.6826 | +0.0062 | +0.0040 | 3 |
| GraphCare-style (adapted) | labs | 0.5608 | 0.5943 | +0.1135 | +0.0923 | 3 |
| GraphCare-style (adapted) | triage | 0.3604 | 0.3621 | +0.3138 | +0.3245 | 3 |

| baseline | mask | macro-F1 drop difference (CEI - baseline) | 95% CI | accuracy drop difference | 95% CI |
|---|---|---|---|---|---|
| ProtGNN | history | +0.0005 | [-0.0059, +0.0064] | +0.0034 | [-0.0020, +0.0085] |
| ProtGNN | labs | -0.0242 | [-0.0348, -0.0136] | -0.0121 | [-0.0208, -0.0028] |
| ProtGNN | triage | +0.0076 | [-0.0051, +0.0193] | +0.0076 | [-0.0030, +0.0175] |
| GraphCare-style (adapted) | history | -0.0018 | [-0.0084, +0.0042] | +0.0012 | [-0.0048, +0.0063] |
| GraphCare-style (adapted) | labs | -0.0205 | [-0.0318, -0.0096] | -0.0199 | [-0.0287, -0.0112] |
| GraphCare-style (adapted) | triage | -0.0207 | [-0.0337, -0.0081] | -0.0182 | [-0.0300, -0.0067] |

## K=30 (30 classes, 4534 validation rows, 3388 patients, seeds [1234, 2025, 7])

Classes: Acute kidney failure, Acute pharyngitis, Acute upper respiratory infection, Alcohol abuse with intoxication, Anemia, Anxiety disorder, Back or spine pain, Cardiovascular risk factor, Dehydration, Diabetes mellitus, End stage renal disease, Epilepsy, GI bleed, Head injury, Heart failure, Hypokalemia, Hypotension, Hypothyroidism, Laceration w/o fb of l idx fngr w/o damage to nail, Limb injury or pain, Lower respiratory disease, Major depressive disorder, Multiple fractures of ribs, Non-ST elevation (NSTEMI) myocardial infarction, Sepsis, Skin or soft-tissue infection, UTI or pyelonephritis, Unsp intestnl obst, Unspecified asthma with (acute) exacerbation, Unspecified atrial fibrillation

| method | mask | macro-F1 | accuracy | macro-F1 drop | accuracy drop | seeds |
|---|---|---|---|---|---|---|
| CEI-GNN mp2 | none | 0.5708 | 0.6576 | +0.0000 | +0.0000 | 3 |
| CEI-GNN mp2 | history | 0.5603 | 0.6523 | +0.0104 | +0.0054 | 3 |
| CEI-GNN mp2 | labs | 0.4856 | 0.5789 | +0.0852 | +0.0787 | 3 |
| CEI-GNN mp2 | triage | 0.2647 | 0.3380 | +0.3061 | +0.3196 | 3 |
| ProtGNN | none | 0.5632 | 0.6471 | +0.0000 | +0.0000 | 3 |
| ProtGNN | history | 0.5534 | 0.6451 | +0.0098 | +0.0021 | 3 |
| ProtGNN | labs | 0.4661 | 0.5718 | +0.0971 | +0.0754 | 3 |
| ProtGNN | triage | 0.2641 | 0.3475 | +0.2991 | +0.2996 | 3 |
| GSAT | none | 0.5548 | 0.6445 | +0.0000 | +0.0000 | 3 |
| GSAT | history | 0.5505 | 0.6426 | +0.0043 | +0.0019 | 3 |
| GSAT | labs | 0.4626 | 0.5459 | +0.0922 | +0.0987 | 3 |
| GSAT | triage | 0.2210 | 0.2878 | +0.3338 | +0.3568 | 3 |

| baseline | mask | macro-F1 drop difference (CEI - baseline) | 95% CI | accuracy drop difference | 95% CI |
|---|---|---|---|---|---|
| ProtGNN | history | +0.0007 | [-0.0065, +0.0095] | +0.0033 | [-0.0019, +0.0087] |
| ProtGNN | labs | -0.0119 | [-0.0225, +0.0008] | +0.0034 | [-0.0050, +0.0118] |
| ProtGNN | triage | +0.0070 | [-0.0042, +0.0210] | +0.0200 | [+0.0084, +0.0317] |
| GSAT | history | +0.0062 | [-0.0009, +0.0132] | +0.0035 | [-0.0017, +0.0089] |
| GSAT | labs | -0.0070 | [-0.0191, +0.0051] | -0.0199 | [-0.0283, -0.0116] |
| GSAT | triage | -0.0277 | [-0.0397, -0.0151] | -0.0372 | [-0.0481, -0.0246] |

Parity gate (unmasked vs stored validation probabilities, tolerance 0.0001): 27/27 runs passed, max abs diff 1.519918441772461e-06.
