# Macro-F1 variants

Final runs, protocol fair_v10t; seeds 1234, 2025, 7; methods CEI-GNN mp2, GraphCare-style (adapted), GSAT, ProtGNN. Validation decides; test fold: reported only, not used for decisions.

fixed = mean F1 over all K classes (the reported metric, primary); present = mean F1 over classes with at least 1 true row in the fold; ge30 = mean F1 over classes with at least 30 true rows in the fold. Mean +- SD over seeds, per-seed values in brackets; ranking best first.

## K=10, validation fold (decision fold): 3165 rows, 2473 patients, seeds 1234, 2025, 7

| variant | n classes | CEI-GNN mp2 | GraphCare-style (adapted) | GSAT | ProtGNN | ranking |
|---|---|---|---|---|---|---|
| fixed | 10 | 0.7451 +- 0.0033 [0.7444, 0.7487, 0.7422] | 0.7332 +- 0.0033 [0.7339, 0.7296, 0.7361] | 0.7262 +- 0.0015 [0.7257, 0.7250, 0.7278] | 0.7338 +- 0.0035 [0.7313, 0.7324, 0.7378] | CEI-GNN mp2 > ProtGNN > GraphCare-style (adapted) > GSAT |
| present | 10 | 0.7451 +- 0.0033 [0.7444, 0.7487, 0.7422] | 0.7332 +- 0.0033 [0.7339, 0.7296, 0.7361] | 0.7262 +- 0.0015 [0.7257, 0.7250, 0.7278] | 0.7338 +- 0.0035 [0.7313, 0.7324, 0.7378] | CEI-GNN mp2 > ProtGNN > GraphCare-style (adapted) > GSAT |
| ge30 | 10 | 0.7451 +- 0.0033 [0.7444, 0.7487, 0.7422] | 0.7332 +- 0.0033 [0.7339, 0.7296, 0.7361] | 0.7262 +- 0.0015 [0.7257, 0.7250, 0.7278] | 0.7338 +- 0.0035 [0.7313, 0.7324, 0.7378] | CEI-GNN mp2 > ProtGNN > GraphCare-style (adapted) > GSAT |

## K=10, test fold (test fold: reported only, not used for decisions): 3175 rows, 2497 patients, seeds 1234, 2025, 7

| variant | n classes | CEI-GNN mp2 | GraphCare-style (adapted) | GSAT | ProtGNN | ranking |
|---|---|---|---|---|---|---|
| fixed | 10 | 0.7612 +- 0.0090 [0.7529, 0.7707, 0.7600] | 0.7629 +- 0.0032 [0.7665, 0.7605, 0.7616] | 0.7509 +- 0.0042 [0.7462, 0.7524, 0.7541] | 0.7510 +- 0.0062 [0.7456, 0.7497, 0.7577] | GraphCare-style (adapted) > CEI-GNN mp2 > ProtGNN > GSAT |
| present | 10 | 0.7612 +- 0.0090 [0.7529, 0.7707, 0.7600] | 0.7629 +- 0.0032 [0.7665, 0.7605, 0.7616] | 0.7509 +- 0.0042 [0.7462, 0.7524, 0.7541] | 0.7510 +- 0.0062 [0.7456, 0.7497, 0.7577] | GraphCare-style (adapted) > CEI-GNN mp2 > ProtGNN > GSAT |
| ge30 | 10 | 0.7612 +- 0.0090 [0.7529, 0.7707, 0.7600] | 0.7629 +- 0.0032 [0.7665, 0.7605, 0.7616] | 0.7509 +- 0.0042 [0.7462, 0.7524, 0.7541] | 0.7510 +- 0.0062 [0.7456, 0.7497, 0.7577] | GraphCare-style (adapted) > CEI-GNN mp2 > ProtGNN > GSAT |

## K=20, validation fold (decision fold): 4183 rows, 3142 patients, seeds 1234, 2025, 7

| variant | n classes | CEI-GNN mp2 | GraphCare-style (adapted) | GSAT | ProtGNN | ranking |
|---|---|---|---|---|---|---|
| fixed | 20 | 0.6859 +- 0.0040 [0.6843, 0.6829, 0.6904] | 0.6742 +- 0.0017 [0.6759, 0.6725, 0.6743] | 0.6718 +- 0.0029 [0.6744, 0.6723, 0.6687] | 0.6743 +- 0.0041 [0.6759, 0.6697, 0.6774] | CEI-GNN mp2 > ProtGNN > GraphCare-style (adapted) > GSAT |
| present | 20 | 0.6859 +- 0.0040 [0.6843, 0.6829, 0.6904] | 0.6742 +- 0.0017 [0.6759, 0.6725, 0.6743] | 0.6718 +- 0.0029 [0.6744, 0.6723, 0.6687] | 0.6743 +- 0.0041 [0.6759, 0.6697, 0.6774] | CEI-GNN mp2 > ProtGNN > GraphCare-style (adapted) > GSAT |
| ge30 | 20 | 0.6859 +- 0.0040 [0.6843, 0.6829, 0.6904] | 0.6742 +- 0.0017 [0.6759, 0.6725, 0.6743] | 0.6718 +- 0.0029 [0.6744, 0.6723, 0.6687] | 0.6743 +- 0.0041 [0.6759, 0.6697, 0.6774] | CEI-GNN mp2 > ProtGNN > GraphCare-style (adapted) > GSAT |

## K=20, test fold (test fold: reported only, not used for decisions): 4174 rows, 3151 patients, seeds 1234, 2025, 7

| variant | n classes | CEI-GNN mp2 | GraphCare-style (adapted) | GSAT | ProtGNN | ranking |
|---|---|---|---|---|---|---|
| fixed | 20 | 0.6894 +- 0.0013 [0.6889, 0.6883, 0.6909] | 0.6714 +- 0.0044 [0.6724, 0.6666, 0.6751] | 0.6692 +- 0.0041 [0.6737, 0.6657, 0.6683] | 0.6655 +- 0.0023 [0.6646, 0.6681, 0.6637] | CEI-GNN mp2 > GraphCare-style (adapted) > GSAT > ProtGNN |
| present | 20 | 0.6894 +- 0.0013 [0.6889, 0.6883, 0.6909] | 0.6714 +- 0.0044 [0.6724, 0.6666, 0.6751] | 0.6692 +- 0.0041 [0.6737, 0.6657, 0.6683] | 0.6655 +- 0.0023 [0.6646, 0.6681, 0.6637] | CEI-GNN mp2 > GraphCare-style (adapted) > GSAT > ProtGNN |
| ge30 | 20 | 0.6894 +- 0.0013 [0.6889, 0.6883, 0.6909] | 0.6714 +- 0.0044 [0.6724, 0.6666, 0.6751] | 0.6692 +- 0.0041 [0.6737, 0.6657, 0.6683] | 0.6655 +- 0.0023 [0.6646, 0.6681, 0.6637] | CEI-GNN mp2 > GraphCare-style (adapted) > GSAT > ProtGNN |

## K=30, validation fold (decision fold): 4534 rows, 3388 patients, seeds 1234, 2025, 7

| variant | n classes | CEI-GNN mp2 | GraphCare-style (adapted) | GSAT | ProtGNN | ranking |
|---|---|---|---|---|---|---|
| fixed | 30 | 0.5708 +- 0.0037 [0.5749, 0.5677, 0.5697] | 0.5491 +- 0.0019 [0.5506, 0.5498, 0.5469] | 0.5548 +- 0.0023 [0.5568, 0.5522, 0.5552] | 0.5632 +- 0.0089 [0.5731, 0.5605, 0.5559] | CEI-GNN mp2 > ProtGNN > GSAT > GraphCare-style (adapted) |
| present | 30 | 0.5708 +- 0.0037 [0.5749, 0.5677, 0.5697] | 0.5491 +- 0.0019 [0.5506, 0.5498, 0.5469] | 0.5548 +- 0.0023 [0.5568, 0.5522, 0.5552] | 0.5632 +- 0.0089 [0.5731, 0.5605, 0.5559] | CEI-GNN mp2 > ProtGNN > GSAT > GraphCare-style (adapted) |
| ge30 | 27 | 0.6179 +- 0.0016 [0.6198, 0.6169, 0.6170] | 0.6026 +- 0.0008 [0.6030, 0.6017, 0.6031] | 0.6050 +- 0.0015 [0.6039, 0.6067, 0.6044] | 0.6062 +- 0.0025 [0.6090, 0.6042, 0.6053] | CEI-GNN mp2 > ProtGNN > GSAT > GraphCare-style (adapted) |

## K=30, test fold (test fold: reported only, not used for decisions): 4566 rows, 3404 patients, seeds 1234, 2025, 7

| variant | n classes | CEI-GNN mp2 | GraphCare-style (adapted) | GSAT | ProtGNN | ranking |
|---|---|---|---|---|---|---|
| fixed | 30 | 0.5781 +- 0.0035 [0.5754, 0.5768, 0.5822] | 0.5467 +- 0.0020 [0.5491, 0.5453, 0.5458] | 0.5570 +- 0.0075 [0.5637, 0.5584, 0.5489] | 0.5567 +- 0.0023 [0.5591, 0.5562, 0.5546] | CEI-GNN mp2 > GSAT > ProtGNN > GraphCare-style (adapted) |
| present | 29 | 0.5981 +- 0.0037 [0.5953, 0.5967, 0.6022] | 0.5656 +- 0.0021 [0.5680, 0.5641, 0.5646] | 0.5762 +- 0.0078 [0.5831, 0.5776, 0.5678] | 0.5758 +- 0.0024 [0.5784, 0.5754, 0.5737] | CEI-GNN mp2 > GSAT > ProtGNN > GraphCare-style (adapted) |
| ge30 | 27 | 0.6266 +- 0.0043 [0.6244, 0.6237, 0.6315] | 0.5998 +- 0.0036 [0.5967, 0.5988, 0.6038] | 0.6031 +- 0.0086 [0.6094, 0.6067, 0.5933] | 0.6026 +- 0.0057 [0.6066, 0.6051, 0.5960] | CEI-GNN mp2 > GSAT > ProtGNN > GraphCare-style (adapted) |

