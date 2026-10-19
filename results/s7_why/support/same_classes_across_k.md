# Same classes across K (K-way task)

F1 averaged over a fixed class set (top10: the K=10 classes, top20: the K=20 classes, all: all K classes), computed inside each K-way task. delta = CEI minus the mean of the baselines (delta_mean) or minus the validation-chosen best baseline (delta_best); the change is relative to the first K that contains the set, with the same best baseline (chosen at the later K) at both ends. class_set=all is the official K-way macro-F1 gap. 95% interval: subject bootstrap over the pooled subjects of the fold (B=1000, seed 20261017). Test is reported only, not used for decisions.

| fold | class_set | K | n_classes_in_set | f1_cei | f1_mean_baselines | f1_best_baseline | best_baseline | delta_mean | delta_mean_ci | delta_best | delta_best_ci | delta_mean_change | delta_mean_change_ci | delta_best_change | delta_best_change_ci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | top10 | 10 | 10 | 0.7451 | 0.7311 | 0.7338 | protgnn | 0.014 | [0.0084, 0.0202] | 0.0113 | [0.0039, 0.0184] |  |  |  |  |
| validation | top10 | 20 | 10 | 0.7069 | 0.6943 | 0.6923 | protgnn | 0.0126 | [0.0067, 0.0187] | 0.0146 | [0.0072, 0.0217] | -0.0015 (vs K10) | [-0.0083, 0.0056] | +0.0033 (vs K10) | [-0.0052, 0.0124] |
| validation | top10 | 30 | 10 | 0.6855 | 0.6731 | 0.6736 | protgnn | 0.0124 | [0.0061, 0.0190] | 0.0119 | [0.0042, 0.0193] | -0.0016 (vs K10) | [-0.0093, 0.0058] | +0.0006 (vs K10) | [-0.0089, 0.0101] |
| validation | top20 | 20 | 20 | 0.6859 | 0.6735 | 0.6743 | protgnn | 0.0124 | [0.0055, 0.0195] | 0.0115 | [0.0037, 0.0195] |  |  |  |  |
| validation | top20 | 30 | 20 | 0.6655 | 0.6535 | 0.6567 | protgnn | 0.012 | [0.0049, 0.0184] | 0.0088 | [0.0003, 0.0165] | -0.0005 (vs K20) | [-0.0086, 0.0074] | -0.0027 (vs K20) | [-0.0125, 0.0072] |
| validation | all | 10 | 10 | 0.7451 | 0.7311 | 0.7338 | protgnn | 0.014 | [0.0084, 0.0202] | 0.0113 | [0.0039, 0.0184] |  |  |  |  |
| validation | all | 20 | 20 | 0.6859 | 0.6735 | 0.6743 | protgnn | 0.0124 | [0.0055, 0.0195] | 0.0115 | [0.0037, 0.0195] | -0.0016 (vs K10) | [-0.0099, 0.0063] | +0.0003 (vs K10) | [-0.0096, 0.0108] |
| validation | all | 30 | 30 | 0.5708 | 0.5557 | 0.5632 | protgnn | 0.0151 | [0.0073, 0.0243] | 0.0076 | [-0.0007, 0.0189] | +0.0011 (vs K10) | [-0.0085, 0.0113] | -0.0036 (vs K10) | [-0.0140, 0.0092] |
| test | top10 | 10 | 10 | 0.7612 | 0.7549 | 0.751 | protgnn | 0.0063 | [-0.0000, 0.0126] | 0.0102 | [0.0025, 0.0181] |  |  |  |  |
| test | top10 | 20 | 10 | 0.7252 | 0.7065 | 0.7059 | protgnn | 0.0187 | [0.0122, 0.0249] | 0.0193 | [0.0120, 0.0265] | +0.0125 (vs K10) | [0.0052, 0.0193] | +0.0091 (vs K10) | [0.0004, 0.0174] |
| test | top10 | 30 | 10 | 0.702 | 0.6845 | 0.6854 | protgnn | 0.0176 | [0.0111, 0.0242] | 0.0166 | [0.0091, 0.0237] | +0.0113 (vs K10) | [0.0040, 0.0182] | +0.0064 (vs K10) | [-0.0024, 0.0150] |
| test | top20 | 20 | 20 | 0.6894 | 0.6687 | 0.6655 | protgnn | 0.0207 | [0.0137, 0.0282] | 0.0239 | [0.0157, 0.0334] |  |  |  |  |
| test | top20 | 30 | 20 | 0.6718 | 0.65 | 0.6467 | protgnn | 0.0219 | [0.0146, 0.0289] | 0.0251 | [0.0165, 0.0339] | +0.0012 (vs K20) | [-0.0063, 0.0081] | +0.0012 (vs K20) | [-0.0082, 0.0101] |
| test | all | 10 | 10 | 0.7612 | 0.7549 | 0.751 | protgnn | 0.0063 | [-0.0000, 0.0126] | 0.0102 | [0.0025, 0.0181] |  |  |  |  |
| test | all | 20 | 20 | 0.6894 | 0.6687 | 0.6655 | protgnn | 0.0207 | [0.0137, 0.0282] | 0.0239 | [0.0157, 0.0334] | +0.0144 (vs K10) | [0.0052, 0.0236] | +0.0136 (vs K10) | [0.0032, 0.0243] |
| test | all | 30 | 30 | 0.5781 | 0.5535 | 0.5567 | protgnn | 0.0247 | [0.0161, 0.0346] | 0.0215 | [0.0125, 0.0314] | +0.0184 (vs K10) | [0.0087, 0.0294] | +0.0112 (vs K10) | [0.0007, 0.0233] |
