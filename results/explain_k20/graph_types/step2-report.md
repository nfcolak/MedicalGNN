# Step 2: graph-structure axis

## 1. Setup

Unit: the visit (1345 visits, 1000 patients, test fold, K=20, primary_dx_v4, fair_v10t). Measure: per visit, the mean over the seeds that are ok (a visit counts when ok in at least 2 of the 3 seeds) of `deletion_mean_change_vs_random` ("del"). Paired differences on the same visit; CEI = cei_final and, separately, cei_R1_ls0. CI: patient-cluster bootstrap (resample patients with replacement, 2000 draws, seed 20261025), 95% percentile. Column codes: `X/GSAT` = CEI source X minus GSAT source X; `X/GC` = CEI source X minus GraphCare source X (X = IG, Grad, GNNExp); `bi/GSATbi` = CEI built-in minus GSAT built-in; `bi/GSATIG` = CEI built-in minus GSAT IG; `bi/GCGrad` = CEI built-in minus GraphCare Grad (GraphCare built-in is not usable). Positive = CEI is better (headline del = deletion_mean_change_vs_random).

Cell format: `mean [95% CI]`; `*` = CI excludes 0; `+` = cell has fewer than 30 visits (printed, not interpreted); `n<10` = not printed. n visits and n patients are the cell sizes of the first column (GSAT IG comparison); the other columns differ only by the visits dropped below.

**Raw** = plain paired difference. **Adjusted** = confidence-adjusted by regression: d is regressed on indicator bins of the CEI model's confidence and of the other model's confidence (quintile edges from all visits), refit in every bootstrap draw; the adjusted stratum mean is the mean of d minus the fitted confidence part (re-centred, so the all-visit mean is unchanged). It removes the additive part of the confidence difference, not any interaction.

**Bin rule (fixed in the script before any difference was computed)**: features with at most 6 distinct values get one bin per value; otherwise bins are cut at the lower quartile values (q25, q50, q75 of the feature over the 1,345 visits, duplicates merged, so a feature with many zeros has a `0` bin). Features: n_nodes, n_evidence, n_edges, density, num_visits, counts and shares (of evidence nodes) of complaint / measurement / vital / diagnosis nodes, knowledge and analyte node counts, n_prior_dx, relation counts trajectory_of, recurrence_of, comorbid_with, co_complaint, baseline_of, and medical_* (total and each of the 7 relations). Constant features skipped: rel_medical_interacts_with.

**Visits dropped (ok in fewer than 2 of 3 seeds) per model and source**: graphcare GNNExp: <10.

## 2. All visits (paired difference, CEI minus other)

**cei_final**

| comparison | n visits | n patients | mean difference [95% CI] |
|---|---|---|---|
| IG/GSAT | 1345 | 1000 | -0.0506 [-0.0549, -0.0463]* |
| IG/GC | 1345 | 1000 | -0.0483 [-0.0528, -0.0440]* |
| Grad/GSAT | 1345 | 1000 | -0.0473 [-0.0515, -0.0429]* |
| Grad/GC | 1345 | 1000 | -0.0790 [-0.0849, -0.0731]* |
| GNNExp/GSAT | 1345 | 1000 | -0.0296 [-0.0357, -0.0234]* |
| GNNExp/GC | 1344 | 1000 | +0.0645 [+0.0561, +0.0731]* |
| bi/GSATbi | 1345 | 1000 | -0.0163 [-0.0208, -0.0114]* |
| bi/GSATIG | 1345 | 1000 | -0.0339 [-0.0384, -0.0294]* |
| bi/GCGrad | 1345 | 1000 | -0.0575 [-0.0633, -0.0514]* |

**cei_R1_ls0**

| comparison | n visits | n patients | mean difference [95% CI] |
|---|---|---|---|
| IG/GSAT | 1345 | 1000 | -0.0158 [-0.0207, -0.0110]* |
| IG/GC | 1345 | 1000 | -0.0136 [-0.0187, -0.0087]* |
| Grad/GSAT | 1345 | 1000 | -0.0157 [-0.0202, -0.0111]* |
| Grad/GC | 1345 | 1000 | -0.0475 [-0.0540, -0.0409]* |
| GNNExp/GSAT | 1345 | 1000 | -0.0152 [-0.0228, -0.0081]* |
| GNNExp/GC | 1344 | 1000 | +0.0789 [+0.0712, +0.0865]* |
| bi/GSATbi | 1345 | 1000 | +0.0301 [+0.0252, +0.0353]* |
| bi/GSATIG | 1345 | 1000 | +0.0125 [+0.0078, +0.0171]* |
| bi/GCGrad | 1345 | 1000 | -0.0111 [-0.0170, -0.0053]* |

## 3. Feature overview (all strata looked at)

Per feature: number of bins; cells = (bin with at least 30 visits) x (18 comparisons: 9 per CEI version); "d excludes 0" counts cells whose CI of d excludes 0 (raw / adjusted; d is negative nearly everywhere because CEI is behind on average, so this counts mostly "CEI worse" cells); "d>0 sig." counts cells where CEI is significantly better; "contrast" counts cells where the stratum differs from the rest of the visits (CI of stratum mean minus rest mean excludes 0), raw / adjusted; "contrast >0" counts the subset where the stratum is better for CEI than the rest.

| feature | bins | cells | d excludes 0 (raw/adj) | d>0 sig. (raw/adj) | contrast (raw/adj) | contrast >0 (raw/adj) |
|---|---|---|---|---|---|---|
| n_nodes | 4 | 72 | 62/64 | 21/16 | 49/32 | 25/16 |
| n_evidence | 4 | 72 | 62/64 | 21/16 | 45/36 | 21/17 |
| n_edges | 4 | 72 | 63/65 | 21/16 | 47/34 | 23/16 |
| density | 4 | 72 | 61/65 | 20/16 | 46/32 | 22/16 |
| num_visits | 6 | 90 | 62/67 | 26/17 | 47/44 | 33/31 |
| n_complaint | 5 | 72 | 53/58 | 15/16 | 33/28 | 20/16 |
| n_measurement | 4 | 72 | 65/65 | 21/16 | 46/32 | 22/16 |
| n_vital | 6 | 18 | 18/18 | 4/4 | 10/5 | 0/0 |
| n_diagnosis | 3 | 54 | 43/46 | 15/11 | 30/35 | 18/21 |
| share_complaint | 4 | 72 | 63/66 | 18/15 | 49/22 | 23/10 |
| share_measurement | 4 | 72 | 61/65 | 20/16 | 56/32 | 30/17 |
| share_vital | 4 | 72 | 63/66 | 21/16 | 45/31 | 21/14 |
| share_diagnosis | 3 | 54 | 48/46 | 16/13 | 36/34 | 21/21 |
| n_knowledge | 4 | 72 | 59/66 | 18/16 | 41/30 | 21/15 |
| n_analyte | 4 | 72 | 62/64 | 18/15 | 44/33 | 26/18 |
| n_prior_dx | 3 | 54 | 43/46 | 15/11 | 30/35 | 18/21 |
| rel_trajectory_of | 2 | 36 | 30/32 | 12/9 | 24/20 | 12/10 |
| rel_recurrence_of | 3 | 54 | 43/46 | 15/11 | 30/35 | 18/21 |
| rel_comorbid_with | 3 | 54 | 46/49 | 14/12 | 25/31 | 15/18 |
| rel_co_complaint | 4 | 54 | 43/42 | 10/12 | 15/25 | 8/15 |
| rel_baseline_of | 3 | 54 | 47/46 | 17/11 | 36/37 | 20/22 |
| rel_medical_total | 4 | 72 | 61/64 | 19/15 | 40/32 | 20/16 |
| rel_medical_member_of | 4 | 72 | 62/64 | 17/15 | 40/24 | 24/12 |
| rel_medical_assesses | 4 | 72 | 61/64 | 19/15 | 36/20 | 22/12 |
| rel_medical_measures | 4 | 72 | 61/59 | 17/15 | 43/23 | 26/12 |
| rel_medical_is_a | 4 | 72 | 59/64 | 18/16 | 39/38 | 20/21 |
| rel_medical_may_affect | 6 | 72 | 62/62 | 15/15 | 20/15 | 9/5 |
| rel_medical_indicates | 4 | 72 | 59/66 | 16/16 | 27/19 | 12/11 |

## 4. Supported strata (rule fixed before the analysis)

A stratum effect is "supported" when its CI excludes 0 (cell of at least 30 visits) AND either the means of the feature's interpretable bins are strictly monotone across at least 3 bins for that comparison (`mono:`) OR the same-sign significant effect repeats for both CEI versions and at least two explainers within the same comparison model (`repeats(...)`). Every (feature, bin, comparison model, sign) that passes is listed; sign + means CEI better, - means CEI worse.

### 4.1 CEI better than the other model (d > 0)

| basis | feature | bin | vs | #sig comparisons | of which also differ from the rest | why |
|---|---|---|---|---|---|---|
| raw | n_nodes | >134 | GSAT | 4 | 3 | mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_nodes | >134 | GraphCare | 6 | 6 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_evidence | >70 | GSAT | 4 | 3 | mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_evidence | >70 | GraphCare | 6 | 6 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| raw | n_edges | >225 | GSAT | 4 | 3 | mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_edges | >225 | GraphCare | 6 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| raw | density | <=0.0127 | GSAT | 4 | 3 | mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | density | <=0.0127 | GraphCare | 5 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | num_visits | 1 | GraphCare | 2 | 0 | mono:GNNExp/GC/R1_ls0 |
| raw | num_visits | 2 | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | num_visits | 3 | GraphCare | 5 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | num_visits | 4 | GraphCare | 5 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | num_visits | 5 | GraphCare | 5 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_complaint | 0 | GraphCare | 3 | 1 | repeats(GNNExp,builtin) |
| raw | n_measurement | >60 | GSAT | 4 | 3 | mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_measurement | >60 | GraphCare | 6 | 6 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_diagnosis | <=0 | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| raw | n_diagnosis | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | n_diagnosis | (0,3] | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| raw | n_diagnosis | (0,3] | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_diagnosis | >3 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| raw | n_diagnosis | >3 | GraphCare | 5 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GC/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | share_complaint | <=0.0137 | GraphCare | 5 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | share_measurement | (0.767,0.814] | GSAT | 2 | 1 | mono:bi/GSATIG/R1_ls0 |
| raw | share_measurement | (0.814,0.86] | GSAT | 2 | 1 | mono:bi/GSATIG/R1_ls0 |
| raw | share_measurement | >0.86 | GSAT | 4 | 3 | mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0,bi/GSATIG/R1_ls0 |
| raw | share_measurement | >0.86 | GraphCare | 5 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | share_vital | <=0.0845 | GSAT | 4 | 3 | mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | share_vital | <=0.0845 | GraphCare | 6 | 6 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| raw | share_diagnosis | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | share_diagnosis | (0,0.0441] | GraphCare | 5 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | share_diagnosis | >0.0441 | GraphCare | 3 | 3 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | n_knowledge | >30 | GraphCare | 5 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_analyte | >33 | GSAT | 3 | 2 | mono:IG/GSAT/R1_ls0 |
| raw | n_analyte | >33 | GraphCare | 4 | 4 | repeats(GNNExp,IG,builtin) mono:IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_prior_dx | <=0 | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| raw | n_prior_dx | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | n_prior_dx | (0,3] | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| raw | n_prior_dx | (0,3] | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_prior_dx | >3 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| raw | n_prior_dx | >3 | GraphCare | 5 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GC/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_trajectory_of | >0 | GraphCare | 5 | 5 | repeats(GNNExp,Grad,IG,builtin) |
| raw | rel_recurrence_of | <=0 | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| raw | rel_recurrence_of | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | rel_recurrence_of | (0,3] | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| raw | rel_recurrence_of | (0,3] | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_recurrence_of | >3 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| raw | rel_recurrence_of | >3 | GraphCare | 5 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GC/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_comorbid_with | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/R1_ls0 |
| raw | rel_comorbid_with | (0,2] | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_comorbid_with | >2 | GraphCare | 5 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_co_complaint | 0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | rel_co_complaint | 2 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | rel_co_complaint | 6 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | rel_baseline_of | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | rel_baseline_of | (0,25] | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_baseline_of | >25 | GSAT | 4 | 2 | mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_baseline_of | >25 | GraphCare | 5 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GC/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_total | >87 | GSAT | 3 | 2 | mono:IG/GSAT/R1_ls0 |
| raw | rel_medical_total | >87 | GraphCare | 5 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_member_of | >31 | GSAT | 3 | 2 | mono:IG/GSAT/R1_ls0 |
| raw | rel_medical_member_of | >31 | GraphCare | 3 | 3 | repeats(GNNExp,IG) mono:IG/GC/R1_ls0 |
| raw | rel_medical_assesses | >16 | GSAT | 4 | 4 | mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_medical_assesses | >16 | GraphCare | 4 | 2 | repeats(GNNExp,IG,builtin) mono:IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_measures | (22,26] | GSAT | 2 | 1 | mono:bi/GSATIG/R1_ls0 |
| raw | rel_medical_measures | (26,28] | GSAT | 2 | 0 | mono:bi/GSATIG/R1_ls0 |
| raw | rel_medical_measures | (26,28] | GraphCare | 3 | 1 | repeats(GNNExp,IG) |
| raw | rel_medical_measures | >28 | GSAT | 2 | 1 | mono:bi/GSATIG/R1_ls0 |
| raw | rel_medical_measures | >28 | GraphCare | 3 | 1 | repeats(GNNExp,IG) |
| raw | rel_medical_is_a | (3,10] | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:bi/GCGrad/R1_ls0 |
| raw | rel_medical_is_a | >10 | GraphCare | 5 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_may_affect | 0 | GraphCare | 2 | 1 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | rel_medical_may_affect | 1 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | rel_medical_may_affect | 2 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| raw | rel_medical_may_affect | 3 | GraphCare | 1 | 0 | mono:GNNExp/GC/R1_ls0 |
| raw | rel_medical_indicates | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/R1_ls0 |
| raw | rel_medical_indicates | (0,1] | GraphCare | 2 | 0 | mono:GNNExp/GC/R1_ls0 |
| raw | rel_medical_indicates | (1,3] | GraphCare | 2 | 0 | mono:GNNExp/GC/R1_ls0 |
| raw | rel_medical_indicates | >3 | GraphCare | 3 | 2 | repeats(GNNExp,builtin) mono:GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| adj | n_nodes | >134 | GraphCare | 4 | 4 | repeats(GNNExp,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | n_evidence | <=32 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | n_evidence | (32,41] | GSAT | 2 | 1 | mono:bi/GSATbi/R1_ls0 |
| adj | n_evidence | (41,70] | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | n_evidence | >70 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | n_evidence | >70 | GraphCare | 4 | 4 | repeats(GNNExp,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | n_edges | >225 | GraphCare | 4 | 4 | repeats(GNNExp,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | density | <=0.0127 | GraphCare | 4 | 4 | repeats(GNNExp,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | num_visits | 2 | GraphCare | 3 | 1 | repeats(GNNExp,builtin) |
| adj | num_visits | 3 | GraphCare | 3 | 1 | repeats(GNNExp,builtin) |
| adj | n_measurement | >60 | GraphCare | 4 | 4 | repeats(GNNExp,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | n_diagnosis | <=0 | GSAT | 2 | 1 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| adj | n_diagnosis | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | n_diagnosis | (0,3] | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | n_diagnosis | (0,3] | GraphCare | 3 | 3 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | n_diagnosis | >3 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | n_diagnosis | >3 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | share_measurement | <=0.767 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | share_measurement | <=0.767 | GraphCare | 2 | 0 | mono:GNNExp/GC/final |
| adj | share_measurement | (0.767,0.814] | GSAT | 2 | 1 | mono:bi/GSATbi/R1_ls0 |
| adj | share_measurement | (0.767,0.814] | GraphCare | 2 | 0 | mono:GNNExp/GC/final |
| adj | share_measurement | (0.814,0.86] | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | share_measurement | (0.814,0.86] | GraphCare | 2 | 0 | mono:GNNExp/GC/final |
| adj | share_measurement | >0.86 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | share_measurement | >0.86 | GraphCare | 4 | 4 | repeats(GNNExp,IG,builtin) mono:GNNExp/GC/final |
| adj | share_vital | <=0.0845 | GraphCare | 4 | 4 | repeats(GNNExp,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | share_diagnosis | <=0 | GSAT | 2 | 1 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| adj | share_diagnosis | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | share_diagnosis | (0,0.0441] | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| adj | share_diagnosis | (0,0.0441] | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| adj | share_diagnosis | >0.0441 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | share_diagnosis | >0.0441 | GraphCare | 3 | 3 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| adj | n_knowledge | <=21 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | n_knowledge | (21,25] | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | n_knowledge | (25,30] | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | n_knowledge | >30 | GraphCare | 3 | 3 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | n_prior_dx | <=0 | GSAT | 2 | 1 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| adj | n_prior_dx | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | n_prior_dx | (0,3] | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | n_prior_dx | (0,3] | GraphCare | 3 | 3 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | n_prior_dx | >3 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | n_prior_dx | >3 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_trajectory_of | >0 | GraphCare | 4 | 4 | repeats(GNNExp,IG,builtin) |
| adj | rel_recurrence_of | <=0 | GSAT | 2 | 1 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| adj | rel_recurrence_of | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_recurrence_of | (0,3] | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | rel_recurrence_of | (0,3] | GraphCare | 3 | 3 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_recurrence_of | >3 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | rel_recurrence_of | >3 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_comorbid_with | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_comorbid_with | (0,2] | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_comorbid_with | >2 | GraphCare | 3 | 3 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_co_complaint | 0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_co_complaint | 2 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_co_complaint | 6 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_baseline_of | <=0 | GSAT | 2 | 2 | mono:bi/GSATbi/R1_ls0 |
| adj | rel_baseline_of | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_baseline_of | (0,25] | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | rel_baseline_of | (0,25] | GraphCare | 3 | 1 | repeats(GNNExp,builtin) mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_baseline_of | >25 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | rel_baseline_of | >25 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_medical_is_a | <=1 | GSAT | 2 | 1 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| adj | rel_medical_is_a | (1,3] | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| adj | rel_medical_is_a | (3,10] | GSAT | 2 | 0 | mono:bi/GSATbi/R1_ls0,bi/GSATIG/R1_ls0 |
| adj | rel_medical_is_a | (3,10] | GraphCare | 3 | 2 | repeats(GNNExp,builtin) |
| adj | rel_medical_is_a | >10 | GSAT | 1 | 0 | mono:bi/GSATbi/R1_ls0 |
| adj | rel_medical_may_affect | 0 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_medical_may_affect | 1 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_medical_may_affect | 2 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_medical_may_affect | 3 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_medical_indicates | <=0 | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_medical_indicates | (0,1] | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_medical_indicates | (1,3] | GraphCare | 2 | 0 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |
| adj | rel_medical_indicates | >3 | GraphCare | 2 | 2 | mono:GNNExp/GC/final,GNNExp/GC/R1_ls0 |

### 4.2 CEI worse than the other model (d < 0)

| basis | feature | bin | vs | #sig comparisons | of which also differ from the rest | why |
|---|---|---|---|---|---|---|
| raw | n_nodes | <=81 | GSAT | 8 | 7 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_nodes | <=81 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_nodes | (81,99] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_nodes | (81,99] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_nodes | (99,134] | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | n_nodes | (99,134] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final |
| raw | n_nodes | >134 | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | n_evidence | <=32 | GSAT | 8 | 7 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_evidence | <=32 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| raw | n_evidence | (32,41] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_evidence | (32,41] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| raw | n_evidence | (41,70] | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | n_evidence | (41,70] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| raw | n_evidence | >70 | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | n_edges | <=125 | GSAT | 8 | 7 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_edges | <=125 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| raw | n_edges | (125,157] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_edges | (125,157] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| raw | n_edges | (157,225] | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final |
| raw | n_edges | (157,225] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| raw | n_edges | >225 | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final |
| raw | density | <=0.0127 | GSAT | 4 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | density | (0.0127,0.016] | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | density | (0.0127,0.016] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| raw | density | (0.016,0.0191] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | density | (0.016,0.0191] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | density | >0.0191 | GSAT | 9 | 8 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | density | >0.0191 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | num_visits | 1 | GSAT | 8 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,Grad/GSAT/R1_ls0 |
| raw | num_visits | 1 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | num_visits | 2 | GSAT | 8 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,Grad/GSAT/R1_ls0 |
| raw | num_visits | 2 | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| raw | num_visits | 3 | GSAT | 6 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final |
| raw | num_visits | 3 | GraphCare | 2 | 0 | mono:IG/GC/final,Grad/GC/final |
| raw | num_visits | 4 | GSAT | 3 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | n_complaint | 1 | GSAT | 8 | 4 | repeats(GNNExp,Grad,IG,builtin) |
| raw | n_complaint | 1 | GraphCare | 6 | 5 | repeats(Grad,IG,builtin) |
| raw | n_complaint | 2 | GSAT | 6 | 1 | repeats(Grad,IG,builtin) |
| raw | n_complaint | 2 | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) |
| raw | n_complaint | 3 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| raw | n_measurement | <=25 | GSAT | 9 | 8 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_measurement | <=25 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_measurement | (25,33] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_measurement | (25,33] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_measurement | (33,60] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_measurement | (33,60] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,IG/GC/R1_ls0 |
| raw | n_measurement | >60 | GSAT | 4 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | n_vital | 6 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) |
| raw | n_vital | 6 | GraphCare | 6 | 5 | repeats(Grad,IG,builtin) |
| raw | n_diagnosis | <=0 | GSAT | 8 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0,GNNExp/GSAT/R1_ls0 |
| raw | n_diagnosis | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_diagnosis | (0,3] | GSAT | 5 | 0 | mono:IG/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| raw | n_diagnosis | (0,3] | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| raw | n_diagnosis | >3 | GSAT | 5 | 0 | mono:IG/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| raw | n_diagnosis | >3 | GraphCare | 1 | 0 | mono:Grad/GC/final |
| raw | share_complaint | <=0.0137 | GSAT | 5 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final |
| raw | share_complaint | <=0.0137 | GraphCare | 1 | 0 | mono:Grad/GC/final |
| raw | share_complaint | (0.0137,0.027] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | share_complaint | (0.0137,0.027] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| raw | share_complaint | (0.027,0.0444] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | share_complaint | (0.027,0.0444] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | share_complaint | >0.0444 | GSAT | 9 | 8 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | share_complaint | >0.0444 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | share_measurement | <=0.767 | GSAT | 9 | 8 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0,bi/GSATIG/R1_ls0 |
| raw | share_measurement | <=0.767 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | share_measurement | (0.767,0.814] | GSAT | 6 | 4 | repeats(Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | share_measurement | (0.767,0.814] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | share_measurement | (0.814,0.86] | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| raw | share_measurement | (0.814,0.86] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| raw | share_measurement | >0.86 | GSAT | 4 | 0 | mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final |
| raw | share_vital | <=0.0845 | GSAT | 4 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | share_vital | (0.0845,0.146] | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,Grad/GSAT/R1_ls0 |
| raw | share_vital | (0.0845,0.146] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| raw | share_vital | (0.146,0.182] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | share_vital | (0.146,0.182] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| raw | share_vital | >0.182 | GSAT | 8 | 7 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | share_vital | >0.182 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| raw | share_diagnosis | <=0 | GSAT | 8 | 4 | repeats(GNNExp,Grad,IG,builtin) |
| raw | share_diagnosis | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) |
| raw | share_diagnosis | >0.0441 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) |
| raw | n_knowledge | <=21 | GSAT | 8 | 7 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_knowledge | <=21 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_knowledge | (21,25] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_knowledge | (21,25] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_knowledge | (25,30] | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | n_knowledge | (25,30] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| raw | n_knowledge | >30 | GSAT | 4 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | n_knowledge | >30 | GraphCare | 1 | 0 | mono:Grad/GC/final |
| raw | n_analyte | <=23 | GSAT | 9 | 8 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_analyte | <=23 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_analyte | (23,29] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | n_analyte | (23,29] | GraphCare | 6 | 2 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_analyte | (29,33] | GSAT | 5 | 0 | mono:Grad/GSAT/final |
| raw | n_analyte | (29,33] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| raw | n_analyte | >33 | GSAT | 4 | 0 | mono:Grad/GSAT/final |
| raw | n_analyte | >33 | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| raw | n_prior_dx | <=0 | GSAT | 8 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0,GNNExp/GSAT/R1_ls0 |
| raw | n_prior_dx | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | n_prior_dx | (0,3] | GSAT | 5 | 0 | mono:IG/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| raw | n_prior_dx | (0,3] | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| raw | n_prior_dx | >3 | GSAT | 5 | 0 | mono:IG/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| raw | n_prior_dx | >3 | GraphCare | 1 | 0 | mono:Grad/GC/final |
| raw | rel_trajectory_of | <=0 | GSAT | 8 | 4 | repeats(GNNExp,Grad,IG,builtin) |
| raw | rel_trajectory_of | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) |
| raw | rel_recurrence_of | <=0 | GSAT | 8 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0,GNNExp/GSAT/R1_ls0 |
| raw | rel_recurrence_of | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_recurrence_of | (0,3] | GSAT | 5 | 0 | mono:IG/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| raw | rel_recurrence_of | (0,3] | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| raw | rel_recurrence_of | >3 | GSAT | 5 | 0 | mono:IG/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| raw | rel_recurrence_of | >3 | GraphCare | 1 | 0 | mono:Grad/GC/final |
| raw | rel_comorbid_with | <=0 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| raw | rel_comorbid_with | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_comorbid_with | (0,2] | GSAT | 8 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| raw | rel_comorbid_with | (0,2] | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| raw | rel_comorbid_with | >2 | GraphCare | 2 | 0 | mono:IG/GC/final,Grad/GC/final |
| raw | rel_co_complaint | 0 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,GNNExp/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_co_complaint | 0 | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final |
| raw | rel_co_complaint | 2 | GSAT | 6 | 1 | repeats(Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_co_complaint | 2 | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final |
| raw | rel_co_complaint | 6 | GSAT | 3 | 0 | mono:IG/GSAT/final |
| raw | rel_co_complaint | 6 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final |
| raw | rel_baseline_of | <=0 | GSAT | 8 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_baseline_of | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_baseline_of | (0,25] | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_baseline_of | (0,25] | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| raw | rel_baseline_of | >25 | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | rel_medical_total | <=63 | GSAT | 8 | 7 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_medical_total | <=63 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_total | (63,75] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_medical_total | (63,75] | GraphCare | 6 | 4 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_total | (75,87] | GSAT | 6 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,Grad/GSAT/R1_ls0 |
| raw | rel_medical_total | (75,87] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| raw | rel_medical_total | >87 | GSAT | 3 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | rel_medical_total | >87 | GraphCare | 2 | 0 | mono:IG/GC/final,Grad/GC/final |
| raw | rel_medical_member_of | <=21 | GSAT | 9 | 8 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,GNNExp/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_medical_member_of | <=21 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_member_of | (21,28] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,GNNExp/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_medical_member_of | (21,28] | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_member_of | (28,31] | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final,GNNExp/GSAT/final |
| raw | rel_medical_member_of | (28,31] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| raw | rel_medical_member_of | >31 | GSAT | 4 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| raw | rel_medical_member_of | >31 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| raw | rel_medical_assesses | <=12 | GSAT | 8 | 7 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0,GNNExp/GSAT/R1_ls0 |
| raw | rel_medical_assesses | <=12 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_assesses | (12,15] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_medical_assesses | (12,15] | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_assesses | (15,16] | GSAT | 5 | 0 | mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| raw | rel_medical_assesses | (15,16] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| raw | rel_medical_assesses | >16 | GSAT | 3 | 0 | mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATIG/final |
| raw | rel_medical_assesses | >16 | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| raw | rel_medical_measures | <=22 | GSAT | 9 | 8 | repeats(GNNExp,Grad,IG,builtin) mono:bi/GSATIG/final,IG/GSAT/R1_ls0,GNNExp/GSAT/R1_ls0,bi/GSATIG/R1_ls0 |
| raw | rel_medical_measures | <=22 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:Grad/GC/final,bi/GCGrad/final,bi/GCGrad/R1_ls0 |
| raw | rel_medical_measures | (22,26] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:bi/GSATIG/final,IG/GSAT/R1_ls0 |
| raw | rel_medical_measures | (22,26] | GraphCare | 6 | 2 | repeats(Grad,IG,builtin) mono:Grad/GC/final,bi/GCGrad/final,bi/GCGrad/R1_ls0 |
| raw | rel_medical_measures | (26,28] | GSAT | 3 | 0 | mono:bi/GSATIG/final |
| raw | rel_medical_measures | (26,28] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:Grad/GC/final,bi/GCGrad/final |
| raw | rel_medical_measures | >28 | GSAT | 5 | 0 | mono:bi/GSATIG/final |
| raw | rel_medical_measures | >28 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:Grad/GC/final,bi/GCGrad/final |
| raw | rel_medical_is_a | <=1 | GSAT | 8 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_medical_is_a | <=1 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_is_a | (1,3] | GSAT | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| raw | rel_medical_is_a | (1,3] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_is_a | (3,10] | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| raw | rel_medical_is_a | (3,10] | GraphCare | 3 | 0 | mono:Grad/GC/final,bi/GCGrad/final |
| raw | rel_medical_is_a | >10 | GSAT | 5 | 0 | mono:IG/GSAT/final |
| raw | rel_medical_is_a | >10 | GraphCare | 1 | 0 | mono:Grad/GC/final |
| raw | rel_medical_may_affect | 0 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| raw | rel_medical_may_affect | 0 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| raw | rel_medical_may_affect | 1 | GSAT | 8 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| raw | rel_medical_may_affect | 1 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) |
| raw | rel_medical_may_affect | 2 | GSAT | 6 | 0 | repeats(Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| raw | rel_medical_may_affect | 2 | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) |
| raw | rel_medical_may_affect | 3 | GSAT | 5 | 0 | repeats(Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| raw | rel_medical_may_affect | 3 | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) |
| raw | rel_medical_indicates | <=0 | GSAT | 8 | 6 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0 |
| raw | rel_medical_indicates | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_indicates | (0,1] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0 |
| raw | rel_medical_indicates | (0,1] | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| raw | rel_medical_indicates | (1,3] | GSAT | 4 | 0 | mono:IG/GSAT/final |
| raw | rel_medical_indicates | (1,3] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| raw | rel_medical_indicates | >3 | GSAT | 4 | 0 | mono:IG/GSAT/final |
| raw | rel_medical_indicates | >3 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| adj | n_nodes | <=81 | GSAT | 8 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,bi/GSATbi/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_nodes | <=81 | GraphCare | 5 | 2 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_nodes | (81,99] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,bi/GSATbi/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_nodes | (81,99] | GraphCare | 6 | 5 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_nodes | (99,134] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,bi/GSATbi/final,Grad/GSAT/R1_ls0 |
| adj | n_nodes | (99,134] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_nodes | >134 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,bi/GSATbi/final |
| adj | n_nodes | >134 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | n_evidence | <=32 | GSAT | 8 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_evidence | <=32 | GraphCare | 6 | 3 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_evidence | (32,41] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_evidence | (32,41] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_evidence | (41,70] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_evidence | (41,70] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_evidence | >70 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | n_evidence | >70 | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | n_edges | <=125 | GSAT | 8 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_edges | <=125 | GraphCare | 6 | 3 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_edges | (125,157] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_edges | (125,157] | GraphCare | 6 | 5 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_edges | (157,225] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final,Grad/GSAT/R1_ls0 |
| adj | n_edges | (157,225] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_edges | >225 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,bi/GSATbi/final |
| adj | n_edges | >225 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | density | <=0.0127 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) |
| adj | density | <=0.0127 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | density | (0.0127,0.016] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | density | (0.0127,0.016] | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | density | (0.016,0.0191] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | density | (0.016,0.0191] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | density | >0.0191 | GSAT | 8 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | density | >0.0191 | GraphCare | 5 | 2 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | num_visits | 1 | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| adj | num_visits | 1 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | num_visits | 2 | GSAT | 8 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| adj | num_visits | 2 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | num_visits | 3 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| adj | num_visits | 3 | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | num_visits | 4 | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | num_visits | 5 | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | n_complaint | 0 | GSAT | 7 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_complaint | 0 | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0 |
| adj | n_complaint | 1 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_complaint | 1 | GraphCare | 6 | 5 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0,Grad/GC/R1_ls0 |
| adj | n_complaint | 2 | GSAT | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_complaint | 2 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:Grad/GC/R1_ls0 |
| adj | n_measurement | <=25 | GSAT | 8 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_measurement | <=25 | GraphCare | 6 | 3 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_measurement | (25,33] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_measurement | (25,33] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_measurement | (33,60] | GSAT | 8 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_measurement | (33,60] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_measurement | >60 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | n_measurement | >60 | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | n_vital | 6 | GSAT | 8 | 0 | repeats(GNNExp,Grad,IG,builtin) |
| adj | n_vital | 6 | GraphCare | 6 | 4 | repeats(Grad,IG,builtin) |
| adj | n_diagnosis | <=0 | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| adj | n_diagnosis | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_diagnosis | (0,3] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,GNNExp/GSAT/R1_ls0 |
| adj | n_diagnosis | (0,3] | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | n_diagnosis | >3 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,GNNExp/GSAT/R1_ls0 |
| adj | n_diagnosis | >3 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | share_complaint | <=0.0137 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | share_complaint | <=0.0137 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| adj | share_complaint | (0.0137,0.027] | GSAT | 8 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | share_complaint | (0.0137,0.027] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) |
| adj | share_complaint | (0.027,0.0444] | GSAT | 7 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | share_complaint | (0.027,0.0444] | GraphCare | 6 | 5 | repeats(Grad,IG,builtin) |
| adj | share_complaint | >0.0444 | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | share_complaint | >0.0444 | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) |
| adj | share_measurement | <=0.767 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | share_measurement | <=0.767 | GraphCare | 5 | 2 | repeats(Grad,IG,builtin) |
| adj | share_measurement | (0.767,0.814] | GSAT | 6 | 2 | repeats(Grad,IG,builtin) mono:IG/GSAT/final |
| adj | share_measurement | (0.767,0.814] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) |
| adj | share_measurement | (0.814,0.86] | GSAT | 8 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | share_measurement | (0.814,0.86] | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) |
| adj | share_measurement | >0.86 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | share_measurement | >0.86 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| adj | share_vital | <=0.0845 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final |
| adj | share_vital | (0.0845,0.146] | GSAT | 8 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | share_vital | (0.0845,0.146] | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | share_vital | (0.146,0.182] | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | share_vital | (0.146,0.182] | GraphCare | 6 | 5 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | share_vital | >0.182 | GSAT | 8 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | share_vital | >0.182 | GraphCare | 6 | 3 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | share_diagnosis | <=0 | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| adj | share_diagnosis | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final,bi/GCGrad/R1_ls0 |
| adj | share_diagnosis | (0,0.0441] | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,GNNExp/GSAT/R1_ls0 |
| adj | share_diagnosis | (0,0.0441] | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| adj | share_diagnosis | >0.0441 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,GNNExp/GSAT/R1_ls0 |
| adj | share_diagnosis | >0.0441 | GraphCare | 3 | 0 | mono:IG/GC/final,Grad/GC/final,bi/GCGrad/final |
| adj | n_knowledge | <=21 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_knowledge | <=21 | GraphCare | 6 | 3 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | n_knowledge | (21,25] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_knowledge | (21,25] | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | n_knowledge | (25,30] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_knowledge | (25,30] | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| adj | n_knowledge | >30 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | n_knowledge | >30 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| adj | n_analyte | <=23 | GSAT | 8 | 6 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| adj | n_analyte | <=23 | GraphCare | 5 | 1 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | n_analyte | (23,29] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| adj | n_analyte | (23,29] | GraphCare | 6 | 4 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | n_analyte | (29,33] | GSAT | 8 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| adj | n_analyte | (29,33] | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | n_analyte | >33 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| adj | n_prior_dx | <=0 | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| adj | n_prior_dx | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | n_prior_dx | (0,3] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,GNNExp/GSAT/R1_ls0 |
| adj | n_prior_dx | (0,3] | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | n_prior_dx | >3 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,GNNExp/GSAT/R1_ls0 |
| adj | n_prior_dx | >3 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | rel_trajectory_of | <=0 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) |
| adj | rel_trajectory_of | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) |
| adj | rel_trajectory_of | >0 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) |
| adj | rel_recurrence_of | <=0 | GSAT | 7 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final |
| adj | rel_recurrence_of | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | rel_recurrence_of | (0,3] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,GNNExp/GSAT/R1_ls0 |
| adj | rel_recurrence_of | (0,3] | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | rel_recurrence_of | >3 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:GNNExp/GSAT/final,bi/GSATbi/final,bi/GSATIG/final,GNNExp/GSAT/R1_ls0 |
| adj | rel_recurrence_of | >3 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | rel_comorbid_with | <=0 | GSAT | 8 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| adj | rel_comorbid_with | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | rel_comorbid_with | (0,2] | GSAT | 8 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| adj | rel_comorbid_with | (0,2] | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | rel_comorbid_with | >2 | GSAT | 8 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/R1_ls0 |
| adj | rel_comorbid_with | >2 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | rel_co_complaint | 0 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,GNNExp/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_co_complaint | 0 | GraphCare | 6 | 4 | repeats(Grad,IG,builtin) mono:Grad/GC/final,bi/GCGrad/final,IG/GC/R1_ls0,Grad/GC/R1_ls0,bi/GCGrad/R1_ls0 |
| adj | rel_co_complaint | 2 | GSAT | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GSAT/final,Grad/GSAT/final,IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_co_complaint | 2 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:Grad/GC/final,bi/GCGrad/final,Grad/GC/R1_ls0 |
| adj | rel_co_complaint | 6 | GSAT | 3 | 0 | mono:IG/GSAT/final,Grad/GSAT/final |
| adj | rel_co_complaint | 6 | GraphCare | 3 | 0 | mono:Grad/GC/final,bi/GCGrad/final |
| adj | rel_baseline_of | <=0 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_baseline_of | <=0 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | rel_baseline_of | (0,25] | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_baseline_of | (0,25] | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | rel_baseline_of | >25 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) |
| adj | rel_baseline_of | >25 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | rel_medical_total | <=63 | GSAT | 8 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_medical_total | <=63 | GraphCare | 5 | 1 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | rel_medical_total | (63,75] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_medical_total | (63,75] | GraphCare | 6 | 4 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | rel_medical_total | (75,87] | GSAT | 8 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_medical_total | (75,87] | GraphCare | 6 | 4 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | rel_medical_total | >87 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_member_of | <=21 | GSAT | 8 | 5 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| adj | rel_medical_member_of | <=21 | GraphCare | 5 | 1 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | rel_medical_member_of | (21,28] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| adj | rel_medical_member_of | (21,28] | GraphCare | 6 | 3 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | rel_medical_member_of | (28,31] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| adj | rel_medical_member_of | (28,31] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_member_of | >31 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final |
| adj | rel_medical_member_of | >31 | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_assesses | <=12 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) |
| adj | rel_medical_assesses | <=12 | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_assesses | (12,15] | GSAT | 7 | 1 | repeats(GNNExp,Grad,IG,builtin) |
| adj | rel_medical_assesses | (12,15] | GraphCare | 6 | 5 | repeats(Grad,IG,builtin) |
| adj | rel_medical_assesses | (15,16] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) |
| adj | rel_medical_assesses | (15,16] | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_assesses | >16 | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_measures | <=22 | GSAT | 7 | 4 | repeats(GNNExp,Grad,IG,builtin) mono:bi/GSATIG/final,IG/GSAT/R1_ls0 |
| adj | rel_medical_measures | <=22 | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | rel_medical_measures | (22,26] | GSAT | 7 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:bi/GSATIG/final,IG/GSAT/R1_ls0 |
| adj | rel_medical_measures | (22,26] | GraphCare | 6 | 4 | repeats(Grad,IG,builtin) mono:IG/GC/R1_ls0 |
| adj | rel_medical_measures | (26,28] | GSAT | 4 | 0 | mono:bi/GSATIG/final |
| adj | rel_medical_measures | (26,28] | GraphCare | 5 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_measures | >28 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:bi/GSATIG/final |
| adj | rel_medical_measures | >28 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_is_a | <=1 | GSAT | 8 | 3 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0 |
| adj | rel_medical_is_a | <=1 | GraphCare | 6 | 6 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | rel_medical_is_a | (1,3] | GSAT | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GSAT/R1_ls0 |
| adj | rel_medical_is_a | (1,3] | GraphCare | 6 | 4 | repeats(Grad,IG,builtin) mono:IG/GC/final,IG/GC/R1_ls0 |
| adj | rel_medical_is_a | (3,10] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0 |
| adj | rel_medical_is_a | (3,10] | GraphCare | 3 | 0 | mono:IG/GC/final |
| adj | rel_medical_is_a | >10 | GSAT | 8 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0 |
| adj | rel_medical_is_a | >10 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) mono:IG/GC/final |
| adj | rel_medical_may_affect | 0 | GSAT | 8 | 2 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_medical_may_affect | 0 | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_may_affect | 1 | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_medical_may_affect | 1 | GraphCare | 6 | 2 | repeats(Grad,IG,builtin) |
| adj | rel_medical_may_affect | 2 | GSAT | 6 | 0 | repeats(Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_medical_may_affect | 2 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_may_affect | 3 | GSAT | 5 | 3 | repeats(Grad,IG,builtin) mono:IG/GSAT/R1_ls0,Grad/GSAT/R1_ls0 |
| adj | rel_medical_may_affect | 3 | GraphCare | 5 | 1 | repeats(Grad,IG,builtin) |
| adj | rel_medical_indicates | <=0 | GSAT | 8 | 1 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| adj | rel_medical_indicates | <=0 | GraphCare | 6 | 0 | repeats(Grad,IG,builtin) |
| adj | rel_medical_indicates | (0,1] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| adj | rel_medical_indicates | (0,1] | GraphCare | 6 | 2 | repeats(Grad,IG,builtin) |
| adj | rel_medical_indicates | (1,3] | GSAT | 7 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final,IG/GSAT/R1_ls0 |
| adj | rel_medical_indicates | (1,3] | GraphCare | 6 | 3 | repeats(Grad,IG,builtin) |
| adj | rel_medical_indicates | >3 | GSAT | 6 | 0 | repeats(GNNExp,Grad,IG,builtin) mono:Grad/GSAT/final |
| adj | rel_medical_indicates | >3 | GraphCare | 4 | 0 | repeats(Grad,IG,builtin) |

## 5. Strata that differ from the rest of the visits

The contrast is stratum mean minus the mean of all other visits, bootstrapped like d. Listed: strata whose contrast CI excludes 0 in the same direction in both the raw and the adjusted analysis for at least 3 of the 18 comparisons (cells of at least 30 visits). This tests whether the stratum behaves differently from the average visit, which is the question behind "where does CEI-GNN explain better".

| feature | bin | direction | raw & adj significant in (comparisons) |
|---|---|---|---|
| n_nodes | <=81 | CEI relatively worse | 7: IG/GSAT/final, IG/GC/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, GNNExp/GSAT/R1_ls0 |
| n_nodes | (81,99] | CEI relatively better | 3: GNNExp/GSAT/final, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| n_nodes | (81,99] | CEI relatively worse | 9: Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_nodes | >134 | CEI relatively better | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_evidence | <=32 | CEI relatively worse | 8: IG/GSAT/final, IG/GC/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GSAT/R1_ls0 |
| n_evidence | (32,41] | CEI relatively worse | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_evidence | >70 | CEI relatively better | 11: IG/GSAT/final, IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_edges | <=125 | CEI relatively worse | 8: IG/GSAT/final, IG/GC/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GSAT/R1_ls0 |
| n_edges | (125,157] | CEI relatively better | 3: GNNExp/GSAT/final, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| n_edges | (125,157] | CEI relatively worse | 9: Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_edges | >225 | CEI relatively better | 9: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| density | <=0.0127 | CEI relatively better | 11: IG/GSAT/final, IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| density | (0.016,0.0191] | CEI relatively better | 3: GNNExp/GSAT/final, bi/GSATIG/final, bi/GSATIG/R1_ls0 |
| density | (0.016,0.0191] | CEI relatively worse | 8: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| density | >0.0191 | CEI relatively worse | 8: IG/GSAT/final, IG/GC/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| num_visits | 1 | CEI relatively worse | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| num_visits | 2 | CEI relatively better | 4: Grad/GC/final, bi/GCGrad/final, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| num_visits | 3 | CEI relatively better | 6: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| num_visits | 4 | CEI relatively better | 9: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| num_visits | 5 | CEI relatively better | 8: IG/GSAT/final, IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0 |
| n_complaint | 1 | CEI relatively worse | 10: Grad/GC/final, GNNExp/GSAT/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GSAT/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_complaint | 2 | CEI relatively better | 4: GNNExp/GSAT/final, GNNExp/GC/final, GNNExp/GSAT/R1_ls0, GNNExp/GC/R1_ls0 |
| n_complaint | 3 | CEI relatively better | 4: GNNExp/GSAT/final, GNNExp/GC/final, IG/GSAT/R1_ls0, GNNExp/GC/R1_ls0 |
| n_measurement | <=25 | CEI relatively worse | 8: IG/GSAT/final, IG/GC/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GSAT/R1_ls0 |
| n_measurement | (25,33] | CEI relatively better | 3: GNNExp/GSAT/final, bi/GSATIG/final, bi/GSATIG/R1_ls0 |
| n_measurement | (25,33] | CEI relatively worse | 8: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_measurement | >60 | CEI relatively better | 11: IG/GSAT/final, IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_vital | 6 | CEI relatively worse | 5: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, Grad/GC/R1_ls0 |
| n_diagnosis | <=0 | CEI relatively worse | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_diagnosis | (0,3] | CEI relatively better | 7: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_diagnosis | >3 | CEI relatively better | 7: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_complaint | <=0.0137 | CEI relatively better | 6: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_complaint | (0.027,0.0444] | CEI relatively worse | 8: Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_measurement | <=0.767 | CEI relatively worse | 6: IG/GSAT/final, IG/GC/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, bi/GSATIG/R1_ls0 |
| share_measurement | (0.767,0.814] | CEI relatively worse | 9: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_measurement | >0.86 | CEI relatively better | 10: IG/GSAT/final, IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_vital | <=0.0845 | CEI relatively better | 11: IG/GSAT/final, IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_vital | (0.146,0.182] | CEI relatively better | 3: GNNExp/GSAT/final, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| share_vital | (0.146,0.182] | CEI relatively worse | 9: Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_vital | >0.182 | CEI relatively worse | 8: IG/GSAT/final, IG/GC/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GSAT/R1_ls0 |
| share_diagnosis | <=0 | CEI relatively worse | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_diagnosis | (0,0.0441] | CEI relatively better | 9: IG/GSAT/final, IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_diagnosis | >0.0441 | CEI relatively better | 8: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| share_diagnosis | >0.0441 | CEI relatively worse | 3: GNNExp/GSAT/final, bi/GSATIG/final, bi/GSATIG/R1_ls0 |
| n_knowledge | <=21 | CEI relatively worse | 6: Grad/GC/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GSAT/R1_ls0 |
| n_knowledge | (21,25] | CEI relatively better | 3: GNNExp/GSAT/final, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| n_knowledge | (21,25] | CEI relatively worse | 6: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_knowledge | (25,30] | CEI relatively better | 3: Grad/GC/final, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_knowledge | >30 | CEI relatively better | 8: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_analyte | <=23 | CEI relatively worse | 8: IG/GSAT/final, GNNExp/GSAT/final, bi/GSATIG/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| n_analyte | (23,29] | CEI relatively better | 4: GNNExp/GSAT/final, bi/GSATIG/final, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| n_analyte | (23,29] | CEI relatively worse | 3: Grad/GC/final, GNNExp/GC/final, Grad/GC/R1_ls0 |
| n_analyte | >33 | CEI relatively better | 11: IG/GSAT/final, IG/GC/final, Grad/GSAT/final, Grad/GC/final, GNNExp/GSAT/final, GNNExp/GC/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0 |
| n_prior_dx | <=0 | CEI relatively worse | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_prior_dx | (0,3] | CEI relatively better | 7: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| n_prior_dx | >3 | CEI relatively better | 7: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_trajectory_of | <=0 | CEI relatively worse | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_trajectory_of | >0 | CEI relatively better | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_recurrence_of | <=0 | CEI relatively worse | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_recurrence_of | (0,3] | CEI relatively better | 7: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_recurrence_of | >3 | CEI relatively better | 7: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_comorbid_with | <=0 | CEI relatively worse | 7: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_comorbid_with | (0,2] | CEI relatively better | 4: Grad/GC/final, bi/GCGrad/final, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_comorbid_with | >2 | CEI relatively better | 8: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_co_complaint | 0 | CEI relatively worse | 5: GNNExp/GSAT/final, GNNExp/GC/final, IG/GSAT/R1_ls0, GNNExp/GSAT/R1_ls0, GNNExp/GC/R1_ls0 |
| rel_co_complaint | 2 | CEI relatively better | 4: GNNExp/GSAT/final, GNNExp/GC/final, GNNExp/GSAT/R1_ls0, GNNExp/GC/R1_ls0 |
| rel_co_complaint | 6 | CEI relatively better | 4: GNNExp/GSAT/final, GNNExp/GC/final, IG/GSAT/R1_ls0, GNNExp/GC/R1_ls0 |
| rel_baseline_of | <=0 | CEI relatively worse | 10: IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_baseline_of | (0,25] | CEI relatively better | 6: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_baseline_of | (0,25] | CEI relatively worse | 4: GNNExp/GSAT/final, bi/GSATIG/final, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| rel_baseline_of | >25 | CEI relatively better | 10: IG/GSAT/final, IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_medical_total | <=63 | CEI relatively worse | 6: IG/GSAT/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| rel_medical_total | (63,75] | CEI relatively better | 4: GNNExp/GSAT/final, bi/GSATIG/final, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| rel_medical_total | (63,75] | CEI relatively worse | 4: Grad/GC/final, bi/GCGrad/final, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_medical_total | >87 | CEI relatively better | 11: IG/GSAT/final, IG/GC/final, Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_medical_member_of | <=21 | CEI relatively worse | 7: IG/GSAT/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| rel_medical_member_of | >31 | CEI relatively better | 6: GNNExp/GSAT/final, GNNExp/GC/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, GNNExp/GSAT/R1_ls0, GNNExp/GC/R1_ls0 |
| rel_medical_assesses | >16 | CEI relatively better | 6: IG/GC/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0 |
| rel_medical_measures | <=22 | CEI relatively worse | 5: IG/GSAT/final, GNNExp/GSAT/final, IG/GSAT/R1_ls0, GNNExp/GSAT/R1_ls0, bi/GSATIG/R1_ls0 |
| rel_medical_measures | (22,26] | CEI relatively worse | 3: Grad/GC/final, GNNExp/GC/final, bi/GCGrad/final |
| rel_medical_measures | (26,28] | CEI relatively better | 3: GNNExp/GSAT/final, IG/GSAT/R1_ls0, Grad/GSAT/R1_ls0 |
| rel_medical_is_a | <=1 | CEI relatively worse | 11: IG/GC/final, Grad/GC/final, GNNExp/GSAT/final, GNNExp/GC/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GSAT/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_medical_is_a | (1,3] | CEI relatively better | 4: GNNExp/GSAT/final, GNNExp/GC/final, GNNExp/GSAT/R1_ls0, GNNExp/GC/R1_ls0 |
| rel_medical_is_a | (1,3] | CEI relatively worse | 4: Grad/GC/final, bi/GCGrad/final, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_medical_is_a | (3,10] | CEI relatively better | 6: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_medical_is_a | >10 | CEI relatively better | 7: IG/GC/final, Grad/GC/final, bi/GCGrad/final, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GCGrad/R1_ls0 |
| rel_medical_indicates | >3 | CEI relatively better | 9: Grad/GC/final, bi/GSATbi/final, bi/GCGrad/final, IG/GSAT/R1_ls0, IG/GC/R1_ls0, Grad/GC/R1_ls0, GNNExp/GC/R1_ls0, bi/GSATbi/R1_ls0, bi/GCGrad/R1_ls0 |

## 6. Every stratum: paired difference per feature

Rows are the bins of the feature; n and n patients refer to the visits in the bin (GSAT IG comparison). Sections 6.1-6.4: raw and confidence-adjusted, cei_final and cei_R1_ls0.

### 6.1 cei_final, raw

**n_nodes**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=81 | 339 | 317 | -0.0762 [-0.0853, -0.0677]* | -0.0883 [-0.0970, -0.0792]* | -0.0680 [-0.0770, -0.0591]* | -0.1222 [-0.1333, -0.1107]* | -0.0703 [-0.0834, -0.0585]* | +0.0666 [+0.0474, +0.0850]* | -0.0221 [-0.0329, -0.0115]* | -0.0552 [-0.0652, -0.0449]* | -0.0910 [-0.1033, -0.0784]* |
| (81,99] | 338 | 336 | -0.0532 [-0.0608, -0.0458]* | -0.0611 [-0.0671, -0.0546]* | -0.0489 [-0.0560, -0.0418]* | -0.1217 [-0.1317, -0.1110]* | -0.0149 [-0.0253, -0.0043]* | +0.0453 [+0.0303, +0.0605]* | -0.0157 [-0.0248, -0.0060]* | -0.0278 [-0.0363, -0.0191]* | -0.0939 [-0.1043, -0.0829]* |
| (99,134] | 345 | 327 | -0.0437 [-0.0510, -0.0362]* | -0.0476 [-0.0551, -0.0405]* | -0.0420 [-0.0490, -0.0344]* | -0.0691 [-0.0786, -0.0597]* | -0.0173 [-0.0280, -0.0055]* | +0.0681 [+0.0522, +0.0853]* | -0.0173 [-0.0253, -0.0094]* | -0.0287 [-0.0363, -0.0208]* | -0.0518 [-0.0621, -0.0418]* |
| >134 | 323 | 240 | -0.0282 [-0.0370, -0.0195]* | +0.0064 [-0.0028, +0.0148] | -0.0293 [-0.0386, -0.0200]* | +0.0003 [-0.0093, +0.0092] | -0.0155 [-0.0271, -0.0046]* | +0.0786 [+0.0647, +0.0931]* | -0.0097 [-0.0190, -0.0004]* | -0.0235 [-0.0321, -0.0148]* | +0.0097 [+0.0015, +0.0180]* |

**n_evidence**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=32 | 337 | 320 | -0.0748 [-0.0838, -0.0664]* | -0.0882 [-0.0973, -0.0790]* | -0.0674 [-0.0762, -0.0588]* | -0.1223 [-0.1335, -0.1109]* | -0.0664 [-0.0803, -0.0546]* | +0.0644 [+0.0453, +0.0831]* | -0.0204 [-0.0310, -0.0096]* | -0.0530 [-0.0632, -0.0432]* | -0.0900 [-0.1022, -0.0778]* |
| (32,41] | 349 | 348 | -0.0534 [-0.0608, -0.0465]* | -0.0626 [-0.0687, -0.0563]* | -0.0483 [-0.0554, -0.0414]* | -0.1239 [-0.1338, -0.1136]* | -0.0162 [-0.0261, -0.0066]* | +0.0468 [+0.0311, +0.0621]* | -0.0160 [-0.0252, -0.0069]* | -0.0275 [-0.0355, -0.0197]* | -0.0962 [-0.1069, -0.0855]* |
| (41,70] | 332 | 316 | -0.0465 [-0.0544, -0.0381]* | -0.0464 [-0.0538, -0.0393]* | -0.0446 [-0.0527, -0.0365]* | -0.0661 [-0.0755, -0.0568]* | -0.0220 [-0.0337, -0.0099]* | +0.0678 [+0.0527, +0.0845]* | -0.0195 [-0.0282, -0.0105]* | -0.0319 [-0.0399, -0.0236]* | -0.0492 [-0.0592, -0.0392]* |
| >70 | 327 | 243 | -0.0267 [-0.0353, -0.0184]* | +0.0061 [-0.0031, +0.0147] | -0.0281 [-0.0372, -0.0192]* | +0.0004 [-0.0091, +0.0096] | -0.0138 [-0.0252, -0.0031]* | +0.0802 [+0.0660, +0.0949]* | -0.0092 [-0.0180, -0.0002]* | -0.0231 [-0.0317, -0.0145]* | +0.0089 [+0.0004, +0.0172]* |

**n_edges**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=125 | 338 | 319 | -0.0760 [-0.0854, -0.0673]* | -0.0875 [-0.0963, -0.0782]* | -0.0683 [-0.0771, -0.0593]* | -0.1212 [-0.1325, -0.1097]* | -0.0679 [-0.0814, -0.0557]* | +0.0630 [+0.0433, +0.0820]* | -0.0218 [-0.0322, -0.0111]* | -0.0544 [-0.0647, -0.0441]* | -0.0897 [-0.1019, -0.0775]* |
| (125,157] | 338 | 335 | -0.0539 [-0.0612, -0.0467]* | -0.0620 [-0.0681, -0.0556]* | -0.0484 [-0.0556, -0.0414]* | -0.1223 [-0.1324, -0.1119]* | -0.0155 [-0.0256, -0.0054]* | +0.0442 [+0.0296, +0.0589]* | -0.0161 [-0.0249, -0.0066]* | -0.0291 [-0.0373, -0.0209]* | -0.0955 [-0.1058, -0.0846]* |
| (157,225] | 333 | 315 | -0.0432 [-0.0511, -0.0350]* | -0.0484 [-0.0563, -0.0409]* | -0.0428 [-0.0503, -0.0350]* | -0.0722 [-0.0824, -0.0626]* | -0.0192 [-0.0305, -0.0074]* | +0.0745 [+0.0584, +0.0924]* | -0.0163 [-0.0249, -0.0075]* | -0.0277 [-0.0361, -0.0192]* | -0.0536 [-0.0642, -0.0434]* |
| >225 | 336 | 249 | -0.0289 [-0.0374, -0.0204]* | +0.0050 [-0.0039, +0.0135] | -0.0293 [-0.0383, -0.0204]* | +0.0000 [-0.0092, +0.0090] | -0.0157 [-0.0273, -0.0049]* | +0.0766 [+0.0627, +0.0913]* | -0.0109 [-0.0198, -0.0018]* | -0.0242 [-0.0329, -0.0159]* | +0.0092 [+0.0011, +0.0173]* |

**density**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0127 | 337 | 251 | -0.0259 [-0.0343, -0.0182]* | +0.0030 [-0.0061, +0.0109] | -0.0269 [-0.0354, -0.0188]* | -0.0035 [-0.0130, +0.0051] | -0.0160 [-0.0271, -0.0056]* | +0.0782 [+0.0636, +0.0923]* | -0.0065 [-0.0156, +0.0018] | -0.0211 [-0.0296, -0.0132]* | +0.0060 [-0.0025, +0.0140] |
| (0.0127,0.016] | 336 | 320 | -0.0480 [-0.0559, -0.0403]* | -0.0472 [-0.0548, -0.0397]* | -0.0453 [-0.0531, -0.0374]* | -0.0707 [-0.0808, -0.0611]* | -0.0182 [-0.0291, -0.0069]* | +0.0631 [+0.0485, +0.0795]* | -0.0215 [-0.0296, -0.0128]* | -0.0322 [-0.0401, -0.0242]* | -0.0530 [-0.0634, -0.0430]* |
| (0.016,0.0191] | 337 | 334 | -0.0512 [-0.0591, -0.0440]* | -0.0607 [-0.0670, -0.0544]* | -0.0479 [-0.0553, -0.0409]* | -0.1212 [-0.1308, -0.1111]* | -0.0164 [-0.0271, -0.0053]* | +0.0477 [+0.0322, +0.0634]* | -0.0152 [-0.0242, -0.0058]* | -0.0259 [-0.0341, -0.0175]* | -0.0930 [-0.1035, -0.0821]* |
| >0.0191 | 335 | 312 | -0.0772 [-0.0859, -0.0689]* | -0.0887 [-0.0976, -0.0793]* | -0.0691 [-0.0777, -0.0605]* | -0.1210 [-0.1323, -0.1092]* | -0.0682 [-0.0815, -0.0558]* | +0.0691 [+0.0497, +0.0882]* | -0.0221 [-0.0327, -0.0114]* | -0.0565 [-0.0667, -0.0469]* | -0.0902 [-0.1029, -0.0774]* |

**num_visits**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 679 | 678 | -0.0563 [-0.0620, -0.0509]* | -0.0718 [-0.0773, -0.0663]* | -0.0520 [-0.0574, -0.0464]* | -0.1328 [-0.1402, -0.1250]* | -0.0289 [-0.0374, -0.0213]* | +0.0538 [+0.0425, +0.0660]* | -0.0161 [-0.0223, -0.0095]* | -0.0316 [-0.0376, -0.0257]* | -0.1038 [-0.1120, -0.0956]* |
| 2 | 323 | 323 | -0.0513 [-0.0594, -0.0428]* | -0.0436 [-0.0511, -0.0355]* | -0.0472 [-0.0550, -0.0392]* | -0.0400 [-0.0479, -0.0321]* | -0.0335 [-0.0447, -0.0222]* | +0.0671 [+0.0513, +0.0831]* | -0.0203 [-0.0292, -0.0112]* | -0.0377 [-0.0463, -0.0288]* | -0.0220 [-0.0305, -0.0136]* |
| 3 | 191 | 191 | -0.0497 [-0.0614, -0.0378]* | -0.0187 [-0.0301, -0.0074]* | -0.0463 [-0.0583, -0.0346]* | -0.0195 [-0.0303, -0.0088]* | -0.0364 [-0.0509, -0.0216]* | +0.0761 [+0.0571, +0.0948]* | -0.0212 [-0.0339, -0.0081]* | -0.0452 [-0.0573, -0.0332]* | -0.0095 [-0.0200, +0.0002] |
| 4 | 91 | 91 | -0.0307 [-0.0497, -0.0125]* | +0.0011 [-0.0172, +0.0191] | -0.0343 [-0.0539, -0.0152]* | -0.0057 [-0.0248, +0.0130] | -0.0144 [-0.0349, +0.0055] | +0.0922 [+0.0676, +0.1171]* | -0.0017 [-0.0205, +0.0167] | -0.0245 [-0.0432, -0.0059]* | +0.0102 [-0.0067, +0.0277] |
| 5 | 47 | 47 | -0.0197 [-0.0383, +0.0002] | +0.0207 [-0.0009, +0.0431] | -0.0194 [-0.0402, +0.0035] | +0.0158 [-0.0066, +0.0396] | -0.0258 [-0.0624, +0.0088] | +0.0920 [+0.0549, +0.1324]* | -0.0082 [-0.0303, +0.0164] | -0.0208 [-0.0407, +0.0005] | +0.0205 [-0.0027, +0.0432] |
| 6 | 14 | 14 | -0.0014 [-0.0473, +0.0473]+ | +0.0272 [-0.0091, +0.0682]+ | -0.0073 [-0.0571, +0.0446]+ | +0.0201 [-0.0198, +0.0607]+ | +0.0045 [-0.0451, +0.0575]+ | +0.0931 [+0.0553, +0.1342]*+ | +0.0104 [-0.0431, +0.0539]+ | -0.0078 [-0.0591, +0.0375]+ | +0.0150 [-0.0167, +0.0485]+ |

**n_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 172 | 155 | -0.0365 [-0.0488, -0.0241]* | -0.0196 [-0.0315, -0.0082]* | -0.0334 [-0.0459, -0.0206]* | -0.0362 [-0.0492, -0.0230]* | -0.0295 [-0.0423, -0.0169]* | +0.0504 [+0.0361, +0.0643]* | +0.0059 [-0.0078, +0.0210] | -0.0120 [-0.0259, +0.0019] | -0.0050 [-0.0194, +0.0087] |
| 1 | 701 | 588 | -0.0544 [-0.0606, -0.0484]* | -0.0519 [-0.0581, -0.0458]* | -0.0507 [-0.0564, -0.0448]* | -0.0894 [-0.0976, -0.0812]* | -0.0519 [-0.0595, -0.0437]* | +0.0411 [+0.0301, +0.0520]* | -0.0178 [-0.0241, -0.0111]* | -0.0364 [-0.0427, -0.0302]* | -0.0676 [-0.0756, -0.0595]* |
| 2 | 400 | 354 | -0.0506 [-0.0577, -0.0436]* | -0.0514 [-0.0589, -0.0438]* | -0.0488 [-0.0559, -0.0416]* | -0.0787 [-0.0885, -0.0689]* | +0.0029 [-0.0081, +0.0135] | +0.1000 [+0.0841, +0.1174]* | -0.0253 [-0.0331, -0.0175]* | -0.0396 [-0.0472, -0.0325]* | -0.0618 [-0.0717, -0.0522]* |
| 3 | 65 | 64 | -0.0426 [-0.0625, -0.0236]* | -0.0631 [-0.0861, -0.0426]* | -0.0343 [-0.0549, -0.0136]* | -0.0773 [-0.1030, -0.0514]* | +0.0075 [-0.0244, +0.0396] | +0.1276 [+0.0893, +0.1714]* | -0.0046 [-0.0258, +0.0160] | -0.0279 [-0.0474, -0.0093]* | -0.0586 [-0.0865, -0.0321]* |
| 4 | n<10 | |  |  |  |  |  |  |  |  |  |

**n_measurement**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=25 | 348 | 328 | -0.0762 [-0.0851, -0.0679]* | -0.0896 [-0.0983, -0.0808]* | -0.0692 [-0.0781, -0.0607]* | -0.1240 [-0.1349, -0.1126]* | -0.0653 [-0.0785, -0.0531]* | +0.0651 [+0.0457, +0.0833]* | -0.0219 [-0.0324, -0.0114]* | -0.0552 [-0.0652, -0.0454]* | -0.0924 [-0.1044, -0.0803]* |
| (25,33] | 336 | 335 | -0.0524 [-0.0598, -0.0451]* | -0.0611 [-0.0674, -0.0546]* | -0.0477 [-0.0550, -0.0406]* | -0.1218 [-0.1317, -0.1113]* | -0.0167 [-0.0269, -0.0063]* | +0.0462 [+0.0305, +0.0618]* | -0.0153 [-0.0244, -0.0055]* | -0.0261 [-0.0340, -0.0181]* | -0.0939 [-0.1047, -0.0829]* |
| (33,60] | 339 | 324 | -0.0476 [-0.0555, -0.0390]* | -0.0463 [-0.0537, -0.0393]* | -0.0454 [-0.0532, -0.0370]* | -0.0676 [-0.0770, -0.0585]* | -0.0240 [-0.0361, -0.0122]* | +0.0655 [+0.0503, +0.0822]* | -0.0194 [-0.0277, -0.0100]* | -0.0320 [-0.0402, -0.0237]* | -0.0492 [-0.0593, -0.0389]* |
| >60 | 322 | 237 | -0.0240 [-0.0325, -0.0157]* | +0.0076 [-0.0015, +0.0160] | -0.0250 [-0.0339, -0.0162]* | +0.0022 [-0.0073, +0.0109] | -0.0104 [-0.0208, -0.0001]* | +0.0819 [+0.0671, +0.0966]* | -0.0081 [-0.0173, +0.0008] | -0.0209 [-0.0299, -0.0123]* | +0.0095 [+0.0009, +0.0180]* |

**n_vital**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 16 | -0.0192 [-0.0757, +0.0397]+ | -0.0095 [-0.0459, +0.0278]+ | -0.0079 [-0.0620, +0.0480]+ | -0.0210 [-0.0489, +0.0086]+ | -0.0449 [-0.1038, +0.0198]+ | +0.0862 [+0.0019, +0.1836]*+ | -0.0134 [-0.0818, +0.0514]+ | -0.0018 [-0.0650, +0.0595]+ | -0.0013 [-0.0355, +0.0354]+ |
| 1 | n<10 | |  |  |  |  |  |  |  |  |  |
| 2 | n<10 | |  |  |  |  |  |  |  |  |  |
| 4 | n<10 | |  |  |  |  |  |  |  |  |  |
| 5 | 24 | 23 | -0.0333 [-0.0586, -0.0039]*+ | -0.0350 [-0.0692, +0.0030]+ | -0.0383 [-0.0659, -0.0083]*+ | -0.0548 [-0.0972, -0.0092]*+ | -0.0275 [-0.0511, -0.0038]*+ | +0.1142 [+0.0447, +0.1999]*+ | +0.0077 [-0.0218, +0.0377]+ | -0.0307 [-0.0611, +0.0023]+ | -0.0448 [-0.0893, +0.0025]+ |
| 6 | 1298 | 990 | -0.0516 [-0.0560, -0.0472]* | -0.0492 [-0.0539, -0.0446]* | -0.0482 [-0.0526, -0.0438]* | -0.0804 [-0.0863, -0.0744]* | -0.0298 [-0.0359, -0.0236]* | +0.0627 [+0.0545, +0.0713]* | -0.0172 [-0.0218, -0.0122]* | -0.0347 [-0.0393, -0.0301]* | -0.0586 [-0.0646, -0.0526]* |

**n_diagnosis**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0562 [-0.0618, -0.0507]* | -0.0717 [-0.0771, -0.0662]* | -0.0519 [-0.0574, -0.0463]* | -0.1325 [-0.1398, -0.1246]* | -0.0289 [-0.0374, -0.0213]* | +0.0538 [+0.0425, +0.0660]* | -0.0159 [-0.0221, -0.0093]* | -0.0314 [-0.0375, -0.0255]* | -0.1035 [-0.1117, -0.0952]* |
| (0,3] | 390 | 334 | -0.0467 [-0.0552, -0.0381]* | -0.0363 [-0.0440, -0.0281]* | -0.0423 [-0.0504, -0.0340]* | -0.0326 [-0.0404, -0.0247]* | -0.0279 [-0.0390, -0.0165]* | +0.0753 [+0.0597, +0.0911]* | -0.0163 [-0.0252, -0.0069]* | -0.0358 [-0.0448, -0.0269]* | -0.0177 [-0.0258, -0.0092]* |
| >3 | 275 | 204 | -0.0422 [-0.0519, -0.0324]* | -0.0075 [-0.0193, +0.0032] | -0.0427 [-0.0530, -0.0324]* | -0.0126 [-0.0239, -0.0018]* | -0.0340 [-0.0472, -0.0211]* | +0.0757 [+0.0591, +0.0925]* | -0.0175 [-0.0281, -0.0061]* | -0.0372 [-0.0468, -0.0277]* | -0.0001 [-0.0100, +0.0092] |

**share_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0137 | 340 | 277 | -0.0290 [-0.0375, -0.0202]* | -0.0037 [-0.0130, +0.0047] | -0.0284 [-0.0372, -0.0195]* | -0.0151 [-0.0252, -0.0054]* | -0.0273 [-0.0371, -0.0181]* | +0.0599 [+0.0487, +0.0716]* | +0.0015 [-0.0074, +0.0109] | -0.0146 [-0.0238, -0.0057]* | +0.0047 [-0.0049, +0.0137] |
| (0.0137,0.027] | 356 | 321 | -0.0466 [-0.0542, -0.0391]* | -0.0354 [-0.0422, -0.0285]* | -0.0425 [-0.0498, -0.0349]* | -0.0607 [-0.0700, -0.0512]* | -0.0331 [-0.0433, -0.0225]* | +0.0501 [+0.0355, +0.0654]* | -0.0207 [-0.0290, -0.0121]* | -0.0325 [-0.0405, -0.0244]* | -0.0442 [-0.0541, -0.0347]* |
| (0.027,0.0444] | 318 | 305 | -0.0574 [-0.0662, -0.0486]* | -0.0671 [-0.0749, -0.0596]* | -0.0539 [-0.0618, -0.0459]* | -0.1181 [-0.1304, -0.1061]* | -0.0233 [-0.0353, -0.0118]* | +0.0412 [+0.0268, +0.0577]* | -0.0183 [-0.0275, -0.0091]* | -0.0347 [-0.0439, -0.0255]* | -0.0944 [-0.1059, -0.0824]* |
| >0.0444 | 331 | 316 | -0.0703 [-0.0789, -0.0621]* | -0.0901 [-0.0985, -0.0818]* | -0.0653 [-0.0732, -0.0573]* | -0.1269 [-0.1375, -0.1169]* | -0.0344 [-0.0502, -0.0202]* | +0.1073 [+0.0864, +0.1279]* | -0.0279 [-0.0377, -0.0182]* | -0.0544 [-0.0634, -0.0460]* | -0.1002 [-0.1117, -0.0888]* |

**share_measurement**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.767 | 337 | 317 | -0.0795 [-0.0884, -0.0707]* | -0.0954 [-0.1044, -0.0867]* | -0.0724 [-0.0816, -0.0633]* | -0.1226 [-0.1340, -0.1112]* | -0.0679 [-0.0811, -0.0553]* | +0.0668 [+0.0480, +0.0853]* | -0.0236 [-0.0341, -0.0131]* | -0.0603 [-0.0702, -0.0509]* | -0.0920 [-0.1047, -0.0793]* |
| (0.767,0.814] | 339 | 330 | -0.0575 [-0.0644, -0.0503]* | -0.0625 [-0.0692, -0.0563]* | -0.0546 [-0.0614, -0.0478]* | -0.1237 [-0.1342, -0.1133]* | -0.0100 [-0.0218, +0.0016] | +0.0615 [+0.0456, +0.0784]* | -0.0207 [-0.0294, -0.0121]* | -0.0328 [-0.0407, -0.0252]* | -0.0977 [-0.1081, -0.0869]* |
| (0.814,0.86] | 333 | 313 | -0.0418 [-0.0495, -0.0343]* | -0.0395 [-0.0468, -0.0323]* | -0.0374 [-0.0450, -0.0303]* | -0.0662 [-0.0759, -0.0565]* | -0.0268 [-0.0375, -0.0156]* | +0.0540 [+0.0396, +0.0695]* | -0.0142 [-0.0230, -0.0053]* | -0.0237 [-0.0318, -0.0156]* | -0.0462 [-0.0573, -0.0360]* |
| >0.86 | 336 | 256 | -0.0232 [-0.0317, -0.0149]* | +0.0045 [-0.0039, +0.0129] | -0.0244 [-0.0330, -0.0156]* | -0.0030 [-0.0121, +0.0056] | -0.0138 [-0.0244, -0.0041]* | +0.0756 [+0.0607, +0.0908]* | -0.0067 [-0.0153, +0.0018] | -0.0186 [-0.0267, -0.0103]* | +0.0064 [-0.0018, +0.0147] |

**share_vital**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0845 | 341 | 253 | -0.0261 [-0.0350, -0.0177]* | +0.0056 [-0.0038, +0.0141] | -0.0270 [-0.0363, -0.0180]* | -0.0002 [-0.0092, +0.0087] | -0.0149 [-0.0263, -0.0043]* | +0.0818 [+0.0672, +0.0963]* | -0.0090 [-0.0180, +0.0002] | -0.0218 [-0.0306, -0.0131]* | +0.0089 [+0.0005, +0.0170]* |
| (0.0845,0.146] | 350 | 334 | -0.0471 [-0.0544, -0.0396]* | -0.0494 [-0.0565, -0.0424]* | -0.0451 [-0.0523, -0.0377]* | -0.0712 [-0.0807, -0.0624]* | -0.0212 [-0.0320, -0.0100]* | +0.0686 [+0.0539, +0.0849]* | -0.0208 [-0.0288, -0.0128]* | -0.0318 [-0.0394, -0.0241]* | -0.0535 [-0.0638, -0.0437]* |
| (0.146,0.182] | 324 | 323 | -0.0542 [-0.0615, -0.0472]* | -0.0621 [-0.0683, -0.0557]* | -0.0496 [-0.0564, -0.0425]* | -0.1255 [-0.1356, -0.1152]* | -0.0160 [-0.0266, -0.0057]* | +0.0423 [+0.0266, +0.0586]* | -0.0153 [-0.0247, -0.0057]* | -0.0288 [-0.0366, -0.0207]* | -0.0983 [-0.1090, -0.0873]* |
| >0.182 | 330 | 315 | -0.0759 [-0.0849, -0.0675]* | -0.0893 [-0.0984, -0.0801]* | -0.0682 [-0.0773, -0.0595]* | -0.1231 [-0.1344, -0.1117]* | -0.0673 [-0.0811, -0.0557]* | +0.0640 [+0.0455, +0.0818]* | -0.0201 [-0.0309, -0.0093]* | -0.0536 [-0.0638, -0.0435]* | -0.0903 [-0.1028, -0.0777]* |

**share_diagnosis**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0562 [-0.0618, -0.0507]* | -0.0717 [-0.0771, -0.0662]* | -0.0519 [-0.0574, -0.0463]* | -0.1325 [-0.1398, -0.1246]* | -0.0289 [-0.0374, -0.0213]* | +0.0538 [+0.0425, +0.0660]* | -0.0159 [-0.0221, -0.0093]* | -0.0314 [-0.0375, -0.0255]* | -0.1035 [-0.1117, -0.0952]* |
| (0,0.0441] | 329 | 259 | -0.0305 [-0.0385, -0.0225]* | -0.0152 [-0.0238, -0.0067]* | -0.0303 [-0.0391, -0.0220]* | -0.0176 [-0.0265, -0.0090]* | -0.0187 [-0.0305, -0.0068]* | +0.0728 [+0.0561, +0.0900]* | -0.0108 [-0.0195, -0.0023]* | -0.0241 [-0.0325, -0.0152]* | -0.0075 [-0.0159, +0.0009] |
| >0.0441 | 336 | 257 | -0.0588 [-0.0684, -0.0487]* | -0.0333 [-0.0431, -0.0237]* | -0.0544 [-0.0635, -0.0449]* | -0.0309 [-0.0406, -0.0219]* | -0.0419 [-0.0539, -0.0296]* | +0.0781 [+0.0634, +0.0930]* | -0.0225 [-0.0338, -0.0108]* | -0.0485 [-0.0583, -0.0384]* | -0.0132 [-0.0231, -0.0036]* |

**n_knowledge**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=21 | 347 | 331 | -0.0674 [-0.0757, -0.0595]* | -0.0802 [-0.0884, -0.0721]* | -0.0608 [-0.0688, -0.0527]* | -0.1229 [-0.1346, -0.1119]* | -0.0592 [-0.0717, -0.0477]* | +0.0660 [+0.0472, +0.0825]* | -0.0188 [-0.0283, -0.0086]* | -0.0454 [-0.0553, -0.0363]* | -0.0922 [-0.1045, -0.0800]* |
| (21,25] | 407 | 389 | -0.0519 [-0.0597, -0.0444]* | -0.0581 [-0.0653, -0.0514]* | -0.0455 [-0.0525, -0.0385]* | -0.1047 [-0.1145, -0.0950]* | -0.0137 [-0.0245, -0.0020]* | +0.0607 [+0.0462, +0.0760]* | -0.0185 [-0.0265, -0.0097]* | -0.0300 [-0.0386, -0.0218]* | -0.0818 [-0.0916, -0.0719]* |
| (25,30] | 290 | 265 | -0.0454 [-0.0535, -0.0370]* | -0.0440 [-0.0526, -0.0359]* | -0.0449 [-0.0534, -0.0363]* | -0.0627 [-0.0729, -0.0531]* | -0.0185 [-0.0305, -0.0068]* | +0.0607 [+0.0449, +0.0775]* | -0.0188 [-0.0280, -0.0093]* | -0.0335 [-0.0424, -0.0243]* | -0.0461 [-0.0567, -0.0368]* |
| >30 | 301 | 218 | -0.0343 [-0.0431, -0.0252]* | -0.0024 [-0.0126, +0.0070] | -0.0363 [-0.0458, -0.0271]* | -0.0095 [-0.0204, -0.0001]* | -0.0278 [-0.0398, -0.0167]* | +0.0718 [+0.0577, +0.0864]* | -0.0081 [-0.0178, +0.0014] | -0.0264 [-0.0351, -0.0179]* | +0.0045 [-0.0048, +0.0131] |

**n_analyte**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=23 | 367 | 328 | -0.0763 [-0.0849, -0.0679]* | -0.0826 [-0.0918, -0.0737]* | -0.0686 [-0.0775, -0.0601]* | -0.1049 [-0.1163, -0.0934]* | -0.0690 [-0.0814, -0.0567]* | +0.0711 [+0.0528, +0.0888]* | -0.0212 [-0.0312, -0.0114]* | -0.0572 [-0.0667, -0.0482]* | -0.0763 [-0.0886, -0.0641]* |
| (23,29] | 388 | 344 | -0.0435 [-0.0504, -0.0365]* | -0.0455 [-0.0518, -0.0386]* | -0.0450 [-0.0521, -0.0378]* | -0.0892 [-0.0994, -0.0783]* | -0.0161 [-0.0260, -0.0062]* | +0.0489 [+0.0340, +0.0642]* | -0.0127 [-0.0211, -0.0037]* | -0.0242 [-0.0319, -0.0164]* | -0.0657 [-0.0758, -0.0549]* |
| (29,33] | 294 | 274 | -0.0455 [-0.0541, -0.0367]* | -0.0361 [-0.0438, -0.0280]* | -0.0412 [-0.0507, -0.0326]* | -0.0656 [-0.0764, -0.0544]* | -0.0236 [-0.0359, -0.0122]* | +0.0582 [+0.0422, +0.0751]* | -0.0207 [-0.0301, -0.0113]* | -0.0293 [-0.0384, -0.0201]* | -0.0473 [-0.0582, -0.0358]* |
| >33 | 296 | 258 | -0.0330 [-0.0415, -0.0242]* | -0.0217 [-0.0317, -0.0116]* | -0.0298 [-0.0382, -0.0212]* | -0.0469 [-0.0596, -0.0345]* | -0.0046 [-0.0161, +0.0072] | +0.0830 [+0.0679, +0.0992]* | -0.0106 [-0.0198, -0.0007]* | -0.0223 [-0.0312, -0.0136]* | -0.0336 [-0.0465, -0.0215]* |

**n_prior_dx**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0562 [-0.0618, -0.0507]* | -0.0717 [-0.0771, -0.0662]* | -0.0519 [-0.0574, -0.0463]* | -0.1325 [-0.1398, -0.1246]* | -0.0289 [-0.0374, -0.0213]* | +0.0538 [+0.0425, +0.0660]* | -0.0159 [-0.0221, -0.0093]* | -0.0314 [-0.0375, -0.0255]* | -0.1035 [-0.1117, -0.0952]* |
| (0,3] | 390 | 334 | -0.0467 [-0.0552, -0.0381]* | -0.0363 [-0.0440, -0.0281]* | -0.0423 [-0.0504, -0.0340]* | -0.0326 [-0.0404, -0.0247]* | -0.0279 [-0.0390, -0.0165]* | +0.0753 [+0.0597, +0.0911]* | -0.0163 [-0.0252, -0.0069]* | -0.0358 [-0.0448, -0.0269]* | -0.0177 [-0.0258, -0.0092]* |
| >3 | 275 | 204 | -0.0422 [-0.0519, -0.0324]* | -0.0075 [-0.0193, +0.0032] | -0.0427 [-0.0530, -0.0324]* | -0.0126 [-0.0239, -0.0018]* | -0.0340 [-0.0472, -0.0211]* | +0.0757 [+0.0591, +0.0925]* | -0.0175 [-0.0281, -0.0061]* | -0.0372 [-0.0468, -0.0277]* | -0.0001 [-0.0100, +0.0092] |

**rel_trajectory_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 1039 | 907 | -0.0567 [-0.0614, -0.0522]* | -0.0631 [-0.0675, -0.0588]* | -0.0522 [-0.0565, -0.0476]* | -0.1014 [-0.1076, -0.0953]* | -0.0327 [-0.0394, -0.0260]* | +0.0601 [+0.0510, +0.0699]* | -0.0185 [-0.0235, -0.0132]* | -0.0359 [-0.0408, -0.0311]* | -0.0762 [-0.0827, -0.0696]* |
| >0 | 306 | 229 | -0.0297 [-0.0398, -0.0194]* | +0.0021 [-0.0083, +0.0118] | -0.0305 [-0.0409, -0.0204]* | -0.0031 [-0.0136, +0.0069] | -0.0193 [-0.0325, -0.0066]* | +0.0797 [+0.0633, +0.0949]* | -0.0087 [-0.0192, +0.0029] | -0.0269 [-0.0373, -0.0163]* | +0.0059 [-0.0037, +0.0153] |

**rel_recurrence_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0562 [-0.0618, -0.0507]* | -0.0717 [-0.0771, -0.0662]* | -0.0519 [-0.0574, -0.0463]* | -0.1325 [-0.1398, -0.1246]* | -0.0289 [-0.0374, -0.0213]* | +0.0538 [+0.0425, +0.0660]* | -0.0159 [-0.0221, -0.0093]* | -0.0314 [-0.0375, -0.0255]* | -0.1035 [-0.1117, -0.0952]* |
| (0,3] | 390 | 334 | -0.0467 [-0.0552, -0.0381]* | -0.0363 [-0.0440, -0.0281]* | -0.0423 [-0.0504, -0.0340]* | -0.0326 [-0.0404, -0.0247]* | -0.0279 [-0.0390, -0.0165]* | +0.0753 [+0.0597, +0.0911]* | -0.0163 [-0.0252, -0.0069]* | -0.0358 [-0.0448, -0.0269]* | -0.0177 [-0.0258, -0.0092]* |
| >3 | 275 | 204 | -0.0422 [-0.0519, -0.0324]* | -0.0075 [-0.0193, +0.0032] | -0.0427 [-0.0530, -0.0324]* | -0.0126 [-0.0239, -0.0018]* | -0.0340 [-0.0472, -0.0211]* | +0.0757 [+0.0591, +0.0925]* | -0.0175 [-0.0281, -0.0061]* | -0.0372 [-0.0468, -0.0277]* | -0.0001 [-0.0100, +0.0092] |

**rel_comorbid_with**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 862 | 783 | -0.0538 [-0.0589, -0.0490]* | -0.0643 [-0.0693, -0.0593]* | -0.0495 [-0.0544, -0.0444]* | -0.1114 [-0.1185, -0.1042]* | -0.0278 [-0.0354, -0.0205]* | +0.0599 [+0.0498, +0.0713]* | -0.0151 [-0.0208, -0.0091]* | -0.0325 [-0.0377, -0.0272]* | -0.0858 [-0.0929, -0.0781]* |
| (0,2] | 159 | 144 | -0.0572 [-0.0713, -0.0437]* | -0.0395 [-0.0525, -0.0271]* | -0.0533 [-0.0659, -0.0404]* | -0.0359 [-0.0481, -0.0240]* | -0.0449 [-0.0611, -0.0277]* | +0.0751 [+0.0532, +0.0980]* | -0.0267 [-0.0398, -0.0132]* | -0.0441 [-0.0580, -0.0303]* | -0.0174 [-0.0285, -0.0064]* |
| >2 | 324 | 229 | -0.0386 [-0.0473, -0.0296]* | -0.0101 [-0.0207, -0.0007]* | -0.0384 [-0.0473, -0.0290]* | -0.0141 [-0.0244, -0.0043]* | -0.0269 [-0.0394, -0.0146]* | +0.0716 [+0.0565, +0.0872]* | -0.0145 [-0.0244, -0.0037]* | -0.0327 [-0.0418, -0.0231]* | -0.0019 [-0.0116, +0.0071] |

**rel_co_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 873 | 697 | -0.0509 [-0.0563, -0.0455]* | -0.0455 [-0.0510, -0.0400]* | -0.0473 [-0.0527, -0.0418]* | -0.0790 [-0.0863, -0.0718]* | -0.0475 [-0.0546, -0.0407]* | +0.0429 [+0.0343, +0.0524]* | -0.0131 [-0.0187, -0.0072]* | -0.0316 [-0.0375, -0.0258]* | -0.0552 [-0.0623, -0.0477]* |
| 2 | 400 | 354 | -0.0506 [-0.0577, -0.0436]* | -0.0514 [-0.0589, -0.0438]* | -0.0488 [-0.0559, -0.0416]* | -0.0787 [-0.0885, -0.0689]* | +0.0029 [-0.0081, +0.0135] | +0.1000 [+0.0841, +0.1174]* | -0.0253 [-0.0331, -0.0175]* | -0.0396 [-0.0472, -0.0325]* | -0.0618 [-0.0717, -0.0522]* |
| 6 | 65 | 64 | -0.0426 [-0.0625, -0.0236]* | -0.0631 [-0.0861, -0.0426]* | -0.0343 [-0.0549, -0.0136]* | -0.0773 [-0.1030, -0.0514]* | +0.0075 [-0.0244, +0.0396] | +0.1276 [+0.0893, +0.1714]* | -0.0046 [-0.0258, +0.0160] | -0.0279 [-0.0474, -0.0093]* | -0.0586 [-0.0865, -0.0321]* |
| 12 | n<10 | |  |  |  |  |  |  |  |  |  |

**rel_baseline_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 768 | 755 | -0.0592 [-0.0645, -0.0540]* | -0.0731 [-0.0782, -0.0679]* | -0.0542 [-0.0593, -0.0490]* | -0.1248 [-0.1317, -0.1177]* | -0.0303 [-0.0384, -0.0232]* | +0.0565 [+0.0456, +0.0684]* | -0.0172 [-0.0229, -0.0112]* | -0.0347 [-0.0405, -0.0293]* | -0.0956 [-0.1031, -0.0879]* |
| (0,25] | 287 | 242 | -0.0522 [-0.0624, -0.0424]* | -0.0355 [-0.0448, -0.0265]* | -0.0487 [-0.0587, -0.0393]* | -0.0359 [-0.0451, -0.0275]* | -0.0421 [-0.0552, -0.0280]* | +0.0742 [+0.0553, +0.0933]* | -0.0200 [-0.0313, -0.0080]* | -0.0432 [-0.0537, -0.0329]* | -0.0218 [-0.0314, -0.0125]* |
| >25 | 290 | 233 | -0.0260 [-0.0351, -0.0172]* | +0.0046 [-0.0051, +0.0133] | -0.0273 [-0.0366, -0.0181]* | -0.0004 [-0.0104, +0.0089] | -0.0155 [-0.0277, -0.0038]* | +0.0761 [+0.0621, +0.0908]* | -0.0102 [-0.0198, -0.0009]* | -0.0225 [-0.0321, -0.0139]* | +0.0081 [-0.0011, +0.0172] |

**rel_medical_total**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=63 | 353 | 327 | -0.0745 [-0.0832, -0.0660]* | -0.0844 [-0.0932, -0.0757]* | -0.0663 [-0.0751, -0.0573]* | -0.1129 [-0.1236, -0.1018]* | -0.0747 [-0.0879, -0.0625]* | +0.0672 [+0.0482, +0.0849]* | -0.0214 [-0.0321, -0.0108]* | -0.0551 [-0.0649, -0.0454]* | -0.0840 [-0.0958, -0.0719]* |
| (63,75] | 352 | 327 | -0.0476 [-0.0551, -0.0403]* | -0.0514 [-0.0585, -0.0444]* | -0.0460 [-0.0530, -0.0390]* | -0.1001 [-0.1113, -0.0892]* | -0.0120 [-0.0219, -0.0023]* | +0.0519 [+0.0364, +0.0683]* | -0.0116 [-0.0197, -0.0030]* | -0.0257 [-0.0338, -0.0176]* | -0.0756 [-0.0869, -0.0646]* |
| (75,87] | 320 | 291 | -0.0454 [-0.0536, -0.0374]* | -0.0426 [-0.0507, -0.0347]* | -0.0445 [-0.0532, -0.0362]* | -0.0755 [-0.0866, -0.0655]* | -0.0194 [-0.0320, -0.0072]* | +0.0577 [+0.0420, +0.0742]* | -0.0237 [-0.0319, -0.0155]* | -0.0311 [-0.0398, -0.0224]* | -0.0572 [-0.0678, -0.0471]* |
| >87 | 320 | 251 | -0.0326 [-0.0411, -0.0244]* | -0.0107 [-0.0205, -0.0012]* | -0.0304 [-0.0387, -0.0219]* | -0.0221 [-0.0330, -0.0117]* | -0.0095 [-0.0207, +0.0014] | +0.0822 [+0.0686, +0.0964]* | -0.0085 [-0.0181, +0.0008] | -0.0223 [-0.0305, -0.0142]* | -0.0087 [-0.0194, +0.0009] |

**rel_medical_member_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=21 | 354 | 317 | -0.0766 [-0.0854, -0.0683]* | -0.0819 [-0.0915, -0.0724]* | -0.0684 [-0.0773, -0.0597]* | -0.1025 [-0.1139, -0.0909]* | -0.0710 [-0.0841, -0.0580]* | +0.0711 [+0.0526, +0.0897]* | -0.0207 [-0.0309, -0.0107]* | -0.0574 [-0.0671, -0.0481]* | -0.0735 [-0.0858, -0.0613]* |
| (21,28] | 483 | 424 | -0.0455 [-0.0520, -0.0390]* | -0.0446 [-0.0504, -0.0387]* | -0.0466 [-0.0533, -0.0398]* | -0.0853 [-0.0939, -0.0762]* | -0.0211 [-0.0304, -0.0124]* | +0.0463 [+0.0329, +0.0597]* | -0.0148 [-0.0223, -0.0070]* | -0.0266 [-0.0335, -0.0195]* | -0.0615 [-0.0703, -0.0522]* |
| (28,31] | 188 | 180 | -0.0413 [-0.0522, -0.0313]* | -0.0341 [-0.0440, -0.0241]* | -0.0350 [-0.0458, -0.0254]* | -0.0681 [-0.0823, -0.0546]* | -0.0175 [-0.0313, -0.0045]* | +0.0632 [+0.0439, +0.0824]* | -0.0182 [-0.0302, -0.0073]* | -0.0244 [-0.0366, -0.0134]* | -0.0508 [-0.0651, -0.0372]* |
| >31 | 320 | 280 | -0.0348 [-0.0436, -0.0259]* | -0.0252 [-0.0356, -0.0152]* | -0.0320 [-0.0406, -0.0236]* | -0.0501 [-0.0623, -0.0378]* | -0.0039 [-0.0151, +0.0076] | +0.0855 [+0.0699, +0.1016]* | -0.0125 [-0.0221, -0.0034]* | -0.0244 [-0.0331, -0.0157]* | -0.0377 [-0.0500, -0.0257]* |

**rel_medical_assesses**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=12 | 457 | 400 | -0.0633 [-0.0716, -0.0552]* | -0.0709 [-0.0788, -0.0633]* | -0.0603 [-0.0684, -0.0523]* | -0.0998 [-0.1097, -0.0899]* | -0.0609 [-0.0718, -0.0495]* | +0.0651 [+0.0496, +0.0816]* | -0.0197 [-0.0289, -0.0105]* | -0.0461 [-0.0555, -0.0370]* | -0.0726 [-0.0830, -0.0622]* |
| (12,15] | 459 | 401 | -0.0513 [-0.0573, -0.0450]* | -0.0463 [-0.0523, -0.0400]* | -0.0473 [-0.0532, -0.0412]* | -0.0827 [-0.0919, -0.0733]* | -0.0174 [-0.0269, -0.0085]* | +0.0603 [+0.0473, +0.0736]* | -0.0192 [-0.0261, -0.0118]* | -0.0312 [-0.0381, -0.0242]* | -0.0620 [-0.0709, -0.0530]* |
| (15,16] | 142 | 133 | -0.0445 [-0.0564, -0.0328]* | -0.0341 [-0.0466, -0.0215]* | -0.0424 [-0.0555, -0.0293]* | -0.0638 [-0.0806, -0.0472]* | -0.0206 [-0.0392, -0.0034]* | +0.0473 [+0.0254, +0.0684]* | -0.0169 [-0.0298, -0.0038]* | -0.0279 [-0.0407, -0.0151]* | -0.0405 [-0.0574, -0.0240]* |
| >16 | 287 | 255 | -0.0320 [-0.0407, -0.0235]* | -0.0226 [-0.0330, -0.0124]* | -0.0287 [-0.0373, -0.0203]* | -0.0477 [-0.0609, -0.0352]* | -0.0038 [-0.0155, +0.0082] | +0.0788 [+0.0626, +0.0955]* | -0.0060 [-0.0161, +0.0038] | -0.0217 [-0.0302, -0.0132]* | -0.0347 [-0.0477, -0.0227]* |

**rel_medical_measures**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=22 | 382 | 335 | -0.0731 [-0.0818, -0.0648]* | -0.0771 [-0.0860, -0.0681]* | -0.0648 [-0.0734, -0.0563]* | -0.0973 [-0.1081, -0.0866]* | -0.0657 [-0.0787, -0.0526]* | +0.0717 [+0.0547, +0.0887]* | -0.0179 [-0.0280, -0.0076]* | -0.0549 [-0.0643, -0.0458]* | -0.0700 [-0.0816, -0.0586]* |
| (22,26] | 525 | 457 | -0.0479 [-0.0541, -0.0416]* | -0.0458 [-0.0517, -0.0401]* | -0.0480 [-0.0544, -0.0413]* | -0.0866 [-0.0950, -0.0782]* | -0.0217 [-0.0310, -0.0125]* | +0.0498 [+0.0364, +0.0632]* | -0.0195 [-0.0266, -0.0120]* | -0.0298 [-0.0369, -0.0227]* | -0.0649 [-0.0734, -0.0559]* |
| (26,28] | 152 | 146 | -0.0337 [-0.0445, -0.0228]* | -0.0253 [-0.0373, -0.0129]* | -0.0297 [-0.0403, -0.0193]* | -0.0550 [-0.0714, -0.0386]* | -0.0032 [-0.0206, +0.0138] | +0.0760 [+0.0538, +0.0984]* | -0.0064 [-0.0182, +0.0057] | -0.0237 [-0.0354, -0.0127]* | -0.0432 [-0.0593, -0.0274]* |
| >28 | 286 | 246 | -0.0343 [-0.0430, -0.0256]* | -0.0267 [-0.0364, -0.0168]* | -0.0318 [-0.0403, -0.0233]* | -0.0535 [-0.0666, -0.0412]* | -0.0101 [-0.0195, -0.0004]* | +0.0759 [+0.0604, +0.0924]* | -0.0135 [-0.0231, -0.0043]* | -0.0187 [-0.0275, -0.0098]* | -0.0349 [-0.0479, -0.0231]* |

**rel_medical_is_a**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=1 | 506 | 506 | -0.0570 [-0.0638, -0.0505]* | -0.0694 [-0.0757, -0.0635]* | -0.0518 [-0.0584, -0.0452]* | -0.1316 [-0.1402, -0.1226]* | -0.0448 [-0.0539, -0.0365]* | +0.0329 [+0.0213, +0.0448]* | -0.0156 [-0.0228, -0.0080]* | -0.0317 [-0.0392, -0.0247]* | -0.1023 [-0.1119, -0.0927]* |
| (1,3] | 189 | 188 | -0.0531 [-0.0633, -0.0426]* | -0.0740 [-0.0853, -0.0623]* | -0.0521 [-0.0628, -0.0414]* | -0.1269 [-0.1408, -0.1128]* | +0.0130 [-0.0036, +0.0294] | +0.1133 [+0.0897, +0.1408]* | -0.0150 [-0.0262, -0.0035]* | -0.0310 [-0.0416, -0.0202]* | -0.0987 [-0.1127, -0.0845]* |
| (3,10] | 349 | 307 | -0.0486 [-0.0576, -0.0399]* | -0.0395 [-0.0475, -0.0315]* | -0.0441 [-0.0526, -0.0351]* | -0.0353 [-0.0433, -0.0274]* | -0.0325 [-0.0437, -0.0204]* | +0.0734 [+0.0572, +0.0903]* | -0.0166 [-0.0262, -0.0067]* | -0.0363 [-0.0459, -0.0271]* | -0.0194 [-0.0277, -0.0110]* |
| >10 | 301 | 223 | -0.0403 [-0.0494, -0.0312]* | -0.0069 [-0.0178, +0.0031] | -0.0402 [-0.0496, -0.0304]* | -0.0112 [-0.0218, -0.0009]* | -0.0275 [-0.0396, -0.0156]* | +0.0768 [+0.0613, +0.0932]* | -0.0179 [-0.0279, -0.0073]* | -0.0366 [-0.0454, -0.0273]* | -0.0005 [-0.0102, +0.0083] |

**rel_medical_may_affect**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 623 | 519 | -0.0503 [-0.0567, -0.0438]* | -0.0419 [-0.0486, -0.0354]* | -0.0455 [-0.0520, -0.0389]* | -0.0678 [-0.0759, -0.0597]* | -0.0427 [-0.0527, -0.0335]* | +0.0746 [+0.0623, +0.0862]* | -0.0171 [-0.0238, -0.0098]* | -0.0359 [-0.0432, -0.0288]* | -0.0472 [-0.0558, -0.0388]* |
| 1 | 479 | 415 | -0.0485 [-0.0556, -0.0415]* | -0.0566 [-0.0642, -0.0491]* | -0.0475 [-0.0545, -0.0404]* | -0.0916 [-0.1017, -0.0815]* | -0.0288 [-0.0375, -0.0193]* | +0.0661 [+0.0528, +0.0804]* | -0.0142 [-0.0223, -0.0062]* | -0.0327 [-0.0400, -0.0253]* | -0.0727 [-0.0825, -0.0632]* |
| 2 | 172 | 162 | -0.0537 [-0.0633, -0.0441]* | -0.0482 [-0.0593, -0.0369]* | -0.0488 [-0.0592, -0.0384]* | -0.0809 [-0.0963, -0.0659]* | +0.0051 [-0.0104, +0.0205] | +0.0408 [+0.0217, +0.0598]* | -0.0225 [-0.0328, -0.0119]* | -0.0333 [-0.0422, -0.0242]* | -0.0575 [-0.0724, -0.0427]* |
| 3 | 54 | 54 | -0.0567 [-0.0761, -0.0376]* | -0.0513 [-0.0716, -0.0308]* | -0.0569 [-0.0771, -0.0373]* | -0.0944 [-0.1202, -0.0670]* | -0.0002 [-0.0329, +0.0292] | +0.0179 [-0.0090, +0.0471] | -0.0098 [-0.0288, +0.0093] | -0.0227 [-0.0425, -0.0040]* | -0.0522 [-0.0754, -0.0296]* |
| 4 | 11 | 11 | -0.0629 [-0.1019, -0.0261]*+ | -0.0596 [-0.0939, -0.0325]*+ | -0.0523 [-0.0883, -0.0179]*+ | -0.0975 [-0.1498, -0.0532]*+ | -0.0311 [-0.1047, +0.0374]+ | -0.0023 [-0.0437, +0.0384]+ | +0.0025 [-0.0429, +0.0479]+ | -0.0299 [-0.0712, +0.0112]+ | -0.0563 [-0.1011, -0.0142]*+ |
| 5 | n<10 | |  |  |  |  |  |  |  |  |  |

**rel_medical_indicates**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 494 | 440 | -0.0580 [-0.0649, -0.0514]* | -0.0542 [-0.0622, -0.0468]* | -0.0512 [-0.0583, -0.0440]* | -0.0938 [-0.1043, -0.0838]* | -0.0444 [-0.0552, -0.0337]* | +0.0604 [+0.0461, +0.0741]* | -0.0215 [-0.0294, -0.0131]* | -0.0403 [-0.0478, -0.0330]* | -0.0704 [-0.0812, -0.0600]* |
| (0,1] | 272 | 254 | -0.0481 [-0.0579, -0.0383]* | -0.0517 [-0.0613, -0.0424]* | -0.0447 [-0.0536, -0.0361]* | -0.0847 [-0.0965, -0.0724]* | -0.0241 [-0.0361, -0.0120]* | +0.0548 [+0.0381, +0.0734]* | -0.0186 [-0.0291, -0.0079]* | -0.0298 [-0.0402, -0.0195]* | -0.0666 [-0.0791, -0.0540]* |
| (1,3] | 267 | 240 | -0.0479 [-0.0569, -0.0384]* | -0.0482 [-0.0578, -0.0393]* | -0.0457 [-0.0551, -0.0366]* | -0.0775 [-0.0889, -0.0661]* | -0.0112 [-0.0238, +0.0026] | +0.0684 [+0.0500, +0.0879]* | -0.0212 [-0.0308, -0.0117]* | -0.0326 [-0.0425, -0.0234]* | -0.0566 [-0.0676, -0.0456]* |
| >3 | 312 | 243 | -0.0433 [-0.0518, -0.0346]* | -0.0361 [-0.0461, -0.0270]* | -0.0445 [-0.0536, -0.0352]* | -0.0521 [-0.0632, -0.0413]* | -0.0269 [-0.0377, -0.0154]* | +0.0763 [+0.0624, +0.0907]* | -0.0019 [-0.0111, +0.0081] | -0.0283 [-0.0373, -0.0196]* | -0.0299 [-0.0413, -0.0190]* |

### 6.2 cei_final, confidence-adjusted

**n_nodes**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=81 | 339 | 317 | -0.0589 [-0.0662, -0.0514]* | -0.0560 [-0.0629, -0.0488]* | -0.0505 [-0.0575, -0.0427]* | -0.0803 [-0.0900, -0.0704]* | -0.0509 [-0.0620, -0.0400]* | +0.0547 [+0.0408, +0.0690]* | -0.0123 [-0.0213, -0.0034]* | -0.0384 [-0.0465, -0.0299]* | -0.0520 [-0.0621, -0.0421]* |
| (81,99] | 338 | 336 | -0.0505 [-0.0576, -0.0433]* | -0.0530 [-0.0589, -0.0472]* | -0.0461 [-0.0528, -0.0392]* | -0.1120 [-0.1202, -0.1034]* | -0.0115 [-0.0220, -0.0017]* | +0.0477 [+0.0343, +0.0621]* | -0.0147 [-0.0233, -0.0056]* | -0.0255 [-0.0335, -0.0177]* | -0.0851 [-0.0940, -0.0757]* |
| (99,134] | 345 | 327 | -0.0480 [-0.0548, -0.0411]* | -0.0512 [-0.0581, -0.0445]* | -0.0464 [-0.0531, -0.0396]* | -0.0737 [-0.0825, -0.0654]* | -0.0239 [-0.0344, -0.0126]* | +0.0737 [+0.0582, +0.0900]* | -0.0186 [-0.0257, -0.0109]* | -0.0323 [-0.0391, -0.0250]* | -0.0555 [-0.0645, -0.0467]* |
| >134 | 323 | 240 | -0.0446 [-0.0524, -0.0372]* | -0.0323 [-0.0398, -0.0257]* | -0.0459 [-0.0542, -0.0381]* | -0.0489 [-0.0572, -0.0413]* | -0.0324 [-0.0435, -0.0222]* | +0.0826 [+0.0680, +0.0971]* | -0.0198 [-0.0280, -0.0114]* | -0.0396 [-0.0475, -0.0318]* | -0.0366 [-0.0447, -0.0294]* |

**n_evidence**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=32 | 337 | 320 | -0.0573 [-0.0648, -0.0500]* | -0.0570 [-0.0642, -0.0496]* | -0.0499 [-0.0571, -0.0423]* | -0.0824 [-0.0919, -0.0725]* | -0.0475 [-0.0594, -0.0368]* | +0.0551 [+0.0411, +0.0693]* | -0.0102 [-0.0193, -0.0012]* | -0.0363 [-0.0444, -0.0275]* | -0.0530 [-0.0633, -0.0429]* |
| (32,41] | 349 | 348 | -0.0519 [-0.0588, -0.0451]* | -0.0539 [-0.0598, -0.0478]* | -0.0467 [-0.0533, -0.0403]* | -0.1131 [-0.1215, -0.1040]* | -0.0137 [-0.0237, -0.0042]* | +0.0478 [+0.0343, +0.0624]* | -0.0160 [-0.0246, -0.0077]* | -0.0263 [-0.0341, -0.0190]* | -0.0862 [-0.0953, -0.0767]* |
| (41,70] | 332 | 316 | -0.0496 [-0.0566, -0.0418]* | -0.0498 [-0.0564, -0.0432]* | -0.0478 [-0.0547, -0.0404]* | -0.0707 [-0.0791, -0.0624]* | -0.0271 [-0.0379, -0.0158]* | +0.0731 [+0.0579, +0.0891]* | -0.0200 [-0.0279, -0.0117]* | -0.0342 [-0.0414, -0.0268]* | -0.0528 [-0.0614, -0.0442]* |
| >70 | 327 | 243 | -0.0432 [-0.0510, -0.0361]* | -0.0320 [-0.0394, -0.0254]* | -0.0446 [-0.0527, -0.0371]* | -0.0477 [-0.0561, -0.0402]* | -0.0308 [-0.0420, -0.0209]* | +0.0833 [+0.0685, +0.0981]* | -0.0193 [-0.0274, -0.0111]* | -0.0392 [-0.0470, -0.0317]* | -0.0362 [-0.0443, -0.0287]* |

**n_edges**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=125 | 338 | 319 | -0.0583 [-0.0658, -0.0510]* | -0.0566 [-0.0637, -0.0491]* | -0.0506 [-0.0578, -0.0428]* | -0.0814 [-0.0910, -0.0716]* | -0.0488 [-0.0609, -0.0377]* | +0.0533 [+0.0392, +0.0678]* | -0.0114 [-0.0208, -0.0024]* | -0.0374 [-0.0458, -0.0287]* | -0.0527 [-0.0630, -0.0427]* |
| (125,157] | 338 | 335 | -0.0521 [-0.0589, -0.0452]* | -0.0527 [-0.0587, -0.0467]* | -0.0466 [-0.0531, -0.0400]* | -0.1110 [-0.1196, -0.1018]* | -0.0123 [-0.0226, -0.0027]* | +0.0458 [+0.0325, +0.0598]* | -0.0162 [-0.0243, -0.0074]* | -0.0278 [-0.0356, -0.0201]* | -0.0851 [-0.0943, -0.0757]* |
| (157,225] | 333 | 315 | -0.0468 [-0.0540, -0.0395]* | -0.0504 [-0.0576, -0.0436]* | -0.0465 [-0.0535, -0.0392]* | -0.0747 [-0.0837, -0.0662]* | -0.0250 [-0.0359, -0.0135]* | +0.0788 [+0.0630, +0.0957]* | -0.0171 [-0.0247, -0.0088]* | -0.0305 [-0.0381, -0.0227]* | -0.0553 [-0.0643, -0.0467]* |
| >225 | 336 | 249 | -0.0449 [-0.0526, -0.0377]* | -0.0335 [-0.0407, -0.0270]* | -0.0454 [-0.0537, -0.0379]* | -0.0489 [-0.0571, -0.0416]* | -0.0323 [-0.0434, -0.0225]* | +0.0805 [+0.0662, +0.0951]* | -0.0205 [-0.0287, -0.0125]* | -0.0399 [-0.0478, -0.0321]* | -0.0367 [-0.0445, -0.0295]* |

**density**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0127 | 337 | 251 | -0.0426 [-0.0501, -0.0358]* | -0.0329 [-0.0401, -0.0264]* | -0.0437 [-0.0512, -0.0368]* | -0.0494 [-0.0575, -0.0420]* | -0.0335 [-0.0443, -0.0237]* | +0.0830 [+0.0683, +0.0979]* | -0.0165 [-0.0245, -0.0084]* | -0.0374 [-0.0450, -0.0301]* | -0.0371 [-0.0448, -0.0299]* |
| (0.0127,0.016] | 336 | 320 | -0.0519 [-0.0592, -0.0447]* | -0.0506 [-0.0580, -0.0438]* | -0.0492 [-0.0566, -0.0416]* | -0.0756 [-0.0845, -0.0668]* | -0.0235 [-0.0341, -0.0127]* | +0.0690 [+0.0540, +0.0843]* | -0.0230 [-0.0302, -0.0152]* | -0.0356 [-0.0428, -0.0282]* | -0.0572 [-0.0662, -0.0481]* |
| (0.016,0.0191] | 337 | 334 | -0.0488 [-0.0562, -0.0415]* | -0.0545 [-0.0606, -0.0484]* | -0.0454 [-0.0523, -0.0386]* | -0.1131 [-0.1212, -0.1045]* | -0.0133 [-0.0240, -0.0030]* | +0.0483 [+0.0348, +0.0622]* | -0.0143 [-0.0229, -0.0058]* | -0.0237 [-0.0317, -0.0159]* | -0.0854 [-0.0943, -0.0764]* |
| >0.0191 | 335 | 312 | -0.0590 [-0.0661, -0.0518]* | -0.0553 [-0.0621, -0.0483]* | -0.0507 [-0.0579, -0.0432]* | -0.0780 [-0.0877, -0.0684]* | -0.0483 [-0.0599, -0.0372]* | +0.0578 [+0.0437, +0.0723]* | -0.0114 [-0.0208, -0.0024]* | -0.0389 [-0.0470, -0.0309]* | -0.0502 [-0.0603, -0.0403]* |

**num_visits**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 679 | 678 | -0.0514 [-0.0565, -0.0464]* | -0.0570 [-0.0618, -0.0519]* | -0.0471 [-0.0521, -0.0420]* | -0.1142 [-0.1209, -0.1067]* | -0.0224 [-0.0305, -0.0152]* | +0.0507 [+0.0410, +0.0609]* | -0.0138 [-0.0198, -0.0075]* | -0.0271 [-0.0329, -0.0214]* | -0.0868 [-0.0937, -0.0789]* |
| 2 | 323 | 323 | -0.0515 [-0.0586, -0.0444]* | -0.0459 [-0.0523, -0.0396]* | -0.0474 [-0.0544, -0.0400]* | -0.0420 [-0.0498, -0.0347]* | -0.0369 [-0.0479, -0.0260]* | +0.0707 [+0.0556, +0.0860]* | -0.0192 [-0.0272, -0.0101]* | -0.0373 [-0.0448, -0.0294]* | -0.0231 [-0.0317, -0.0154]* |
| 3 | 191 | 191 | -0.0538 [-0.0636, -0.0442]* | -0.0356 [-0.0458, -0.0263]* | -0.0505 [-0.0607, -0.0406]* | -0.0404 [-0.0501, -0.0316]* | -0.0420 [-0.0554, -0.0282]* | +0.0796 [+0.0608, +0.0980]* | -0.0229 [-0.0340, -0.0106]* | -0.0489 [-0.0589, -0.0385]* | -0.0286 [-0.0380, -0.0197]* |
| 4 | 91 | 91 | -0.0439 [-0.0595, -0.0286]* | -0.0335 [-0.0471, -0.0206]* | -0.0477 [-0.0644, -0.0317]* | -0.0520 [-0.0664, -0.0385]* | -0.0267 [-0.0451, -0.0097]* | +0.0989 [+0.0754, +0.1230]* | -0.0099 [-0.0259, +0.0068] | -0.0378 [-0.0537, -0.0220]* | -0.0338 [-0.0471, -0.0212]* |
| 5 | 47 | 47 | -0.0369 [-0.0508, -0.0222]* | -0.0270 [-0.0421, -0.0125]* | -0.0368 [-0.0524, -0.0192]* | -0.0438 [-0.0598, -0.0280]* | -0.0432 [-0.0768, -0.0119]* | +0.0867 [+0.0491, +0.1266]* | -0.0196 [-0.0385, +0.0002] | -0.0383 [-0.0563, -0.0213]* | -0.0362 [-0.0537, -0.0190]* |
| 6 | 14 | 14 | -0.0306 [-0.0666, +0.0076]+ | -0.0245 [-0.0555, +0.0066]+ | -0.0364 [-0.0755, +0.0066]+ | -0.0472 [-0.0844, -0.0115]*+ | -0.0179 [-0.0522, +0.0180]+ | +0.0893 [+0.0484, +0.1279]*+ | -0.0122 [-0.0550, +0.0250]+ | -0.0386 [-0.0791, -0.0021]*+ | -0.0503 [-0.0813, -0.0211]*+ |

**n_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 172 | 155 | -0.0502 [-0.0600, -0.0400]* | -0.0492 [-0.0578, -0.0404]* | -0.0482 [-0.0583, -0.0374]* | -0.0849 [-0.0951, -0.0746]* | -0.0347 [-0.0468, -0.0229]* | +0.0698 [+0.0556, +0.0842]* | -0.0056 [-0.0182, +0.0077] | -0.0293 [-0.0409, -0.0169]* | -0.0540 [-0.0658, -0.0425]* |
| 1 | 701 | 588 | -0.0529 [-0.0585, -0.0476]* | -0.0507 [-0.0563, -0.0454]* | -0.0488 [-0.0540, -0.0433]* | -0.0853 [-0.0927, -0.0779]* | -0.0505 [-0.0576, -0.0426]* | +0.0347 [+0.0253, +0.0451]* | -0.0171 [-0.0230, -0.0110]* | -0.0344 [-0.0400, -0.0287]* | -0.0633 [-0.0705, -0.0560]* |
| 2 | 400 | 354 | -0.0479 [-0.0544, -0.0413]* | -0.0440 [-0.0505, -0.0377]* | -0.0462 [-0.0530, -0.0399]* | -0.0692 [-0.0776, -0.0605]* | +0.0022 [-0.0087, +0.0122] | +0.1032 [+0.0880, +0.1187]* | -0.0219 [-0.0293, -0.0142]* | -0.0363 [-0.0434, -0.0294]* | -0.0521 [-0.0609, -0.0436]* |
| 3 | 65 | 64 | -0.0386 [-0.0579, -0.0211]* | -0.0446 [-0.0632, -0.0282]* | -0.0301 [-0.0492, -0.0114]* | -0.0534 [-0.0751, -0.0308]* | +0.0112 [-0.0192, +0.0411] | +0.1232 [+0.0896, +0.1592]* | -0.0014 [-0.0210, +0.0172] | -0.0231 [-0.0404, -0.0060]* | -0.0355 [-0.0586, -0.0106]* |
| 4 | n<10 | |  |  |  |  |  |  |  |  |  |

**n_measurement**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=25 | 348 | 328 | -0.0589 [-0.0662, -0.0518]* | -0.0572 [-0.0640, -0.0502]* | -0.0518 [-0.0587, -0.0447]* | -0.0824 [-0.0919, -0.0724]* | -0.0466 [-0.0579, -0.0360]* | +0.0555 [+0.0418, +0.0692]* | -0.0118 [-0.0206, -0.0029]* | -0.0386 [-0.0466, -0.0303]* | -0.0537 [-0.0636, -0.0439]* |
| (25,33] | 336 | 335 | -0.0505 [-0.0576, -0.0435]* | -0.0537 [-0.0599, -0.0474]* | -0.0458 [-0.0526, -0.0392]* | -0.1129 [-0.1216, -0.1037]* | -0.0139 [-0.0241, -0.0039]* | +0.0479 [+0.0337, +0.0625]* | -0.0148 [-0.0236, -0.0063]* | -0.0245 [-0.0327, -0.0168]* | -0.0858 [-0.0952, -0.0759]* |
| (33,60] | 339 | 324 | -0.0505 [-0.0578, -0.0429]* | -0.0501 [-0.0566, -0.0435]* | -0.0483 [-0.0557, -0.0408]* | -0.0726 [-0.0808, -0.0644]* | -0.0291 [-0.0404, -0.0179]* | +0.0709 [+0.0560, +0.0868]* | -0.0198 [-0.0274, -0.0114]* | -0.0343 [-0.0415, -0.0268]* | -0.0533 [-0.0619, -0.0451]* |
| >60 | 322 | 237 | -0.0417 [-0.0493, -0.0348]* | -0.0312 [-0.0388, -0.0245]* | -0.0427 [-0.0506, -0.0355]* | -0.0468 [-0.0553, -0.0391]* | -0.0283 [-0.0387, -0.0186]* | +0.0849 [+0.0703, +0.0998]* | -0.0190 [-0.0274, -0.0110]* | -0.0382 [-0.0461, -0.0304]* | -0.0364 [-0.0450, -0.0291]* |

**n_vital**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 16 | -0.0230 [-0.0570, +0.0117]+ | +0.0028 [-0.0393, +0.0456]+ | -0.0109 [-0.0470, +0.0266]+ | -0.0100 [-0.0516, +0.0288]+ | -0.0439 [-0.0921, +0.0063]+ | +0.0933 [+0.0166, +0.1755]*+ | -0.0162 [-0.0647, +0.0303]+ | -0.0055 [-0.0437, +0.0314]+ | +0.0096 [-0.0271, +0.0476]+ |
| 1 | n<10 | |  |  |  |  |  |  |  |  |  |
| 2 | n<10 | |  |  |  |  |  |  |  |  |  |
| 4 | n<10 | |  |  |  |  |  |  |  |  |  |
| 5 | 24 | 23 | -0.0489 [-0.0705, -0.0271]*+ | -0.0435 [-0.0693, -0.0149]*+ | -0.0531 [-0.0782, -0.0272]*+ | -0.0653 [-0.0983, -0.0309]*+ | -0.0355 [-0.0594, -0.0098]*+ | +0.1042 [+0.0387, +0.1827]*+ | -0.0055 [-0.0344, +0.0257]+ | -0.0460 [-0.0745, -0.0154]*+ | -0.0551 [-0.0896, -0.0183]*+ |
| 6 | 1298 | 990 | -0.0511 [-0.0555, -0.0468]* | -0.0490 [-0.0537, -0.0445]* | -0.0477 [-0.0520, -0.0434]* | -0.0802 [-0.0861, -0.0741]* | -0.0295 [-0.0356, -0.0235]* | +0.0628 [+0.0548, +0.0714]* | -0.0169 [-0.0215, -0.0118]* | -0.0342 [-0.0388, -0.0298]* | -0.0585 [-0.0644, -0.0523]* |

**n_diagnosis**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0514 [-0.0565, -0.0464]* | -0.0569 [-0.0617, -0.0519]* | -0.0471 [-0.0521, -0.0420]* | -0.1140 [-0.1207, -0.1066]* | -0.0225 [-0.0306, -0.0152]* | +0.0507 [+0.0411, +0.0609]* | -0.0136 [-0.0196, -0.0073]* | -0.0271 [-0.0328, -0.0214]* | -0.0865 [-0.0934, -0.0786]* |
| (0,3] | 390 | 334 | -0.0494 [-0.0566, -0.0423]* | -0.0417 [-0.0483, -0.0354]* | -0.0449 [-0.0520, -0.0375]* | -0.0378 [-0.0452, -0.0310]* | -0.0324 [-0.0429, -0.0215]* | +0.0763 [+0.0612, +0.0915]* | -0.0172 [-0.0254, -0.0086]* | -0.0377 [-0.0455, -0.0300]* | -0.0216 [-0.0295, -0.0141]* |
| >3 | 275 | 204 | -0.0501 [-0.0582, -0.0421]* | -0.0365 [-0.0452, -0.0285]* | -0.0510 [-0.0596, -0.0427]* | -0.0512 [-0.0600, -0.0427]* | -0.0433 [-0.0549, -0.0314]* | +0.0819 [+0.0653, +0.0985]* | -0.0216 [-0.0316, -0.0119]* | -0.0454 [-0.0541, -0.0369]* | -0.0367 [-0.0450, -0.0289]* |

**share_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0137 | 340 | 277 | -0.0462 [-0.0534, -0.0389]* | -0.0411 [-0.0481, -0.0344]* | -0.0461 [-0.0538, -0.0383]* | -0.0674 [-0.0756, -0.0596]* | -0.0396 [-0.0489, -0.0305]* | +0.0697 [+0.0575, +0.0820]* | -0.0112 [-0.0199, -0.0024]* | -0.0334 [-0.0418, -0.0254]* | -0.0463 [-0.0545, -0.0386]* |
| (0.0137,0.027] | 356 | 321 | -0.0500 [-0.0567, -0.0432]* | -0.0465 [-0.0528, -0.0400]* | -0.0460 [-0.0528, -0.0390]* | -0.0737 [-0.0823, -0.0653]* | -0.0397 [-0.0487, -0.0304]* | +0.0535 [+0.0391, +0.0689]* | -0.0213 [-0.0287, -0.0130]* | -0.0352 [-0.0424, -0.0279]* | -0.0557 [-0.0643, -0.0470]* |
| (0.027,0.0444] | 318 | 305 | -0.0523 [-0.0604, -0.0441]* | -0.0536 [-0.0608, -0.0463]* | -0.0485 [-0.0559, -0.0410]* | -0.0994 [-0.1092, -0.0889]* | -0.0197 [-0.0309, -0.0086]* | +0.0384 [+0.0256, +0.0528]* | -0.0145 [-0.0228, -0.0058]* | -0.0290 [-0.0371, -0.0207]* | -0.0761 [-0.0858, -0.0658]* |
| >0.0444 | 331 | 316 | -0.0539 [-0.0613, -0.0466]* | -0.0526 [-0.0591, -0.0459]* | -0.0486 [-0.0554, -0.0415]* | -0.0772 [-0.0861, -0.0679]* | -0.0181 [-0.0316, -0.0056]* | +0.0963 [+0.0808, +0.1121]* | -0.0179 [-0.0266, -0.0094]* | -0.0378 [-0.0454, -0.0300]* | -0.0530 [-0.0630, -0.0430]* |

**share_measurement**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.767 | 337 | 317 | -0.0575 [-0.0650, -0.0502]* | -0.0554 [-0.0626, -0.0482]* | -0.0503 [-0.0577, -0.0427]* | -0.0714 [-0.0807, -0.0618]* | -0.0453 [-0.0572, -0.0346]* | +0.0581 [+0.0438, +0.0722]* | -0.0100 [-0.0187, -0.0014]* | -0.0387 [-0.0469, -0.0305]* | -0.0438 [-0.0538, -0.0337]* |
| (0.767,0.814] | 339 | 330 | -0.0548 [-0.0617, -0.0480]* | -0.0576 [-0.0640, -0.0517]* | -0.0518 [-0.0583, -0.0455]* | -0.1157 [-0.1243, -0.1069]* | -0.0085 [-0.0199, +0.0022] | +0.0587 [+0.0444, +0.0730]* | -0.0190 [-0.0271, -0.0105]* | -0.0299 [-0.0373, -0.0228]* | -0.0901 [-0.0987, -0.0809]* |
| (0.814,0.86] | 333 | 313 | -0.0480 [-0.0557, -0.0411]* | -0.0478 [-0.0544, -0.0415]* | -0.0439 [-0.0509, -0.0372]* | -0.0793 [-0.0878, -0.0710]* | -0.0323 [-0.0427, -0.0219]* | +0.0634 [+0.0485, +0.0792]* | -0.0179 [-0.0260, -0.0096]* | -0.0303 [-0.0381, -0.0231]* | -0.0588 [-0.0682, -0.0499]* |
| >0.86 | 336 | 256 | -0.0417 [-0.0492, -0.0348]* | -0.0324 [-0.0393, -0.0260]* | -0.0430 [-0.0508, -0.0353]* | -0.0494 [-0.0572, -0.0418]* | -0.0325 [-0.0424, -0.0234]* | +0.0779 [+0.0626, +0.0934]* | -0.0183 [-0.0258, -0.0105]* | -0.0367 [-0.0437, -0.0292]* | -0.0371 [-0.0445, -0.0297]* |

**share_vital**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0845 | 341 | 253 | -0.0421 [-0.0497, -0.0351]* | -0.0303 [-0.0378, -0.0238]* | -0.0430 [-0.0510, -0.0358]* | -0.0456 [-0.0536, -0.0383]* | -0.0311 [-0.0419, -0.0213]* | +0.0849 [+0.0695, +0.0995]* | -0.0188 [-0.0268, -0.0106]* | -0.0374 [-0.0451, -0.0299]* | -0.0336 [-0.0416, -0.0264]* |
| (0.0845,0.146] | 350 | 334 | -0.0499 [-0.0570, -0.0428]* | -0.0529 [-0.0595, -0.0462]* | -0.0480 [-0.0545, -0.0410]* | -0.0757 [-0.0842, -0.0675]* | -0.0264 [-0.0365, -0.0157]* | +0.0732 [+0.0587, +0.0887]* | -0.0210 [-0.0283, -0.0131]* | -0.0339 [-0.0409, -0.0266]* | -0.0571 [-0.0663, -0.0483]* |
| (0.146,0.182] | 324 | 323 | -0.0525 [-0.0594, -0.0457]* | -0.0527 [-0.0586, -0.0468]* | -0.0477 [-0.0543, -0.0410]* | -0.1140 [-0.1224, -0.1052]* | -0.0130 [-0.0235, -0.0029]* | +0.0435 [+0.0296, +0.0587]* | -0.0153 [-0.0240, -0.0065]* | -0.0274 [-0.0352, -0.0198]* | -0.0877 [-0.0968, -0.0782]* |
| >0.182 | 330 | 315 | -0.0581 [-0.0656, -0.0507]* | -0.0578 [-0.0650, -0.0504]* | -0.0504 [-0.0576, -0.0428]* | -0.0828 [-0.0926, -0.0726]* | -0.0479 [-0.0599, -0.0373]* | +0.0548 [+0.0413, +0.0686]* | -0.0098 [-0.0191, -0.0007]* | -0.0366 [-0.0448, -0.0278]* | -0.0530 [-0.0634, -0.0426]* |

**share_diagnosis**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0514 [-0.0565, -0.0464]* | -0.0569 [-0.0617, -0.0519]* | -0.0471 [-0.0521, -0.0420]* | -0.1140 [-0.1207, -0.1066]* | -0.0225 [-0.0306, -0.0152]* | +0.0507 [+0.0411, +0.0609]* | -0.0136 [-0.0196, -0.0073]* | -0.0271 [-0.0328, -0.0214]* | -0.0865 [-0.0934, -0.0786]* |
| (0,0.0441] | 329 | 259 | -0.0440 [-0.0516, -0.0369]* | -0.0398 [-0.0471, -0.0331]* | -0.0436 [-0.0516, -0.0360]* | -0.0470 [-0.0548, -0.0394]* | -0.0333 [-0.0447, -0.0218]* | +0.0738 [+0.0580, +0.0907]* | -0.0188 [-0.0263, -0.0107]* | -0.0365 [-0.0441, -0.0287]* | -0.0343 [-0.0417, -0.0270]* |
| >0.0441 | 336 | 257 | -0.0552 [-0.0633, -0.0471]* | -0.0392 [-0.0467, -0.0324]* | -0.0511 [-0.0588, -0.0429]* | -0.0397 [-0.0478, -0.0320]* | -0.0404 [-0.0515, -0.0288]* | +0.0833 [+0.0686, +0.0976]* | -0.0192 [-0.0294, -0.0082]* | -0.0451 [-0.0535, -0.0361]* | -0.0214 [-0.0303, -0.0129]* |

**n_knowledge**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=21 | 347 | 331 | -0.0527 [-0.0596, -0.0457]* | -0.0528 [-0.0590, -0.0463]* | -0.0458 [-0.0526, -0.0385]* | -0.0875 [-0.0970, -0.0778]* | -0.0412 [-0.0524, -0.0306]* | +0.0532 [+0.0393, +0.0663]* | -0.0109 [-0.0191, -0.0020]* | -0.0313 [-0.0396, -0.0233]* | -0.0594 [-0.0692, -0.0490]* |
| (21,25] | 407 | 389 | -0.0514 [-0.0588, -0.0445]* | -0.0551 [-0.0613, -0.0489]* | -0.0449 [-0.0512, -0.0385]* | -0.0989 [-0.1071, -0.0907]* | -0.0136 [-0.0241, -0.0025]* | +0.0578 [+0.0445, +0.0722]* | -0.0184 [-0.0261, -0.0100]* | -0.0291 [-0.0369, -0.0215]* | -0.0760 [-0.0846, -0.0673]* |
| (25,30] | 290 | 265 | -0.0490 [-0.0564, -0.0416]* | -0.0464 [-0.0535, -0.0395]* | -0.0488 [-0.0567, -0.0407]* | -0.0682 [-0.0770, -0.0594]* | -0.0244 [-0.0365, -0.0132]* | +0.0737 [+0.0574, +0.0902]* | -0.0197 [-0.0280, -0.0109]* | -0.0369 [-0.0452, -0.0289]* | -0.0510 [-0.0597, -0.0424]* |
| >30 | 301 | 218 | -0.0484 [-0.0565, -0.0411]* | -0.0357 [-0.0437, -0.0289]* | -0.0507 [-0.0592, -0.0433]* | -0.0529 [-0.0616, -0.0455]* | -0.0430 [-0.0544, -0.0326]* | +0.0778 [+0.0630, +0.0926]* | -0.0165 [-0.0250, -0.0078]* | -0.0405 [-0.0487, -0.0329]* | -0.0365 [-0.0441, -0.0295]* |

**n_analyte**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=23 | 367 | 328 | -0.0588 [-0.0663, -0.0519]* | -0.0530 [-0.0602, -0.0460]* | -0.0511 [-0.0584, -0.0436]* | -0.0677 [-0.0771, -0.0585]* | -0.0503 [-0.0619, -0.0388]* | +0.0642 [+0.0509, +0.0777]* | -0.0104 [-0.0192, -0.0018]* | -0.0403 [-0.0483, -0.0323]* | -0.0415 [-0.0514, -0.0320]* |
| (23,29] | 388 | 344 | -0.0456 [-0.0520, -0.0390]* | -0.0489 [-0.0547, -0.0429]* | -0.0469 [-0.0535, -0.0402]* | -0.0934 [-0.1020, -0.0845]* | -0.0171 [-0.0269, -0.0069]* | +0.0487 [+0.0349, +0.0636]* | -0.0150 [-0.0227, -0.0065]* | -0.0264 [-0.0334, -0.0192]* | -0.0696 [-0.0780, -0.0606]* |
| (29,33] | 294 | 274 | -0.0537 [-0.0619, -0.0458]* | -0.0506 [-0.0576, -0.0436]* | -0.0495 [-0.0582, -0.0416]* | -0.0835 [-0.0923, -0.0747]* | -0.0341 [-0.0457, -0.0233]* | +0.0641 [+0.0489, +0.0802]* | -0.0252 [-0.0335, -0.0168]* | -0.0371 [-0.0456, -0.0287]* | -0.0639 [-0.0731, -0.0548]* |
| >33 | 296 | 258 | -0.0437 [-0.0510, -0.0360]* | -0.0395 [-0.0474, -0.0320]* | -0.0407 [-0.0481, -0.0333]* | -0.0699 [-0.0797, -0.0607]* | -0.0160 [-0.0270, -0.0053]* | +0.0860 [+0.0705, +0.1023]* | -0.0165 [-0.0247, -0.0085]* | -0.0325 [-0.0401, -0.0251]* | -0.0551 [-0.0649, -0.0454]* |

**n_prior_dx**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0514 [-0.0565, -0.0464]* | -0.0569 [-0.0617, -0.0519]* | -0.0471 [-0.0521, -0.0420]* | -0.1140 [-0.1207, -0.1066]* | -0.0225 [-0.0306, -0.0152]* | +0.0507 [+0.0411, +0.0609]* | -0.0136 [-0.0196, -0.0073]* | -0.0271 [-0.0328, -0.0214]* | -0.0865 [-0.0934, -0.0786]* |
| (0,3] | 390 | 334 | -0.0494 [-0.0566, -0.0423]* | -0.0417 [-0.0483, -0.0354]* | -0.0449 [-0.0520, -0.0375]* | -0.0378 [-0.0452, -0.0310]* | -0.0324 [-0.0429, -0.0215]* | +0.0763 [+0.0612, +0.0915]* | -0.0172 [-0.0254, -0.0086]* | -0.0377 [-0.0455, -0.0300]* | -0.0216 [-0.0295, -0.0141]* |
| >3 | 275 | 204 | -0.0501 [-0.0582, -0.0421]* | -0.0365 [-0.0452, -0.0285]* | -0.0510 [-0.0596, -0.0427]* | -0.0512 [-0.0600, -0.0427]* | -0.0433 [-0.0549, -0.0314]* | +0.0819 [+0.0653, +0.0985]* | -0.0216 [-0.0316, -0.0119]* | -0.0454 [-0.0541, -0.0369]* | -0.0367 [-0.0450, -0.0289]* |

**rel_trajectory_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 1039 | 907 | -0.0526 [-0.0571, -0.0482]* | -0.0535 [-0.0578, -0.0492]* | -0.0481 [-0.0524, -0.0436]* | -0.0890 [-0.0947, -0.0828]* | -0.0287 [-0.0353, -0.0224]* | +0.0591 [+0.0507, +0.0680]* | -0.0159 [-0.0209, -0.0106]* | -0.0319 [-0.0366, -0.0271]* | -0.0645 [-0.0707, -0.0581]* |
| >0 | 306 | 229 | -0.0438 [-0.0524, -0.0354]* | -0.0307 [-0.0387, -0.0235]* | -0.0445 [-0.0537, -0.0362]* | -0.0451 [-0.0535, -0.0377]* | -0.0327 [-0.0447, -0.0211]* | +0.0831 [+0.0670, +0.0984]* | -0.0177 [-0.0275, -0.0075]* | -0.0408 [-0.0499, -0.0319]* | -0.0336 [-0.0416, -0.0264]* |

**rel_recurrence_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0514 [-0.0565, -0.0464]* | -0.0569 [-0.0617, -0.0519]* | -0.0471 [-0.0521, -0.0420]* | -0.1140 [-0.1207, -0.1066]* | -0.0225 [-0.0306, -0.0152]* | +0.0507 [+0.0411, +0.0609]* | -0.0136 [-0.0196, -0.0073]* | -0.0271 [-0.0328, -0.0214]* | -0.0865 [-0.0934, -0.0786]* |
| (0,3] | 390 | 334 | -0.0494 [-0.0566, -0.0423]* | -0.0417 [-0.0483, -0.0354]* | -0.0449 [-0.0520, -0.0375]* | -0.0378 [-0.0452, -0.0310]* | -0.0324 [-0.0429, -0.0215]* | +0.0763 [+0.0612, +0.0915]* | -0.0172 [-0.0254, -0.0086]* | -0.0377 [-0.0455, -0.0300]* | -0.0216 [-0.0295, -0.0141]* |
| >3 | 275 | 204 | -0.0501 [-0.0582, -0.0421]* | -0.0365 [-0.0452, -0.0285]* | -0.0510 [-0.0596, -0.0427]* | -0.0512 [-0.0600, -0.0427]* | -0.0433 [-0.0549, -0.0314]* | +0.0819 [+0.0653, +0.0985]* | -0.0216 [-0.0316, -0.0119]* | -0.0454 [-0.0541, -0.0369]* | -0.0367 [-0.0450, -0.0289]* |

**rel_comorbid_with**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 862 | 783 | -0.0506 [-0.0554, -0.0460]* | -0.0538 [-0.0584, -0.0490]* | -0.0461 [-0.0509, -0.0412]* | -0.0978 [-0.1043, -0.0912]* | -0.0235 [-0.0308, -0.0164]* | +0.0569 [+0.0474, +0.0667]* | -0.0136 [-0.0190, -0.0079]* | -0.0293 [-0.0344, -0.0240]* | -0.0732 [-0.0797, -0.0662]* |
| (0,2] | 159 | 144 | -0.0580 [-0.0692, -0.0460]* | -0.0449 [-0.0547, -0.0354]* | -0.0540 [-0.0649, -0.0428]* | -0.0394 [-0.0506, -0.0293]* | -0.0483 [-0.0638, -0.0317]* | +0.0732 [+0.0534, +0.0934]* | -0.0268 [-0.0382, -0.0143]* | -0.0441 [-0.0565, -0.0316]* | -0.0195 [-0.0302, -0.0089]* |
| >2 | 324 | 229 | -0.0468 [-0.0545, -0.0394]* | -0.0355 [-0.0431, -0.0284]* | -0.0469 [-0.0547, -0.0396]* | -0.0485 [-0.0565, -0.0411]* | -0.0369 [-0.0483, -0.0256]* | +0.0806 [+0.0657, +0.0957]* | -0.0184 [-0.0274, -0.0091]* | -0.0410 [-0.0487, -0.0328]* | -0.0344 [-0.0424, -0.0267]* |

**rel_co_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 873 | 697 | -0.0524 [-0.0575, -0.0472]* | -0.0504 [-0.0554, -0.0455]* | -0.0487 [-0.0535, -0.0434]* | -0.0852 [-0.0920, -0.0785]* | -0.0473 [-0.0541, -0.0404]* | +0.0416 [+0.0331, +0.0510]* | -0.0149 [-0.0205, -0.0090]* | -0.0334 [-0.0387, -0.0278]* | -0.0615 [-0.0680, -0.0544]* |
| 2 | 400 | 354 | -0.0479 [-0.0544, -0.0413]* | -0.0440 [-0.0505, -0.0377]* | -0.0462 [-0.0530, -0.0399]* | -0.0692 [-0.0776, -0.0605]* | +0.0022 [-0.0087, +0.0122] | +0.1032 [+0.0880, +0.1187]* | -0.0219 [-0.0293, -0.0142]* | -0.0363 [-0.0434, -0.0294]* | -0.0521 [-0.0609, -0.0436]* |
| 6 | 65 | 64 | -0.0386 [-0.0579, -0.0211]* | -0.0446 [-0.0632, -0.0282]* | -0.0301 [-0.0492, -0.0114]* | -0.0534 [-0.0751, -0.0308]* | +0.0112 [-0.0192, +0.0411] | +0.1232 [+0.0896, +0.1592]* | -0.0014 [-0.0210, +0.0172] | -0.0231 [-0.0404, -0.0060]* | -0.0355 [-0.0586, -0.0106]* |
| 12 | n<10 | |  |  |  |  |  |  |  |  |  |

**rel_baseline_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 768 | 755 | -0.0526 [-0.0577, -0.0475]* | -0.0566 [-0.0613, -0.0517]* | -0.0475 [-0.0525, -0.0425]* | -0.1041 [-0.1104, -0.0973]* | -0.0226 [-0.0298, -0.0156]* | +0.0538 [+0.0449, +0.0636]* | -0.0137 [-0.0192, -0.0080]* | -0.0284 [-0.0338, -0.0231]* | -0.0764 [-0.0831, -0.0690]* |
| (0,25] | 287 | 242 | -0.0535 [-0.0622, -0.0447]* | -0.0403 [-0.0485, -0.0326]* | -0.0499 [-0.0587, -0.0411]* | -0.0423 [-0.0508, -0.0347]* | -0.0436 [-0.0558, -0.0304]* | +0.0777 [+0.0590, +0.0953]* | -0.0202 [-0.0307, -0.0092]* | -0.0439 [-0.0531, -0.0346]* | -0.0271 [-0.0360, -0.0190]* |
| >25 | 290 | 233 | -0.0424 [-0.0502, -0.0353]* | -0.0342 [-0.0415, -0.0274]* | -0.0439 [-0.0518, -0.0364]* | -0.0490 [-0.0572, -0.0412]* | -0.0346 [-0.0455, -0.0236]* | +0.0800 [+0.0666, +0.0936]* | -0.0195 [-0.0276, -0.0109]* | -0.0384 [-0.0462, -0.0307]* | -0.0374 [-0.0456, -0.0299]* |

**rel_medical_total**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=63 | 353 | 327 | -0.0575 [-0.0649, -0.0501]* | -0.0534 [-0.0601, -0.0465]* | -0.0491 [-0.0563, -0.0415]* | -0.0735 [-0.0827, -0.0642]* | -0.0552 [-0.0666, -0.0442]* | +0.0580 [+0.0442, +0.0714]* | -0.0116 [-0.0207, -0.0025]* | -0.0387 [-0.0469, -0.0303]* | -0.0473 [-0.0574, -0.0378]* |
| (63,75] | 352 | 327 | -0.0490 [-0.0561, -0.0421]* | -0.0489 [-0.0553, -0.0424]* | -0.0473 [-0.0539, -0.0408]* | -0.0972 [-0.1063, -0.0878]* | -0.0127 [-0.0231, -0.0027]* | +0.0542 [+0.0395, +0.0687]* | -0.0133 [-0.0211, -0.0054]* | -0.0274 [-0.0349, -0.0200]* | -0.0730 [-0.0826, -0.0638]* |
| (75,87] | 320 | 291 | -0.0514 [-0.0591, -0.0438]* | -0.0541 [-0.0610, -0.0475]* | -0.0506 [-0.0589, -0.0426]* | -0.0890 [-0.0977, -0.0801]* | -0.0278 [-0.0396, -0.0160]* | +0.0612 [+0.0456, +0.0775]* | -0.0266 [-0.0341, -0.0189]* | -0.0364 [-0.0443, -0.0284]* | -0.0692 [-0.0780, -0.0608]* |
| >87 | 320 | 251 | -0.0438 [-0.0515, -0.0370]* | -0.0362 [-0.0437, -0.0290]* | -0.0418 [-0.0492, -0.0346]* | -0.0553 [-0.0641, -0.0471]* | -0.0219 [-0.0326, -0.0120]* | +0.0864 [+0.0725, +0.1007]* | -0.0146 [-0.0229, -0.0067]* | -0.0333 [-0.0406, -0.0263]* | -0.0400 [-0.0488, -0.0323]* |

**rel_medical_member_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=21 | 354 | 317 | -0.0591 [-0.0664, -0.0520]* | -0.0528 [-0.0602, -0.0453]* | -0.0509 [-0.0581, -0.0433]* | -0.0659 [-0.0756, -0.0566]* | -0.0521 [-0.0640, -0.0406]* | +0.0638 [+0.0501, +0.0780]* | -0.0099 [-0.0190, -0.0011]* | -0.0404 [-0.0487, -0.0319]* | -0.0393 [-0.0493, -0.0297]* |
| (21,28] | 483 | 424 | -0.0475 [-0.0537, -0.0416]* | -0.0481 [-0.0534, -0.0428]* | -0.0486 [-0.0549, -0.0423]* | -0.0902 [-0.0983, -0.0826]* | -0.0227 [-0.0323, -0.0134]* | +0.0486 [+0.0365, +0.0614]* | -0.0166 [-0.0236, -0.0092]* | -0.0288 [-0.0355, -0.0224]* | -0.0662 [-0.0740, -0.0582]* |
| (28,31] | 188 | 180 | -0.0514 [-0.0612, -0.0421]* | -0.0483 [-0.0566, -0.0400]* | -0.0451 [-0.0545, -0.0364]* | -0.0845 [-0.0951, -0.0737]* | -0.0295 [-0.0426, -0.0178]* | +0.0667 [+0.0479, +0.0853]* | -0.0246 [-0.0350, -0.0142]* | -0.0341 [-0.0453, -0.0241]* | -0.0661 [-0.0772, -0.0549]* |
| >31 | 320 | 280 | -0.0452 [-0.0528, -0.0375]* | -0.0436 [-0.0514, -0.0359]* | -0.0425 [-0.0499, -0.0351]* | -0.0735 [-0.0831, -0.0642]* | -0.0152 [-0.0259, -0.0046]* | +0.0881 [+0.0731, +0.1041]* | -0.0181 [-0.0263, -0.0096]* | -0.0343 [-0.0419, -0.0267]* | -0.0595 [-0.0696, -0.0500]* |

**rel_medical_assesses**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=12 | 457 | 400 | -0.0516 [-0.0588, -0.0447]* | -0.0477 [-0.0540, -0.0414]* | -0.0485 [-0.0551, -0.0416]* | -0.0706 [-0.0789, -0.0627]* | -0.0480 [-0.0578, -0.0376]* | +0.0601 [+0.0483, +0.0731]* | -0.0130 [-0.0214, -0.0049]* | -0.0349 [-0.0428, -0.0268]* | -0.0453 [-0.0542, -0.0370]* |
| (12,15] | 459 | 401 | -0.0523 [-0.0581, -0.0463]* | -0.0521 [-0.0576, -0.0465]* | -0.0482 [-0.0538, -0.0422]* | -0.0893 [-0.0969, -0.0812]* | -0.0187 [-0.0280, -0.0099]* | +0.0595 [+0.0469, +0.0723]* | -0.0196 [-0.0265, -0.0125]* | -0.0318 [-0.0383, -0.0254]* | -0.0679 [-0.0754, -0.0602]* |
| (15,16] | 142 | 133 | -0.0535 [-0.0644, -0.0427]* | -0.0498 [-0.0588, -0.0406]* | -0.0519 [-0.0640, -0.0403]* | -0.0854 [-0.0985, -0.0726]* | -0.0328 [-0.0515, -0.0156]* | +0.0569 [+0.0369, +0.0756]* | -0.0208 [-0.0318, -0.0094]* | -0.0367 [-0.0486, -0.0248]* | -0.0608 [-0.0731, -0.0486]* |
| >16 | 287 | 255 | -0.0446 [-0.0524, -0.0370]* | -0.0424 [-0.0504, -0.0347]* | -0.0414 [-0.0493, -0.0338]* | -0.0730 [-0.0827, -0.0631]* | -0.0164 [-0.0282, -0.0048]* | +0.0834 [+0.0674, +0.1008]* | -0.0141 [-0.0229, -0.0055]* | -0.0342 [-0.0417, -0.0262]* | -0.0586 [-0.0687, -0.0490]* |

**rel_medical_measures**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=22 | 382 | 335 | -0.0576 [-0.0649, -0.0505]* | -0.0509 [-0.0581, -0.0437]* | -0.0492 [-0.0563, -0.0417]* | -0.0646 [-0.0740, -0.0558]* | -0.0488 [-0.0603, -0.0373]* | +0.0658 [+0.0530, +0.0786]* | -0.0084 [-0.0171, +0.0007] | -0.0399 [-0.0477, -0.0321]* | -0.0394 [-0.0488, -0.0301]* |
| (22,26] | 525 | 457 | -0.0512 [-0.0573, -0.0450]* | -0.0509 [-0.0561, -0.0456]* | -0.0514 [-0.0571, -0.0453]* | -0.0931 [-0.1001, -0.0858]* | -0.0254 [-0.0347, -0.0161]* | +0.0528 [+0.0404, +0.0663]* | -0.0220 [-0.0288, -0.0149]* | -0.0333 [-0.0401, -0.0263]* | -0.0709 [-0.0784, -0.0631]* |
| (26,28] | 152 | 146 | -0.0425 [-0.0525, -0.0333]* | -0.0425 [-0.0516, -0.0336]* | -0.0383 [-0.0480, -0.0290]* | -0.0754 [-0.0880, -0.0636]* | -0.0126 [-0.0293, +0.0034] | +0.0761 [+0.0537, +0.0985]* | -0.0117 [-0.0228, -0.0002]* | -0.0319 [-0.0424, -0.0218]* | -0.0623 [-0.0753, -0.0498]* |
| >28 | 286 | 246 | -0.0442 [-0.0519, -0.0366]* | -0.0432 [-0.0508, -0.0358]* | -0.0419 [-0.0493, -0.0343]* | -0.0745 [-0.0847, -0.0647]* | -0.0209 [-0.0299, -0.0116]* | +0.0781 [+0.0631, +0.0942]* | -0.0189 [-0.0269, -0.0107]* | -0.0281 [-0.0358, -0.0205]* | -0.0545 [-0.0644, -0.0445]* |

**rel_medical_is_a**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=1 | 506 | 506 | -0.0523 [-0.0581, -0.0464]* | -0.0585 [-0.0638, -0.0527]* | -0.0471 [-0.0529, -0.0411]* | -0.1181 [-0.1255, -0.1102]* | -0.0376 [-0.0461, -0.0294]* | +0.0299 [+0.0200, +0.0405]* | -0.0140 [-0.0208, -0.0067]* | -0.0276 [-0.0343, -0.0211]* | -0.0902 [-0.0980, -0.0815]* |
| (1,3] | 189 | 188 | -0.0497 [-0.0594, -0.0400]* | -0.0530 [-0.0625, -0.0440]* | -0.0487 [-0.0582, -0.0393]* | -0.1008 [-0.1121, -0.0895]* | +0.0163 [-0.0005, +0.0317] | +0.1107 [+0.0910, +0.1324]* | -0.0122 [-0.0228, -0.0013]* | -0.0274 [-0.0376, -0.0169]* | -0.0741 [-0.0857, -0.0618]* |
| (3,10] | 349 | 307 | -0.0504 [-0.0582, -0.0428]* | -0.0427 [-0.0494, -0.0362]* | -0.0457 [-0.0535, -0.0378]* | -0.0373 [-0.0451, -0.0300]* | -0.0364 [-0.0475, -0.0251]* | +0.0738 [+0.0581, +0.0893]* | -0.0171 [-0.0256, -0.0082]* | -0.0372 [-0.0455, -0.0291]* | -0.0202 [-0.0284, -0.0120]* |
| >10 | 301 | 223 | -0.0483 [-0.0559, -0.0410]* | -0.0347 [-0.0430, -0.0272]* | -0.0485 [-0.0567, -0.0407]* | -0.0481 [-0.0568, -0.0403]* | -0.0373 [-0.0488, -0.0259]* | +0.0832 [+0.0679, +0.0981]* | -0.0219 [-0.0310, -0.0126]* | -0.0447 [-0.0526, -0.0366]* | -0.0354 [-0.0436, -0.0278]* |

**rel_medical_may_affect**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 623 | 519 | -0.0510 [-0.0570, -0.0451]* | -0.0480 [-0.0537, -0.0423]* | -0.0463 [-0.0523, -0.0402]* | -0.0761 [-0.0832, -0.0689]* | -0.0413 [-0.0510, -0.0323]* | +0.0738 [+0.0623, +0.0854]* | -0.0185 [-0.0252, -0.0115]* | -0.0372 [-0.0440, -0.0305]* | -0.0555 [-0.0633, -0.0479]* |
| 1 | 479 | 415 | -0.0460 [-0.0527, -0.0396]* | -0.0505 [-0.0568, -0.0441]* | -0.0446 [-0.0513, -0.0383]* | -0.0816 [-0.0906, -0.0732]* | -0.0260 [-0.0345, -0.0171]* | +0.0591 [+0.0469, +0.0717]* | -0.0125 [-0.0198, -0.0048]* | -0.0294 [-0.0364, -0.0223]* | -0.0626 [-0.0714, -0.0544]* |
| 2 | 172 | 162 | -0.0567 [-0.0654, -0.0479]* | -0.0443 [-0.0533, -0.0347]* | -0.0524 [-0.0614, -0.0430]* | -0.0786 [-0.0922, -0.0651]* | -0.0041 [-0.0187, +0.0102] | +0.0550 [+0.0374, +0.0727]* | -0.0220 [-0.0312, -0.0125]* | -0.0364 [-0.0444, -0.0283]* | -0.0552 [-0.0683, -0.0419]* |
| 3 | 54 | 54 | -0.0660 [-0.0837, -0.0490]* | -0.0546 [-0.0703, -0.0385]* | -0.0667 [-0.0860, -0.0486]* | -0.1015 [-0.1233, -0.0775]* | -0.0144 [-0.0480, +0.0164] | +0.0342 [+0.0114, +0.0594]* | -0.0142 [-0.0300, +0.0029] | -0.0322 [-0.0504, -0.0147]* | -0.0590 [-0.0780, -0.0381]* |
| 4 | 11 | 11 | -0.0567 [-0.0844, -0.0324]*+ | -0.0362 [-0.0664, -0.0035]*+ | -0.0471 [-0.0728, -0.0204]*+ | -0.0781 [-0.1216, -0.0362]*+ | -0.0282 [-0.0894, +0.0278]+ | +0.0316 [-0.0171, +0.0819]+ | +0.0088 [-0.0334, +0.0569]+ | -0.0251 [-0.0619, +0.0154]+ | -0.0387 [-0.0849, +0.0103]+ |
| 5 | n<10 | |  |  |  |  |  |  |  |  |  |

**rel_medical_indicates**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 494 | 440 | -0.0520 [-0.0584, -0.0456]* | -0.0449 [-0.0510, -0.0389]* | -0.0448 [-0.0513, -0.0383]* | -0.0786 [-0.0870, -0.0702]* | -0.0367 [-0.0465, -0.0273]* | +0.0472 [+0.0352, +0.0588]* | -0.0186 [-0.0259, -0.0110]* | -0.0339 [-0.0405, -0.0272]* | -0.0558 [-0.0642, -0.0471]* |
| (0,1] | 272 | 254 | -0.0491 [-0.0583, -0.0401]* | -0.0529 [-0.0609, -0.0454]* | -0.0459 [-0.0536, -0.0379]* | -0.0870 [-0.0972, -0.0765]* | -0.0260 [-0.0380, -0.0142]* | +0.0590 [+0.0422, +0.0761]* | -0.0188 [-0.0292, -0.0087]* | -0.0308 [-0.0405, -0.0213]* | -0.0685 [-0.0790, -0.0574]* |
| (1,3] | 267 | 240 | -0.0518 [-0.0599, -0.0432]* | -0.0549 [-0.0625, -0.0475]* | -0.0498 [-0.0581, -0.0416]* | -0.0872 [-0.0970, -0.0778]* | -0.0155 [-0.0280, -0.0023]* | +0.0743 [+0.0573, +0.0928]* | -0.0237 [-0.0330, -0.0146]* | -0.0368 [-0.0456, -0.0281]* | -0.0660 [-0.0751, -0.0572]* |
| >3 | 312 | 243 | -0.0484 [-0.0556, -0.0412]* | -0.0441 [-0.0517, -0.0366]* | -0.0502 [-0.0576, -0.0422]* | -0.0658 [-0.0755, -0.0566]* | -0.0337 [-0.0448, -0.0229]* | +0.0884 [+0.0745, +0.1029]* | -0.0042 [-0.0127, +0.0047] | -0.0341 [-0.0418, -0.0267]* | -0.0433 [-0.0532, -0.0337]* |

### 6.3 cei_R1_ls0, raw

**n_nodes**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=81 | 339 | 317 | -0.0470 [-0.0568, -0.0377]* | -0.0591 [-0.0683, -0.0499]* | -0.0423 [-0.0510, -0.0330]* | -0.0965 [-0.1088, -0.0845]* | -0.0586 [-0.0728, -0.0448]* | +0.0782 [+0.0591, +0.0971]* | +0.0232 [+0.0122, +0.0344]* | -0.0098 [-0.0201, +0.0006] | -0.0456 [-0.0578, -0.0331]* |
| (81,99] | 338 | 336 | -0.0277 [-0.0360, -0.0194]* | -0.0356 [-0.0441, -0.0273]* | -0.0257 [-0.0334, -0.0177]* | -0.0985 [-0.1101, -0.0868]* | -0.0026 [-0.0153, +0.0108] | +0.0576 [+0.0433, +0.0718]* | +0.0336 [+0.0245, +0.0428]* | +0.0215 [+0.0134, +0.0296]* | -0.0447 [-0.0549, -0.0337]* |
| (99,134] | 345 | 327 | -0.0021 [-0.0104, +0.0064] | -0.0061 [-0.0149, +0.0026] | -0.0052 [-0.0134, +0.0035] | -0.0323 [-0.0433, -0.0216]* | -0.0056 [-0.0193, +0.0091] | +0.0798 [+0.0658, +0.0946]* | +0.0292 [+0.0211, +0.0380]* | +0.0178 [+0.0098, +0.0259]* | -0.0053 [-0.0156, +0.0051] |
| >134 | 323 | 240 | +0.0147 [+0.0049, +0.0243]* | +0.0493 [+0.0396, +0.0581]* | +0.0115 [+0.0015, +0.0213]* | +0.0411 [+0.0306, +0.0511]* | +0.0068 [-0.0070, +0.0192] | +0.1010 [+0.0873, +0.1151]* | +0.0346 [+0.0243, +0.0451]* | +0.0208 [+0.0110, +0.0309]* | +0.0540 [+0.0442, +0.0636]* |

**n_evidence**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=32 | 337 | 320 | -0.0460 [-0.0556, -0.0369]* | -0.0594 [-0.0685, -0.0501]* | -0.0415 [-0.0502, -0.0321]* | -0.0964 [-0.1084, -0.0847]* | -0.0558 [-0.0699, -0.0419]* | +0.0751 [+0.0562, +0.0937]* | +0.0249 [+0.0145, +0.0359]* | -0.0078 [-0.0179, +0.0026] | -0.0448 [-0.0568, -0.0323]* |
| (32,41] | 349 | 348 | -0.0264 [-0.0347, -0.0184]* | -0.0355 [-0.0439, -0.0273]* | -0.0247 [-0.0324, -0.0170]* | -0.1003 [-0.1117, -0.0890]* | -0.0056 [-0.0183, +0.0073] | +0.0573 [+0.0436, +0.0708]* | +0.0325 [+0.0233, +0.0418]* | +0.0210 [+0.0132, +0.0289]* | -0.0477 [-0.0577, -0.0362]* |
| (41,70] | 332 | 316 | -0.0073 [-0.0160, +0.0020] | -0.0073 [-0.0162, +0.0013] | -0.0092 [-0.0181, +0.0000] | -0.0308 [-0.0415, -0.0198]* | -0.0058 [-0.0195, +0.0079] | +0.0839 [+0.0694, +0.0995]* | +0.0267 [+0.0180, +0.0360]* | +0.0144 [+0.0060, +0.0230]* | -0.0030 [-0.0130, +0.0077] |
| >70 | 327 | 243 | +0.0178 [+0.0082, +0.0275]* | +0.0506 [+0.0411, +0.0600]* | +0.0140 [+0.0040, +0.0238]* | +0.0424 [+0.0318, +0.0524]* | +0.0067 [-0.0072, +0.0198] | +0.1008 [+0.0873, +0.1145]* | +0.0362 [+0.0261, +0.0465]* | +0.0223 [+0.0126, +0.0320]* | +0.0544 [+0.0450, +0.0638]* |

**n_edges**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=125 | 338 | 319 | -0.0470 [-0.0567, -0.0377]* | -0.0585 [-0.0677, -0.0493]* | -0.0426 [-0.0514, -0.0331]* | -0.0955 [-0.1075, -0.0836]* | -0.0564 [-0.0712, -0.0423]* | +0.0744 [+0.0556, +0.0927]* | +0.0238 [+0.0131, +0.0349]* | -0.0088 [-0.0193, +0.0020] | -0.0440 [-0.0561, -0.0313]* |
| (125,157] | 338 | 335 | -0.0270 [-0.0353, -0.0190]* | -0.0352 [-0.0435, -0.0268]* | -0.0250 [-0.0325, -0.0173]* | -0.0988 [-0.1106, -0.0872]* | -0.0020 [-0.0144, +0.0103] | +0.0576 [+0.0440, +0.0719]* | +0.0332 [+0.0244, +0.0429]* | +0.0202 [+0.0124, +0.0283]* | -0.0461 [-0.0565, -0.0350]* |
| (157,225] | 333 | 315 | -0.0039 [-0.0124, +0.0052] | -0.0091 [-0.0183, -0.0000]* | -0.0069 [-0.0154, +0.0018] | -0.0363 [-0.0476, -0.0253]* | -0.0097 [-0.0238, +0.0051] | +0.0839 [+0.0690, +0.1000]* | +0.0288 [+0.0198, +0.0382]* | +0.0174 [+0.0090, +0.0261]* | -0.0085 [-0.0192, +0.0021] |
| >225 | 336 | 249 | +0.0149 [+0.0051, +0.0246]* | +0.0488 [+0.0393, +0.0575]* | +0.0120 [+0.0021, +0.0214]* | +0.0413 [+0.0309, +0.0508]* | +0.0074 [-0.0063, +0.0199] | +0.0998 [+0.0861, +0.1135]* | +0.0345 [+0.0245, +0.0449]* | +0.0212 [+0.0117, +0.0308]* | +0.0546 [+0.0455, +0.0639]* |

**density**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0127 | 337 | 251 | +0.0185 [+0.0094, +0.0275]* | +0.0475 [+0.0384, +0.0559]* | +0.0144 [+0.0052, +0.0233]* | +0.0378 [+0.0279, +0.0470]* | +0.0052 [-0.0080, +0.0176] | +0.0994 [+0.0853, +0.1137]* | +0.0378 [+0.0278, +0.0477]* | +0.0232 [+0.0138, +0.0327]* | +0.0503 [+0.0409, +0.0596]* |
| (0.0127,0.016] | 336 | 320 | -0.0083 [-0.0170, +0.0002] | -0.0075 [-0.0169, +0.0012] | -0.0088 [-0.0176, +0.0000] | -0.0342 [-0.0464, -0.0232]* | -0.0046 [-0.0177, +0.0085] | +0.0767 [+0.0631, +0.0921]* | +0.0258 [+0.0170, +0.0347]* | +0.0151 [+0.0069, +0.0231]* | -0.0057 [-0.0163, +0.0045] |
| (0.016,0.0191] | 337 | 334 | -0.0250 [-0.0337, -0.0165]* | -0.0345 [-0.0430, -0.0256]* | -0.0240 [-0.0319, -0.0163]* | -0.0974 [-0.1093, -0.0854]* | -0.0045 [-0.0181, +0.0086] | +0.0595 [+0.0454, +0.0737]* | +0.0346 [+0.0260, +0.0439]* | +0.0239 [+0.0157, +0.0317]* | -0.0431 [-0.0534, -0.0323]* |
| >0.0191 | 335 | 312 | -0.0487 [-0.0580, -0.0394]* | -0.0601 [-0.0694, -0.0509]* | -0.0445 [-0.0533, -0.0356]* | -0.0964 [-0.1090, -0.0843]* | -0.0572 [-0.0712, -0.0430]* | +0.0800 [+0.0610, +0.0995]* | +0.0221 [+0.0113, +0.0338]* | -0.0123 [-0.0226, -0.0019]* | -0.0461 [-0.0586, -0.0331]* |

**num_visits**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 679 | 678 | -0.0252 [-0.0314, -0.0193]* | -0.0408 [-0.0473, -0.0346]* | -0.0259 [-0.0319, -0.0202]* | -0.1067 [-0.1154, -0.0981]* | -0.0174 [-0.0276, -0.0083]* | +0.0653 [+0.0547, +0.0762]* | +0.0325 [+0.0261, +0.0393]* | +0.0170 [+0.0111, +0.0231]* | -0.0553 [-0.0635, -0.0472]* |
| 2 | 323 | 323 | -0.0162 [-0.0260, -0.0065]* | -0.0085 [-0.0185, +0.0009] | -0.0132 [-0.0225, -0.0036]* | -0.0060 [-0.0161, +0.0038] | -0.0193 [-0.0317, -0.0060]* | +0.0813 [+0.0659, +0.0972]* | +0.0244 [+0.0151, +0.0343]* | +0.0070 [-0.0027, +0.0168] | +0.0227 [+0.0130, +0.0321]* |
| 3 | 191 | 191 | -0.0079 [-0.0209, +0.0048] | +0.0231 [+0.0105, +0.0356]* | -0.0087 [-0.0216, +0.0044] | +0.0182 [+0.0059, +0.0300]* | -0.0214 [-0.0417, -0.0017]* | +0.0910 [+0.0713, +0.1109]* | +0.0226 [+0.0082, +0.0374]* | -0.0015 [-0.0146, +0.0115] | +0.0343 [+0.0223, +0.0455]* |
| 4 | 91 | 91 | +0.0151 [-0.0060, +0.0352] | +0.0469 [+0.0265, +0.0670]* | +0.0107 [-0.0117, +0.0306] | +0.0393 [+0.0175, +0.0599]* | +0.0108 [-0.0161, +0.0357] | +0.1174 [+0.0928, +0.1416]* | +0.0427 [+0.0223, +0.0629]* | +0.0200 [-0.0007, +0.0397] | +0.0547 [+0.0360, +0.0731]* |
| 5 | 47 | 47 | +0.0149 [-0.0087, +0.0382] | +0.0553 [+0.0346, +0.0782]* | +0.0195 [-0.0023, +0.0423] | +0.0547 [+0.0328, +0.0776]* | +0.0063 [-0.0240, +0.0384] | +0.1241 [+0.0841, +0.1712]* | +0.0267 [+0.0002, +0.0536]* | +0.0141 [-0.0113, +0.0403] | +0.0554 [+0.0297, +0.0801]* |
| 6 | 14 | 14 | +0.0359 [-0.0192, +0.0963]+ | +0.0644 [+0.0113, +0.1199]*+ | +0.0376 [-0.0272, +0.1022]+ | +0.0651 [+0.0091, +0.1175]*+ | +0.0296 [-0.0337, +0.0898]+ | +0.1181 [+0.0737, +0.1617]*+ | +0.0757 [+0.0199, +0.1242]*+ | +0.0575 [+0.0031, +0.1082]*+ | +0.0803 [+0.0403, +0.1184]*+ |

**n_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 172 | 155 | -0.0090 [-0.0216, +0.0037] | +0.0080 [-0.0057, +0.0213] | -0.0056 [-0.0181, +0.0073] | -0.0085 [-0.0230, +0.0062] | -0.0138 [-0.0289, +0.0011] | +0.0660 [+0.0501, +0.0825]* | +0.0521 [+0.0387, +0.0669]* | +0.0342 [+0.0209, +0.0477]* | +0.0413 [+0.0269, +0.0557]* |
| 1 | 701 | 588 | -0.0229 [-0.0298, -0.0162]* | -0.0204 [-0.0276, -0.0129]* | -0.0210 [-0.0274, -0.0145]* | -0.0597 [-0.0696, -0.0504]* | -0.0335 [-0.0427, -0.0240]* | +0.0595 [+0.0491, +0.0701]* | +0.0290 [+0.0223, +0.0356]* | +0.0104 [+0.0042, +0.0168]* | -0.0208 [-0.0289, -0.0127]* |
| 2 | 400 | 354 | -0.0097 [-0.0177, -0.0016]* | -0.0105 [-0.0194, -0.0017]* | -0.0130 [-0.0215, -0.0046]* | -0.0429 [-0.0544, -0.0316]* | +0.0134 [-0.0005, +0.0267] | +0.1104 [+0.0954, +0.1256]* | +0.0204 [+0.0123, +0.0291]* | +0.0061 [-0.0018, +0.0140] | -0.0161 [-0.0272, -0.0051]* |
| 3 | 65 | 64 | +0.0061 [-0.0116, +0.0233] | -0.0144 [-0.0388, +0.0074] | -0.0003 [-0.0215, +0.0206] | -0.0433 [-0.0722, -0.0136]* | -0.0028 [-0.0501, +0.0398] | +0.1173 [+0.0810, +0.1553]* | +0.0409 [+0.0140, +0.0652]* | +0.0176 [-0.0045, +0.0369] | -0.0130 [-0.0422, +0.0157] |
| 4 | n<10 | |  |  |  |  |  |  |  |  |  |

**n_measurement**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=25 | 348 | 328 | -0.0464 [-0.0559, -0.0375]* | -0.0598 [-0.0684, -0.0510]* | -0.0428 [-0.0513, -0.0337]* | -0.0976 [-0.1095, -0.0860]* | -0.0538 [-0.0682, -0.0401]* | +0.0766 [+0.0586, +0.0956]* | +0.0232 [+0.0127, +0.0347]* | -0.0102 [-0.0201, -0.0000]* | -0.0473 [-0.0587, -0.0346]* |
| (25,33] | 336 | 335 | -0.0255 [-0.0338, -0.0174]* | -0.0342 [-0.0427, -0.0258]* | -0.0244 [-0.0321, -0.0166]* | -0.0985 [-0.1101, -0.0869]* | -0.0058 [-0.0187, +0.0074] | +0.0570 [+0.0433, +0.0709]* | +0.0342 [+0.0251, +0.0436]* | +0.0234 [+0.0156, +0.0312]* | -0.0445 [-0.0552, -0.0332]* |
| (33,60] | 339 | 324 | -0.0101 [-0.0187, -0.0007]* | -0.0088 [-0.0176, -0.0001]* | -0.0105 [-0.0194, -0.0015]* | -0.0328 [-0.0438, -0.0216]* | -0.0092 [-0.0231, +0.0049] | +0.0804 [+0.0662, +0.0951]* | +0.0267 [+0.0182, +0.0360]* | +0.0140 [+0.0060, +0.0226]* | -0.0031 [-0.0133, +0.0073] |
| >60 | 322 | 237 | +0.0213 [+0.0116, +0.0308]* | +0.0529 [+0.0431, +0.0619]* | +0.0173 [+0.0073, +0.0270]* | +0.0445 [+0.0341, +0.0539]* | +0.0103 [-0.0031, +0.0231] | +0.1026 [+0.0889, +0.1169]* | +0.0368 [+0.0265, +0.0473]* | +0.0240 [+0.0142, +0.0341]* | +0.0544 [+0.0450, +0.0639]* |

**n_vital**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 16 | +0.0149 [-0.0354, +0.0698]+ | +0.0246 [-0.0139, +0.0653]+ | +0.0346 [-0.0166, +0.0869]+ | +0.0215 [-0.0098, +0.0545]+ | +0.0360 [-0.0277, +0.1025]+ | +0.1671 [+0.0795, +0.2660]*+ | +0.0319 [-0.0331, +0.0989]+ | +0.0434 [-0.0152, +0.1051]+ | +0.0439 [+0.0136, +0.0770]*+ |
| 1 | n<10 | |  |  |  |  |  |  |  |  |  |
| 2 | n<10 | |  |  |  |  |  |  |  |  |  |
| 4 | n<10 | |  |  |  |  |  |  |  |  |  |
| 5 | 24 | 23 | +0.0066 [-0.0214, +0.0392]+ | +0.0050 [-0.0323, +0.0461]+ | -0.0066 [-0.0426, +0.0304]+ | -0.0232 [-0.0734, +0.0278]+ | -0.0264 [-0.0797, +0.0218]+ | +0.1153 [+0.0483, +0.2007]*+ | +0.0543 [+0.0243, +0.0863]*+ | +0.0159 [-0.0170, +0.0510]+ | +0.0019 [-0.0424, +0.0491]+ |
| 6 | 1298 | 990 | -0.0169 [-0.0220, -0.0120]* | -0.0145 [-0.0197, -0.0095]* | -0.0167 [-0.0214, -0.0122]* | -0.0489 [-0.0556, -0.0422]* | -0.0154 [-0.0231, -0.0080]* | +0.0771 [+0.0692, +0.0849]* | +0.0293 [+0.0244, +0.0345]* | +0.0119 [+0.0069, +0.0165]* | -0.0121 [-0.0181, -0.0062]* |

**n_diagnosis**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0252 [-0.0314, -0.0193]* | -0.0407 [-0.0472, -0.0346]* | -0.0259 [-0.0319, -0.0202]* | -0.1065 [-0.1152, -0.0979]* | -0.0174 [-0.0275, -0.0082]* | +0.0653 [+0.0547, +0.0762]* | +0.0327 [+0.0263, +0.0393]* | +0.0171 [+0.0112, +0.0232]* | -0.0550 [-0.0633, -0.0467]* |
| (0,3] | 390 | 334 | -0.0078 [-0.0177, +0.0020] | +0.0026 [-0.0076, +0.0124] | -0.0059 [-0.0155, +0.0039] | +0.0038 [-0.0059, +0.0134] | -0.0136 [-0.0265, +0.0003] | +0.0895 [+0.0750, +0.1049]* | +0.0298 [+0.0202, +0.0399]* | +0.0102 [+0.0006, +0.0197]* | +0.0284 [+0.0194, +0.0378]* |
| >3 | 275 | 204 | -0.0040 [-0.0140, +0.0066] | +0.0307 [+0.0183, +0.0420]* | -0.0044 [-0.0151, +0.0061] | +0.0258 [+0.0130, +0.0371]* | -0.0122 [-0.0282, +0.0027] | +0.0976 [+0.0801, +0.1145]* | +0.0239 [+0.0128, +0.0363]* | +0.0042 [-0.0065, +0.0149] | +0.0413 [+0.0304, +0.0514]* |

**share_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0137 | 340 | 277 | +0.0057 [-0.0043, +0.0155] | +0.0311 [+0.0207, +0.0403]* | +0.0061 [-0.0036, +0.0158] | +0.0195 [+0.0086, +0.0300]* | -0.0144 [-0.0268, -0.0025]* | +0.0728 [+0.0608, +0.0845]* | +0.0463 [+0.0362, +0.0565]* | +0.0302 [+0.0204, +0.0396]* | +0.0495 [+0.0395, +0.0588]* |
| (0.0137,0.027] | 356 | 321 | -0.0115 [-0.0200, -0.0033]* | -0.0002 [-0.0087, +0.0087] | -0.0102 [-0.0183, -0.0016]* | -0.0283 [-0.0394, -0.0174]* | -0.0100 [-0.0211, +0.0010] | +0.0732 [+0.0588, +0.0876]* | +0.0246 [+0.0160, +0.0333]* | +0.0128 [+0.0047, +0.0211]* | +0.0011 [-0.0095, +0.0112] |
| (0.027,0.0444] | 318 | 305 | -0.0259 [-0.0355, -0.0165]* | -0.0355 [-0.0454, -0.0259]* | -0.0262 [-0.0356, -0.0170]* | -0.0905 [-0.1056, -0.0763]* | -0.0031 [-0.0163, +0.0104] | +0.0615 [+0.0476, +0.0777]* | +0.0328 [+0.0235, +0.0425]* | +0.0164 [+0.0074, +0.0254]* | -0.0433 [-0.0551, -0.0317]* |
| >0.0444 | 331 | 316 | -0.0330 [-0.0424, -0.0241]* | -0.0528 [-0.0624, -0.0440]* | -0.0339 [-0.0426, -0.0259]* | -0.0955 [-0.1074, -0.0846]* | -0.0334 [-0.0519, -0.0157]* | +0.1081 [+0.0893, +0.1273]* | +0.0167 [+0.0066, +0.0269]* | -0.0098 [-0.0195, -0.0003]* | -0.0556 [-0.0673, -0.0443]* |

**share_measurement**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.767 | 337 | 317 | -0.0473 [-0.0574, -0.0381]* | -0.0633 [-0.0727, -0.0541]* | -0.0449 [-0.0539, -0.0360]* | -0.0951 [-0.1076, -0.0830]* | -0.0518 [-0.0665, -0.0379]* | +0.0828 [+0.0640, +0.1026]* | +0.0205 [+0.0095, +0.0319]* | -0.0161 [-0.0261, -0.0063]* | -0.0479 [-0.0598, -0.0354]* |
| (0.767,0.814] | 339 | 330 | -0.0304 [-0.0389, -0.0222]* | -0.0354 [-0.0440, -0.0277]* | -0.0299 [-0.0381, -0.0222]* | -0.0990 [-0.1116, -0.0868]* | -0.0063 [-0.0214, +0.0085] | +0.0653 [+0.0513, +0.0791]* | +0.0314 [+0.0227, +0.0408]* | +0.0193 [+0.0114, +0.0269]* | -0.0456 [-0.0560, -0.0345]* |
| (0.814,0.86] | 333 | 313 | -0.0075 [-0.0163, +0.0010] | -0.0053 [-0.0137, +0.0033] | -0.0054 [-0.0138, +0.0029] | -0.0342 [-0.0453, -0.0233]* | -0.0142 [-0.0268, -0.0009]* | +0.0667 [+0.0525, +0.0804]* | +0.0308 [+0.0220, +0.0402]* | +0.0212 [+0.0127, +0.0297]* | -0.0012 [-0.0121, +0.0092] |
| >0.86 | 336 | 256 | +0.0223 [+0.0128, +0.0315]* | +0.0500 [+0.0407, +0.0587]* | +0.0177 [+0.0079, +0.0274]* | +0.0391 [+0.0296, +0.0486]* | +0.0113 [-0.0013, +0.0237] | +0.1008 [+0.0867, +0.1147]* | +0.0376 [+0.0275, +0.0478]* | +0.0256 [+0.0164, +0.0353]* | +0.0506 [+0.0415, +0.0597]* |

**share_vital**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0845 | 341 | 253 | +0.0177 [+0.0077, +0.0276]* | +0.0493 [+0.0399, +0.0584]* | +0.0146 [+0.0046, +0.0245]* | +0.0413 [+0.0309, +0.0511]* | +0.0077 [-0.0061, +0.0204] | +0.1044 [+0.0906, +0.1182]* | +0.0360 [+0.0257, +0.0464]* | +0.0233 [+0.0137, +0.0334]* | +0.0540 [+0.0447, +0.0631]* |
| (0.0845,0.146] | 350 | 334 | -0.0080 [-0.0165, +0.0008] | -0.0103 [-0.0185, -0.0018]* | -0.0100 [-0.0184, -0.0008]* | -0.0361 [-0.0471, -0.0252]* | -0.0102 [-0.0236, +0.0031] | +0.0797 [+0.0656, +0.0945]* | +0.0247 [+0.0163, +0.0333]* | +0.0137 [+0.0056, +0.0219]* | -0.0080 [-0.0186, +0.0021] |
| (0.146,0.182] | 324 | 323 | -0.0276 [-0.0361, -0.0197]* | -0.0355 [-0.0443, -0.0268]* | -0.0262 [-0.0337, -0.0187]* | -0.1021 [-0.1142, -0.0907]* | -0.0037 [-0.0166, +0.0099] | +0.0545 [+0.0407, +0.0688]* | +0.0347 [+0.0258, +0.0439]* | +0.0212 [+0.0134, +0.0291]* | -0.0483 [-0.0589, -0.0372]* |
| >0.182 | 330 | 315 | -0.0471 [-0.0567, -0.0378]* | -0.0606 [-0.0697, -0.0513]* | -0.0428 [-0.0519, -0.0334]* | -0.0977 [-0.1099, -0.0857]* | -0.0556 [-0.0702, -0.0416]* | +0.0757 [+0.0572, +0.0944]* | +0.0250 [+0.0144, +0.0363]* | -0.0085 [-0.0188, +0.0018] | -0.0452 [-0.0573, -0.0324]* |

**share_diagnosis**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0252 [-0.0314, -0.0193]* | -0.0407 [-0.0472, -0.0346]* | -0.0259 [-0.0319, -0.0202]* | -0.1065 [-0.1152, -0.0979]* | -0.0174 [-0.0275, -0.0082]* | +0.0653 [+0.0547, +0.0762]* | +0.0327 [+0.0263, +0.0393]* | +0.0171 [+0.0112, +0.0232]* | -0.0550 [-0.0633, -0.0467]* |
| (0,0.0441] | 329 | 259 | +0.0125 [+0.0031, +0.0217]* | +0.0278 [+0.0173, +0.0378]* | +0.0094 [-0.0006, +0.0191] | +0.0222 [+0.0113, +0.0324]* | -0.0063 [-0.0211, +0.0087] | +0.0852 [+0.0696, +0.1025]* | +0.0340 [+0.0242, +0.0439]* | +0.0208 [+0.0115, +0.0307]* | +0.0373 [+0.0281, +0.0465]* |
| >0.0441 | 336 | 257 | -0.0246 [-0.0351, -0.0138]* | +0.0008 [-0.0101, +0.0114] | -0.0197 [-0.0300, -0.0094]* | +0.0038 [-0.0076, +0.0138] | -0.0196 [-0.0338, -0.0053]* | +0.1003 [+0.0847, +0.1159]* | +0.0209 [+0.0094, +0.0329]* | -0.0050 [-0.0158, +0.0054] | +0.0302 [+0.0195, +0.0402]* |

**n_knowledge**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=21 | 347 | 331 | -0.0394 [-0.0487, -0.0304]* | -0.0522 [-0.0611, -0.0435]* | -0.0361 [-0.0444, -0.0274]* | -0.0982 [-0.1102, -0.0865]* | -0.0510 [-0.0642, -0.0379]* | +0.0741 [+0.0564, +0.0911]* | +0.0241 [+0.0137, +0.0351]* | -0.0025 [-0.0124, +0.0072] | -0.0494 [-0.0614, -0.0370]* |
| (21,25] | 407 | 389 | -0.0197 [-0.0285, -0.0114]* | -0.0260 [-0.0345, -0.0170]* | -0.0169 [-0.0251, -0.0088]* | -0.0761 [-0.0882, -0.0643]* | -0.0010 [-0.0137, +0.0121] | +0.0734 [+0.0604, +0.0869]* | +0.0322 [+0.0241, +0.0406]* | +0.0207 [+0.0125, +0.0288]* | -0.0312 [-0.0413, -0.0209]* |
| (25,30] | 290 | 265 | -0.0056 [-0.0152, +0.0037] | -0.0042 [-0.0142, +0.0056] | -0.0098 [-0.0193, +0.0004] | -0.0276 [-0.0392, -0.0162]* | -0.0046 [-0.0192, +0.0100] | +0.0746 [+0.0593, +0.0916]* | +0.0285 [+0.0188, +0.0386]* | +0.0139 [+0.0045, +0.0230]* | +0.0013 [-0.0096, +0.0119] |
| >30 | 301 | 218 | +0.0068 [-0.0031, +0.0163] | +0.0387 [+0.0280, +0.0486]* | +0.0037 [-0.0060, +0.0134] | +0.0305 [+0.0192, +0.0409]* | -0.0034 [-0.0182, +0.0093] | +0.0961 [+0.0806, +0.1112]* | +0.0357 [+0.0248, +0.0460]* | +0.0174 [+0.0077, +0.0270]* | +0.0482 [+0.0386, +0.0573]* |

**n_analyte**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=23 | 367 | 328 | -0.0455 [-0.0545, -0.0366]* | -0.0518 [-0.0614, -0.0419]* | -0.0413 [-0.0497, -0.0326]* | -0.0776 [-0.0894, -0.0651]* | -0.0598 [-0.0740, -0.0453]* | +0.0803 [+0.0631, +0.0983]* | +0.0229 [+0.0126, +0.0335]* | -0.0131 [-0.0225, -0.0035]* | -0.0322 [-0.0440, -0.0203]* |
| (23,29] | 388 | 344 | -0.0140 [-0.0223, -0.0055]* | -0.0160 [-0.0244, -0.0071]* | -0.0164 [-0.0249, -0.0083]* | -0.0606 [-0.0725, -0.0480]* | +0.0013 [-0.0112, +0.0138] | +0.0663 [+0.0529, +0.0803]* | +0.0360 [+0.0275, +0.0452]* | +0.0245 [+0.0169, +0.0325]* | -0.0169 [-0.0274, -0.0062]* |
| (29,33] | 294 | 274 | -0.0092 [-0.0199, +0.0010] | +0.0002 [-0.0087, +0.0097] | -0.0061 [-0.0164, +0.0037] | -0.0305 [-0.0428, -0.0181]* | -0.0074 [-0.0234, +0.0075] | +0.0744 [+0.0589, +0.0897]* | +0.0251 [+0.0149, +0.0349]* | +0.0164 [+0.0066, +0.0261]* | -0.0016 [-0.0129, +0.0101] |
| >33 | 296 | 258 | +0.0119 [+0.0026, +0.0211]* | +0.0231 [+0.0122, +0.0339]* | +0.0074 [-0.0020, +0.0171] | -0.0098 [-0.0234, +0.0036] | +0.0106 [-0.0022, +0.0238] | +0.0981 [+0.0831, +0.1139]* | +0.0362 [+0.0256, +0.0464]* | +0.0245 [+0.0149, +0.0341]* | +0.0131 [+0.0000, +0.0254]* |

**n_prior_dx**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0252 [-0.0314, -0.0193]* | -0.0407 [-0.0472, -0.0346]* | -0.0259 [-0.0319, -0.0202]* | -0.1065 [-0.1152, -0.0979]* | -0.0174 [-0.0275, -0.0082]* | +0.0653 [+0.0547, +0.0762]* | +0.0327 [+0.0263, +0.0393]* | +0.0171 [+0.0112, +0.0232]* | -0.0550 [-0.0633, -0.0467]* |
| (0,3] | 390 | 334 | -0.0078 [-0.0177, +0.0020] | +0.0026 [-0.0076, +0.0124] | -0.0059 [-0.0155, +0.0039] | +0.0038 [-0.0059, +0.0134] | -0.0136 [-0.0265, +0.0003] | +0.0895 [+0.0750, +0.1049]* | +0.0298 [+0.0202, +0.0399]* | +0.0102 [+0.0006, +0.0197]* | +0.0284 [+0.0194, +0.0378]* |
| >3 | 275 | 204 | -0.0040 [-0.0140, +0.0066] | +0.0307 [+0.0183, +0.0420]* | -0.0044 [-0.0151, +0.0061] | +0.0258 [+0.0130, +0.0371]* | -0.0122 [-0.0282, +0.0027] | +0.0976 [+0.0801, +0.1145]* | +0.0239 [+0.0128, +0.0363]* | +0.0042 [-0.0065, +0.0149] | +0.0413 [+0.0304, +0.0514]* |

**rel_trajectory_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 1039 | 907 | -0.0242 [-0.0294, -0.0193]* | -0.0307 [-0.0362, -0.0256]* | -0.0230 [-0.0279, -0.0182]* | -0.0722 [-0.0793, -0.0653]* | -0.0192 [-0.0273, -0.0114]* | +0.0735 [+0.0647, +0.0828]* | +0.0289 [+0.0237, +0.0344]* | +0.0115 [+0.0064, +0.0164]* | -0.0288 [-0.0352, -0.0221]* |
| >0 | 306 | 229 | +0.0127 [+0.0017, +0.0241]* | +0.0445 [+0.0330, +0.0549]* | +0.0091 [-0.0023, +0.0202] | +0.0366 [+0.0255, +0.0472]* | -0.0017 [-0.0176, +0.0134] | +0.0973 [+0.0822, +0.1113]* | +0.0342 [+0.0223, +0.0465]* | +0.0160 [+0.0045, +0.0276]* | +0.0488 [+0.0389, +0.0585]* |

**rel_recurrence_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0252 [-0.0314, -0.0193]* | -0.0407 [-0.0472, -0.0346]* | -0.0259 [-0.0319, -0.0202]* | -0.1065 [-0.1152, -0.0979]* | -0.0174 [-0.0275, -0.0082]* | +0.0653 [+0.0547, +0.0762]* | +0.0327 [+0.0263, +0.0393]* | +0.0171 [+0.0112, +0.0232]* | -0.0550 [-0.0633, -0.0467]* |
| (0,3] | 390 | 334 | -0.0078 [-0.0177, +0.0020] | +0.0026 [-0.0076, +0.0124] | -0.0059 [-0.0155, +0.0039] | +0.0038 [-0.0059, +0.0134] | -0.0136 [-0.0265, +0.0003] | +0.0895 [+0.0750, +0.1049]* | +0.0298 [+0.0202, +0.0399]* | +0.0102 [+0.0006, +0.0197]* | +0.0284 [+0.0194, +0.0378]* |
| >3 | 275 | 204 | -0.0040 [-0.0140, +0.0066] | +0.0307 [+0.0183, +0.0420]* | -0.0044 [-0.0151, +0.0061] | +0.0258 [+0.0130, +0.0371]* | -0.0122 [-0.0282, +0.0027] | +0.0976 [+0.0801, +0.1145]* | +0.0239 [+0.0128, +0.0363]* | +0.0042 [-0.0065, +0.0149] | +0.0413 [+0.0304, +0.0514]* |

**rel_comorbid_with**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 862 | 783 | -0.0202 [-0.0261, -0.0144]* | -0.0307 [-0.0368, -0.0245]* | -0.0211 [-0.0268, -0.0155]* | -0.0831 [-0.0913, -0.0749]* | -0.0163 [-0.0255, -0.0076]* | +0.0714 [+0.0618, +0.0815]* | +0.0331 [+0.0274, +0.0392]* | +0.0157 [+0.0101, +0.0214]* | -0.0376 [-0.0451, -0.0297]* |
| (0,2] | 159 | 144 | -0.0213 [-0.0363, -0.0057]* | -0.0035 [-0.0187, +0.0116] | -0.0155 [-0.0286, -0.0014]* | +0.0020 [-0.0125, +0.0160] | -0.0307 [-0.0502, -0.0097]* | +0.0893 [+0.0682, +0.1119]* | +0.0198 [+0.0051, +0.0354]* | +0.0024 [-0.0119, +0.0172] | +0.0291 [+0.0162, +0.0415]* |
| >2 | 324 | 229 | -0.0016 [-0.0110, +0.0082] | +0.0269 [+0.0159, +0.0372]* | -0.0013 [-0.0109, +0.0081] | +0.0231 [+0.0114, +0.0331]* | -0.0047 [-0.0198, +0.0092] | +0.0938 [+0.0781, +0.1092]* | +0.0271 [+0.0166, +0.0385]* | +0.0089 [-0.0011, +0.0194] | +0.0397 [+0.0294, +0.0490]* |

**rel_co_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 873 | 697 | -0.0202 [-0.0262, -0.0141]* | -0.0148 [-0.0214, -0.0084]* | -0.0179 [-0.0237, -0.0121]* | -0.0496 [-0.0584, -0.0417]* | -0.0296 [-0.0378, -0.0215]* | +0.0608 [+0.0520, +0.0698]* | +0.0335 [+0.0274, +0.0398]* | +0.0151 [+0.0092, +0.0210]* | -0.0085 [-0.0156, -0.0012]* |
| 2 | 400 | 354 | -0.0097 [-0.0177, -0.0016]* | -0.0105 [-0.0194, -0.0017]* | -0.0130 [-0.0215, -0.0046]* | -0.0429 [-0.0544, -0.0316]* | +0.0134 [-0.0005, +0.0267] | +0.1104 [+0.0954, +0.1256]* | +0.0204 [+0.0123, +0.0291]* | +0.0061 [-0.0018, +0.0140] | -0.0161 [-0.0272, -0.0051]* |
| 6 | 65 | 64 | +0.0061 [-0.0116, +0.0233] | -0.0144 [-0.0388, +0.0074] | -0.0003 [-0.0215, +0.0206] | -0.0433 [-0.0722, -0.0136]* | -0.0028 [-0.0501, +0.0398] | +0.1173 [+0.0810, +0.1553]* | +0.0409 [+0.0140, +0.0652]* | +0.0176 [-0.0045, +0.0369] | -0.0130 [-0.0422, +0.0157] |
| 12 | n<10 | |  |  |  |  |  |  |  |  |  |

**rel_baseline_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 768 | 755 | -0.0289 [-0.0348, -0.0232]* | -0.0428 [-0.0489, -0.0369]* | -0.0282 [-0.0338, -0.0225]* | -0.0988 [-0.1068, -0.0907]* | -0.0192 [-0.0286, -0.0103]* | +0.0675 [+0.0570, +0.0783]* | +0.0310 [+0.0251, +0.0373]* | +0.0135 [+0.0078, +0.0191]* | -0.0474 [-0.0549, -0.0398]* |
| (0,25] | 287 | 242 | -0.0142 [-0.0249, -0.0036]* | +0.0025 [-0.0081, +0.0124] | -0.0126 [-0.0243, -0.0024]* | +0.0002 [-0.0103, +0.0100] | -0.0290 [-0.0450, -0.0120]* | +0.0873 [+0.0706, +0.1053]* | +0.0242 [+0.0124, +0.0358]* | +0.0010 [-0.0105, +0.0117] | +0.0224 [+0.0116, +0.0326]* |
| >25 | 290 | 233 | +0.0172 [+0.0068, +0.0276]* | +0.0478 [+0.0375, +0.0574]* | +0.0143 [+0.0038, +0.0242]* | +0.0412 [+0.0302, +0.0515]* | +0.0090 [-0.0053, +0.0219] | +0.1007 [+0.0861, +0.1159]* | +0.0334 [+0.0226, +0.0445]* | +0.0211 [+0.0100, +0.0313]* | +0.0517 [+0.0415, +0.0615]* |

**rel_medical_total**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=63 | 353 | 327 | -0.0447 [-0.0541, -0.0355]* | -0.0547 [-0.0639, -0.0455]* | -0.0393 [-0.0478, -0.0302]* | -0.0860 [-0.0977, -0.0743]* | -0.0624 [-0.0765, -0.0482]* | +0.0795 [+0.0619, +0.0978]* | +0.0238 [+0.0129, +0.0353]* | -0.0099 [-0.0197, +0.0006] | -0.0388 [-0.0506, -0.0263]* |
| (63,75] | 352 | 327 | -0.0176 [-0.0258, -0.0091]* | -0.0214 [-0.0306, -0.0123]* | -0.0180 [-0.0262, -0.0102]* | -0.0721 [-0.0852, -0.0595]* | -0.0007 [-0.0134, +0.0117] | +0.0632 [+0.0492, +0.0779]* | +0.0366 [+0.0287, +0.0453]* | +0.0224 [+0.0149, +0.0304]* | -0.0274 [-0.0385, -0.0160]* |
| (75,87] | 320 | 291 | -0.0084 [-0.0178, +0.0001] | -0.0057 [-0.0155, +0.0030] | -0.0095 [-0.0189, -0.0006]* | -0.0404 [-0.0536, -0.0288]* | -0.0053 [-0.0203, +0.0088] | +0.0719 [+0.0564, +0.0873]* | +0.0226 [+0.0137, +0.0318]* | +0.0152 [+0.0063, +0.0241]* | -0.0108 [-0.0219, +0.0000] |
| >87 | 320 | 251 | +0.0106 [+0.0013, +0.0198]* | +0.0324 [+0.0220, +0.0425]* | +0.0068 [-0.0029, +0.0163] | +0.0151 [+0.0028, +0.0268]* | +0.0108 [-0.0025, +0.0233] | +0.1025 [+0.0889, +0.1163]* | +0.0373 [+0.0268, +0.0480]* | +0.0235 [+0.0139, +0.0332]* | +0.0371 [+0.0258, +0.0477]* |

**rel_medical_member_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=21 | 354 | 317 | -0.0458 [-0.0550, -0.0369]* | -0.0511 [-0.0609, -0.0412]* | -0.0412 [-0.0499, -0.0323]* | -0.0752 [-0.0873, -0.0626]* | -0.0620 [-0.0763, -0.0469]* | +0.0801 [+0.0624, +0.0983]* | +0.0235 [+0.0131, +0.0345]* | -0.0131 [-0.0226, -0.0032]* | -0.0292 [-0.0413, -0.0171]* |
| (21,28] | 483 | 424 | -0.0151 [-0.0222, -0.0078]* | -0.0141 [-0.0213, -0.0066]* | -0.0156 [-0.0230, -0.0084]* | -0.0543 [-0.0645, -0.0432]* | -0.0010 [-0.0119, +0.0106] | +0.0663 [+0.0543, +0.0784]* | +0.0332 [+0.0254, +0.0417]* | +0.0214 [+0.0144, +0.0288]* | -0.0135 [-0.0225, -0.0042]* |
| (28,31] | 188 | 180 | -0.0043 [-0.0170, +0.0075] | +0.0029 [-0.0082, +0.0142] | -0.0008 [-0.0125, +0.0100] | -0.0338 [-0.0489, -0.0191]* | -0.0095 [-0.0289, +0.0084] | +0.0713 [+0.0521, +0.0901]* | +0.0256 [+0.0131, +0.0376]* | +0.0194 [+0.0074, +0.0307]* | -0.0070 [-0.0215, +0.0072] |
| >31 | 320 | 280 | +0.0094 [+0.0001, +0.0187]* | +0.0190 [+0.0082, +0.0296]* | +0.0037 [-0.0061, +0.0134] | -0.0144 [-0.0277, -0.0011]* | +0.0116 [-0.0010, +0.0250] | +0.1010 [+0.0868, +0.1158]* | +0.0352 [+0.0249, +0.0448]* | +0.0233 [+0.0137, +0.0328]* | +0.0100 [-0.0025, +0.0221] |

**rel_medical_assesses**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=12 | 457 | 400 | -0.0327 [-0.0414, -0.0237]* | -0.0403 [-0.0486, -0.0321]* | -0.0319 [-0.0402, -0.0233]* | -0.0714 [-0.0823, -0.0608]* | -0.0468 [-0.0591, -0.0336]* | +0.0792 [+0.0642, +0.0951]* | +0.0247 [+0.0149, +0.0345]* | -0.0017 [-0.0111, +0.0074] | -0.0282 [-0.0382, -0.0175]* |
| (12,15] | 459 | 401 | -0.0211 [-0.0284, -0.0134]* | -0.0160 [-0.0238, -0.0079]* | -0.0189 [-0.0262, -0.0114]* | -0.0542 [-0.0650, -0.0435]* | -0.0063 [-0.0191, +0.0062] | +0.0714 [+0.0588, +0.0840]* | +0.0292 [+0.0219, +0.0370]* | +0.0171 [+0.0100, +0.0240]* | -0.0137 [-0.0228, -0.0046]* |
| (15,16] | 142 | 133 | -0.0085 [-0.0212, +0.0036] | +0.0019 [-0.0111, +0.0153] | -0.0059 [-0.0183, +0.0067] | -0.0273 [-0.0449, -0.0092]* | +0.0062 [-0.0132, +0.0251] | +0.0741 [+0.0523, +0.0959]* | +0.0249 [+0.0106, +0.0390]* | +0.0139 [+0.0003, +0.0272]* | +0.0013 [-0.0166, +0.0191] |
| >16 | 287 | 255 | +0.0158 [+0.0065, +0.0249]* | +0.0251 [+0.0140, +0.0359]* | +0.0104 [+0.0012, +0.0196]* | -0.0086 [-0.0226, +0.0051] | +0.0101 [-0.0030, +0.0243] | +0.0927 [+0.0774, +0.1087]* | +0.0427 [+0.0318, +0.0530]* | +0.0270 [+0.0173, +0.0362]* | +0.0140 [+0.0003, +0.0263]* |

**rel_medical_measures**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=22 | 382 | 335 | -0.0412 [-0.0505, -0.0320]* | -0.0452 [-0.0547, -0.0355]* | -0.0368 [-0.0455, -0.0277]* | -0.0693 [-0.0810, -0.0573]* | -0.0566 [-0.0707, -0.0412]* | +0.0808 [+0.0642, +0.0974]* | +0.0263 [+0.0160, +0.0369]* | -0.0108 [-0.0202, -0.0012]* | -0.0258 [-0.0373, -0.0143]* |
| (22,26] | 525 | 457 | -0.0163 [-0.0236, -0.0088]* | -0.0142 [-0.0216, -0.0068]* | -0.0160 [-0.0233, -0.0090]* | -0.0546 [-0.0645, -0.0446]* | -0.0014 [-0.0118, +0.0096] | +0.0700 [+0.0586, +0.0823]* | +0.0293 [+0.0217, +0.0371]* | +0.0190 [+0.0118, +0.0263]* | -0.0160 [-0.0247, -0.0073]* |
| (26,28] | 152 | 146 | +0.0067 [-0.0066, +0.0198] | +0.0151 [+0.0018, +0.0293]* | +0.0070 [-0.0055, +0.0192] | -0.0183 [-0.0370, -0.0004]* | +0.0027 [-0.0196, +0.0247] | +0.0819 [+0.0594, +0.1047]* | +0.0400 [+0.0263, +0.0540]* | +0.0227 [+0.0100, +0.0356]* | +0.0032 [-0.0129, +0.0189] |
| >28 | 286 | 246 | +0.0069 [-0.0025, +0.0163] | +0.0145 [+0.0038, +0.0247]* | +0.0011 [-0.0087, +0.0108] | -0.0206 [-0.0349, -0.0074]* | +0.0051 [-0.0075, +0.0176] | +0.0911 [+0.0757, +0.1062]* | +0.0313 [+0.0207, +0.0411]* | +0.0262 [+0.0168, +0.0356]* | +0.0099 [-0.0032, +0.0229] |

**rel_medical_is_a**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=1 | 506 | 506 | -0.0294 [-0.0366, -0.0227]* | -0.0418 [-0.0491, -0.0348]* | -0.0288 [-0.0354, -0.0219]* | -0.1086 [-0.1185, -0.0987]* | -0.0306 [-0.0410, -0.0208]* | +0.0472 [+0.0361, +0.0580]* | +0.0327 [+0.0252, +0.0405]* | +0.0166 [+0.0094, +0.0235]* | -0.0540 [-0.0632, -0.0445]* |
| (1,3] | 189 | 188 | -0.0124 [-0.0242, -0.0006]* | -0.0333 [-0.0469, -0.0204]* | -0.0163 [-0.0280, -0.0055]* | -0.0912 [-0.1077, -0.0748]* | +0.0182 [-0.0043, +0.0402] | +0.1183 [+0.0958, +0.1422]* | +0.0348 [+0.0230, +0.0474]* | +0.0188 [+0.0080, +0.0292]* | -0.0489 [-0.0639, -0.0340]* |
| (3,10] | 349 | 307 | -0.0104 [-0.0210, +0.0003] | -0.0012 [-0.0113, +0.0092] | -0.0083 [-0.0183, +0.0022] | +0.0005 [-0.0094, +0.0104] | -0.0184 [-0.0319, -0.0041]* | +0.0876 [+0.0728, +0.1034]* | +0.0292 [+0.0188, +0.0395]* | +0.0095 [-0.0008, +0.0198] | +0.0265 [+0.0169, +0.0363]* |
| >10 | 301 | 223 | -0.0015 [-0.0111, +0.0083] | +0.0319 [+0.0203, +0.0427]* | -0.0018 [-0.0119, +0.0081] | +0.0271 [+0.0154, +0.0379]* | -0.0067 [-0.0220, +0.0073] | +0.0975 [+0.0813, +0.1141]* | +0.0237 [+0.0127, +0.0351]* | +0.0051 [-0.0052, +0.0152] | +0.0411 [+0.0308, +0.0509]* |

**rel_medical_may_affect**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 623 | 519 | -0.0134 [-0.0203, -0.0064]* | -0.0050 [-0.0124, +0.0018] | -0.0115 [-0.0184, -0.0050]* | -0.0339 [-0.0427, -0.0252]* | -0.0308 [-0.0415, -0.0200]* | +0.0865 [+0.0753, +0.0973]* | +0.0281 [+0.0208, +0.0357]* | +0.0092 [+0.0018, +0.0165]* | -0.0021 [-0.0112, +0.0065] |
| 1 | 479 | 415 | -0.0159 [-0.0237, -0.0079]* | -0.0241 [-0.0332, -0.0143]* | -0.0190 [-0.0268, -0.0113]* | -0.0630 [-0.0753, -0.0511]* | -0.0134 [-0.0251, -0.0014]* | +0.0815 [+0.0693, +0.0952]* | +0.0342 [+0.0259, +0.0424]* | +0.0157 [+0.0082, +0.0233]* | -0.0243 [-0.0346, -0.0146]* |
| 2 | 172 | 162 | -0.0191 [-0.0307, -0.0072]* | -0.0136 [-0.0251, -0.0015]* | -0.0167 [-0.0291, -0.0048]* | -0.0488 [-0.0654, -0.0322]* | +0.0267 [+0.0095, +0.0432]* | +0.0623 [+0.0440, +0.0808]* | +0.0232 [+0.0110, +0.0361]* | +0.0124 [+0.0021, +0.0234]* | -0.0118 [-0.0271, +0.0034] |
| 3 | 54 | 54 | -0.0283 [-0.0486, -0.0079]* | -0.0230 [-0.0434, -0.0019]* | -0.0302 [-0.0514, -0.0089]* | -0.0677 [-0.0925, -0.0418]* | +0.0120 [-0.0238, +0.0465] | +0.0301 [+0.0009, +0.0592]* | +0.0306 [+0.0088, +0.0527]* | +0.0177 [-0.0052, +0.0385] | -0.0118 [-0.0349, +0.0111] |
| 4 | 11 | 11 | -0.0194 [-0.0738, +0.0350]+ | -0.0160 [-0.0661, +0.0345]+ | -0.0069 [-0.0492, +0.0347]+ | -0.0521 [-0.1181, +0.0050]+ | -0.0105 [-0.1065, +0.0729]+ | +0.0183 [-0.0363, +0.0693]+ | +0.0714 [+0.0288, +0.1103]*+ | +0.0390 [+0.0032, +0.0762]*+ | +0.0127 [-0.0288, +0.0502]+ |
| 5 | n<10 | |  |  |  |  |  |  |  |  |  |

**rel_medical_indicates**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 494 | 440 | -0.0303 [-0.0383, -0.0227]* | -0.0265 [-0.0358, -0.0177]* | -0.0265 [-0.0340, -0.0189]* | -0.0691 [-0.0810, -0.0576]* | -0.0368 [-0.0501, -0.0245]* | +0.0679 [+0.0546, +0.0811]* | +0.0221 [+0.0141, +0.0309]* | +0.0032 [-0.0046, +0.0108] | -0.0268 [-0.0379, -0.0163]* |
| (0,1] | 272 | 254 | -0.0119 [-0.0223, -0.0016]* | -0.0155 [-0.0259, -0.0056]* | -0.0120 [-0.0216, -0.0025]* | -0.0520 [-0.0655, -0.0384]* | -0.0057 [-0.0193, +0.0080] | +0.0732 [+0.0566, +0.0910]* | +0.0322 [+0.0214, +0.0441]* | +0.0211 [+0.0104, +0.0322]* | -0.0157 [-0.0284, -0.0029]* |
| (1,3] | 267 | 240 | -0.0086 [-0.0190, +0.0019] | -0.0090 [-0.0195, +0.0016] | -0.0079 [-0.0187, +0.0028] | -0.0396 [-0.0536, -0.0265]* | +0.0037 [-0.0126, +0.0211] | +0.0833 [+0.0667, +0.1006]* | +0.0291 [+0.0183, +0.0396]* | +0.0177 [+0.0073, +0.0278]* | -0.0063 [-0.0182, +0.0058] |
| >3 | 312 | 243 | -0.0025 [-0.0122, +0.0071] | +0.0046 [-0.0064, +0.0146] | -0.0084 [-0.0181, +0.0013] | -0.0160 [-0.0277, -0.0037]* | -0.0056 [-0.0182, +0.0068] | +0.0975 [+0.0824, +0.1132]* | +0.0416 [+0.0317, +0.0517]* | +0.0152 [+0.0058, +0.0243]* | +0.0136 [+0.0020, +0.0249]* |

### 6.4 cei_R1_ls0, confidence-adjusted

**n_nodes**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=81 | 339 | 317 | -0.0281 [-0.0363, -0.0201]* | -0.0271 [-0.0353, -0.0190]* | -0.0228 [-0.0301, -0.0153]* | -0.0545 [-0.0651, -0.0439]* | -0.0346 [-0.0468, -0.0224]* | +0.0686 [+0.0543, +0.0828]* | +0.0322 [+0.0228, +0.0423]* | +0.0064 [-0.0021, +0.0152] | -0.0093 [-0.0195, +0.0005] |
| (81,99] | 338 | 336 | -0.0242 [-0.0322, -0.0160]* | -0.0262 [-0.0340, -0.0183]* | -0.0221 [-0.0296, -0.0149]* | -0.0874 [-0.0972, -0.0767]* | +0.0040 [-0.0084, +0.0160] | +0.0623 [+0.0488, +0.0767]* | +0.0348 [+0.0263, +0.0433]* | +0.0236 [+0.0160, +0.0311]* | -0.0355 [-0.0449, -0.0257]* |
| (99,134] | 345 | 327 | -0.0073 [-0.0149, +0.0007] | -0.0097 [-0.0176, -0.0017]* | -0.0106 [-0.0183, -0.0023]* | -0.0372 [-0.0476, -0.0269]* | -0.0139 [-0.0275, -0.0002]* | +0.0843 [+0.0701, +0.0990]* | +0.0280 [+0.0202, +0.0364]* | +0.0142 [+0.0069, +0.0216]* | -0.0083 [-0.0178, +0.0008] |
| >134 | 323 | 240 | -0.0033 [-0.0119, +0.0051] | +0.0096 [+0.0013, +0.0168]* | -0.0069 [-0.0160, +0.0015] | -0.0093 [-0.0186, -0.0006]* | -0.0164 [-0.0289, -0.0045]* | +0.1013 [+0.0871, +0.1148]* | +0.0252 [+0.0160, +0.0348]* | +0.0055 [-0.0031, +0.0141] | +0.0096 [+0.0011, +0.0177]* |

**n_evidence**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=32 | 337 | 320 | -0.0276 [-0.0362, -0.0196]* | -0.0291 [-0.0376, -0.0207]* | -0.0227 [-0.0300, -0.0147]* | -0.0572 [-0.0676, -0.0465]* | -0.0335 [-0.0457, -0.0212]* | +0.0666 [+0.0520, +0.0812]* | +0.0342 [+0.0248, +0.0440]* | +0.0081 [-0.0002, +0.0172] | -0.0108 [-0.0209, -0.0006]* |
| (32,41] | 349 | 348 | -0.0238 [-0.0315, -0.0161]* | -0.0254 [-0.0331, -0.0176]* | -0.0221 [-0.0295, -0.0149]* | -0.0881 [-0.0983, -0.0778]* | +0.0002 [-0.0118, +0.0122] | +0.0607 [+0.0470, +0.0744]* | +0.0326 [+0.0243, +0.0411]* | +0.0221 [+0.0146, +0.0294]* | -0.0376 [-0.0467, -0.0280]* |
| (41,70] | 332 | 316 | -0.0107 [-0.0187, -0.0023]* | -0.0102 [-0.0181, -0.0023]* | -0.0127 [-0.0207, -0.0041]* | -0.0349 [-0.0450, -0.0251]* | -0.0123 [-0.0260, +0.0010] | +0.0888 [+0.0740, +0.1044]* | +0.0268 [+0.0184, +0.0356]* | +0.0126 [+0.0048, +0.0202]* | -0.0053 [-0.0147, +0.0042] |
| >70 | 327 | 243 | -0.0004 [-0.0091, +0.0076] | +0.0117 [+0.0034, +0.0192]* | -0.0046 [-0.0138, +0.0040] | -0.0068 [-0.0163, +0.0020] | -0.0159 [-0.0293, -0.0033]* | +0.1009 [+0.0872, +0.1145]* | +0.0265 [+0.0174, +0.0358]* | +0.0066 [-0.0023, +0.0152] | +0.0108 [+0.0024, +0.0191]* |

**n_edges**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=125 | 338 | 319 | -0.0284 [-0.0369, -0.0205]* | -0.0283 [-0.0366, -0.0201]* | -0.0237 [-0.0308, -0.0157]* | -0.0564 [-0.0670, -0.0458]* | -0.0337 [-0.0465, -0.0204]* | +0.0659 [+0.0517, +0.0806]* | +0.0330 [+0.0234, +0.0430]* | +0.0071 [-0.0017, +0.0164] | -0.0102 [-0.0202, -0.0004]* |
| (125,157] | 338 | 335 | -0.0245 [-0.0324, -0.0165]* | -0.0249 [-0.0330, -0.0169]* | -0.0222 [-0.0297, -0.0148]* | -0.0862 [-0.0965, -0.0756]* | +0.0044 [-0.0077, +0.0162] | +0.0614 [+0.0478, +0.0750]* | +0.0333 [+0.0251, +0.0422]* | +0.0215 [+0.0140, +0.0289]* | -0.0356 [-0.0449, -0.0260]* |
| (157,225] | 333 | 315 | -0.0075 [-0.0156, +0.0008] | -0.0101 [-0.0182, -0.0020]* | -0.0107 [-0.0188, -0.0024]* | -0.0379 [-0.0484, -0.0278]* | -0.0159 [-0.0296, -0.0017]* | +0.0887 [+0.0736, +0.1040]* | +0.0286 [+0.0199, +0.0376]* | +0.0151 [+0.0072, +0.0229]* | -0.0088 [-0.0186, +0.0005] |
| >225 | 336 | 249 | -0.0027 [-0.0113, +0.0055] | +0.0091 [+0.0012, +0.0163]* | -0.0061 [-0.0149, +0.0021] | -0.0090 [-0.0181, -0.0005]* | -0.0157 [-0.0281, -0.0039]* | +0.0997 [+0.0865, +0.1132]* | +0.0254 [+0.0166, +0.0345]* | +0.0062 [-0.0026, +0.0148] | +0.0104 [+0.0021, +0.0184]* |

**density**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0127 | 337 | 251 | +0.0004 [-0.0078, +0.0082] | +0.0110 [+0.0031, +0.0179]* | -0.0042 [-0.0124, +0.0035] | -0.0089 [-0.0177, -0.0005]* | -0.0185 [-0.0312, -0.0061]* | +0.1008 [+0.0870, +0.1149]* | +0.0285 [+0.0196, +0.0376]* | +0.0077 [-0.0007, +0.0160] | +0.0093 [+0.0006, +0.0174]* |
| (0.0127,0.016] | 336 | 320 | -0.0131 [-0.0215, -0.0050]* | -0.0115 [-0.0202, -0.0027]* | -0.0137 [-0.0220, -0.0051]* | -0.0398 [-0.0509, -0.0294]* | -0.0118 [-0.0241, +0.0008] | +0.0810 [+0.0668, +0.0967]* | +0.0242 [+0.0162, +0.0326]* | +0.0115 [+0.0041, +0.0188]* | -0.0096 [-0.0194, -0.0003]* |
| (0.016,0.0191] | 337 | 334 | -0.0213 [-0.0292, -0.0133]* | -0.0265 [-0.0339, -0.0185]* | -0.0202 [-0.0279, -0.0129]* | -0.0873 [-0.0973, -0.0765]* | +0.0025 [-0.0101, +0.0145] | +0.0633 [+0.0494, +0.0769]* | +0.0360 [+0.0281, +0.0445]* | +0.0264 [+0.0185, +0.0337]* | -0.0349 [-0.0440, -0.0256]* |
| >0.0191 | 335 | 312 | -0.0294 [-0.0376, -0.0215]* | -0.0274 [-0.0355, -0.0194]* | -0.0246 [-0.0321, -0.0168]* | -0.0538 [-0.0642, -0.0429]* | -0.0333 [-0.0455, -0.0206]* | +0.0705 [+0.0558, +0.0853]* | +0.0317 [+0.0221, +0.0416]* | +0.0043 [-0.0040, +0.0132] | -0.0092 [-0.0194, +0.0012] |

**num_visits**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 679 | 678 | -0.0205 [-0.0264, -0.0145]* | -0.0261 [-0.0318, -0.0202]* | -0.0209 [-0.0264, -0.0152]* | -0.0881 [-0.0956, -0.0799]* | -0.0088 [-0.0182, +0.0002] | +0.0637 [+0.0543, +0.0732]* | +0.0341 [+0.0281, +0.0404]* | +0.0204 [+0.0149, +0.0262]* | -0.0394 [-0.0464, -0.0319]* |
| 2 | 323 | 323 | -0.0157 [-0.0241, -0.0072]* | -0.0099 [-0.0182, -0.0020]* | -0.0130 [-0.0212, -0.0041]* | -0.0074 [-0.0168, +0.0014] | -0.0224 [-0.0347, -0.0099]* | +0.0849 [+0.0700, +0.1009]* | +0.0257 [+0.0168, +0.0351]* | +0.0080 [-0.0006, +0.0168] | +0.0224 [+0.0128, +0.0311]* |
| 3 | 191 | 191 | -0.0120 [-0.0233, -0.0011]* | +0.0061 [-0.0052, +0.0165] | -0.0129 [-0.0243, -0.0017]* | -0.0028 [-0.0139, +0.0077] | -0.0288 [-0.0476, -0.0103]* | +0.0934 [+0.0748, +0.1127]* | +0.0213 [+0.0085, +0.0346]* | -0.0042 [-0.0156, +0.0072] | +0.0165 [+0.0048, +0.0274]* |
| 4 | 91 | 91 | +0.0010 [-0.0167, +0.0181] | +0.0113 [-0.0034, +0.0260] | -0.0040 [-0.0224, +0.0140] | -0.0085 [-0.0268, +0.0075] | -0.0095 [-0.0341, +0.0130] | +0.1188 [+0.0956, +0.1421]* | +0.0367 [+0.0183, +0.0551]* | +0.0083 [-0.0093, +0.0254] | +0.0125 [-0.0022, +0.0267] |
| 5 | 47 | 47 | -0.0005 [-0.0178, +0.0172] | +0.0110 [-0.0060, +0.0279] | +0.0038 [-0.0116, +0.0203] | -0.0020 [-0.0210, +0.0160] | -0.0156 [-0.0420, +0.0122] | +0.1165 [+0.0780, +0.1592]* | +0.0204 [-0.0007, +0.0431] | +0.0023 [-0.0192, +0.0238] | +0.0046 [-0.0164, +0.0253] |
| 6 | 14 | 14 | -0.0062 [-0.0467, +0.0452]+ | -0.0032 [-0.0481, +0.0394]+ | -0.0040 [-0.0509, +0.0465]+ | -0.0196 [-0.0637, +0.0219]+ | -0.0138 [-0.0572, +0.0238]+ | +0.0943 [+0.0447, +0.1373]*+ | +0.0455 [+0.0030, +0.0865]*+ | +0.0186 [-0.0204, +0.0570]+ | +0.0036 [-0.0267, +0.0307]+ |

**n_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 172 | 155 | -0.0215 [-0.0319, -0.0110]* | -0.0209 [-0.0317, -0.0103]* | -0.0199 [-0.0299, -0.0096]* | -0.0571 [-0.0682, -0.0453]* | -0.0303 [-0.0449, -0.0162]* | +0.0760 [+0.0601, +0.0927]* | +0.0447 [+0.0321, +0.0585]* | +0.0207 [+0.0087, +0.0329]* | -0.0045 [-0.0166, +0.0078] |
| 1 | 701 | 588 | -0.0207 [-0.0272, -0.0145]* | -0.0179 [-0.0244, -0.0114]* | -0.0182 [-0.0240, -0.0124]* | -0.0536 [-0.0621, -0.0453]* | -0.0270 [-0.0356, -0.0183]* | +0.0584 [+0.0488, +0.0686]* | +0.0294 [+0.0233, +0.0358]* | +0.0120 [+0.0064, +0.0182]* | -0.0160 [-0.0233, -0.0088]* |
| 2 | 400 | 354 | -0.0088 [-0.0160, -0.0015]* | -0.0057 [-0.0136, +0.0017] | -0.0123 [-0.0199, -0.0048]* | -0.0369 [-0.0472, -0.0272]* | +0.0084 [-0.0050, +0.0209] | +0.1080 [+0.0938, +0.1219]* | +0.0224 [+0.0146, +0.0308]* | +0.0084 [+0.0007, +0.0161]* | -0.0088 [-0.0184, +0.0009] |
| 3 | 65 | 64 | +0.0112 [-0.0080, +0.0278] | +0.0046 [-0.0148, +0.0230] | +0.0051 [-0.0158, +0.0247] | -0.0183 [-0.0440, +0.0077] | +0.0037 [-0.0384, +0.0442] | +0.1159 [+0.0817, +0.1498]* | +0.0447 [+0.0197, +0.0672]* | +0.0229 [+0.0022, +0.0412]* | +0.0101 [-0.0155, +0.0355] |
| 4 | n<10 | |  |  |  |  |  |  |  |  |  |

**n_measurement**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=25 | 348 | 328 | -0.0281 [-0.0362, -0.0203]* | -0.0283 [-0.0365, -0.0205]* | -0.0241 [-0.0311, -0.0164]* | -0.0566 [-0.0670, -0.0462]* | -0.0314 [-0.0433, -0.0193]* | +0.0681 [+0.0533, +0.0819]* | +0.0322 [+0.0227, +0.0421]* | +0.0057 [-0.0027, +0.0144] | -0.0117 [-0.0214, -0.0017]* |
| (25,33] | 336 | 335 | -0.0227 [-0.0307, -0.0147]* | -0.0256 [-0.0336, -0.0178]* | -0.0215 [-0.0290, -0.0144]* | -0.0884 [-0.0989, -0.0776]* | -0.0001 [-0.0123, +0.0122] | +0.0608 [+0.0474, +0.0746]* | +0.0348 [+0.0264, +0.0434]* | +0.0249 [+0.0173, +0.0320]* | -0.0362 [-0.0456, -0.0265]* |
| (33,60] | 339 | 324 | -0.0129 [-0.0211, -0.0043]* | -0.0119 [-0.0199, -0.0040]* | -0.0136 [-0.0218, -0.0049]* | -0.0371 [-0.0471, -0.0277]* | -0.0152 [-0.0286, -0.0017]* | +0.0856 [+0.0709, +0.1007]* | +0.0269 [+0.0188, +0.0356]* | +0.0125 [+0.0048, +0.0200]* | -0.0058 [-0.0151, +0.0034] |
| >60 | 322 | 237 | +0.0015 [-0.0072, +0.0094] | +0.0130 [+0.0045, +0.0205]* | -0.0027 [-0.0115, +0.0056] | -0.0058 [-0.0155, +0.0029] | -0.0135 [-0.0263, -0.0012]* | +0.1024 [+0.0887, +0.1159]* | +0.0262 [+0.0168, +0.0355]* | +0.0069 [-0.0022, +0.0156] | +0.0101 [+0.0012, +0.0183]* |

**n_vital**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 16 | +0.0038 [-0.0224, +0.0329]+ | +0.0274 [-0.0096, +0.0692]+ | +0.0245 [-0.0094, +0.0605]+ | +0.0255 [-0.0166, +0.0670]+ | +0.0279 [-0.0319, +0.0877]+ | +0.1699 [+0.0851, +0.2601]*+ | +0.0229 [-0.0247, +0.0710]+ | +0.0328 [-0.0046, +0.0735]+ | +0.0479 [+0.0155, +0.0813]*+ |
| 1 | n<10 | |  |  |  |  |  |  |  |  |  |
| 2 | n<10 | |  |  |  |  |  |  |  |  |  |
| 4 | n<10 | |  |  |  |  |  |  |  |  |  |
| 5 | 24 | 23 | -0.0123 [-0.0350, +0.0106]+ | -0.0079 [-0.0380, +0.0229]+ | -0.0245 [-0.0576, +0.0065]+ | -0.0373 [-0.0768, +0.0019]+ | -0.0381 [-0.0858, +0.0088]+ | +0.1038 [+0.0364, +0.1880]*+ | +0.0392 [+0.0091, +0.0706]*+ | -0.0021 [-0.0326, +0.0286]+ | -0.0120 [-0.0458, +0.0262]+ |
| 6 | 1298 | 990 | -0.0163 [-0.0213, -0.0114]* | -0.0142 [-0.0193, -0.0093]* | -0.0162 [-0.0208, -0.0116]* | -0.0486 [-0.0554, -0.0419]* | -0.0150 [-0.0225, -0.0078]* | +0.0773 [+0.0694, +0.0850]* | +0.0298 [+0.0248, +0.0351]* | +0.0124 [+0.0075, +0.0171]* | -0.0118 [-0.0178, -0.0059]* |

**n_diagnosis**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0205 [-0.0264, -0.0146]* | -0.0261 [-0.0318, -0.0202]* | -0.0210 [-0.0264, -0.0154]* | -0.0879 [-0.0955, -0.0797]* | -0.0089 [-0.0182, +0.0001] | +0.0637 [+0.0543, +0.0732]* | +0.0342 [+0.0282, +0.0405]* | +0.0205 [+0.0150, +0.0262]* | -0.0391 [-0.0461, -0.0315]* |
| (0,3] | 390 | 334 | -0.0103 [-0.0189, -0.0017]* | -0.0022 [-0.0103, +0.0054] | -0.0083 [-0.0167, +0.0003] | -0.0007 [-0.0096, +0.0076] | -0.0169 [-0.0294, -0.0041]* | +0.0918 [+0.0775, +0.1065]* | +0.0283 [+0.0198, +0.0375]* | +0.0082 [-0.0004, +0.0169] | +0.0250 [+0.0159, +0.0333]* |
| >3 | 275 | 204 | -0.0119 [-0.0205, -0.0034]* | +0.0013 [-0.0085, +0.0100] | -0.0130 [-0.0221, -0.0044]* | -0.0138 [-0.0240, -0.0042]* | -0.0286 [-0.0427, -0.0151]* | +0.0981 [+0.0812, +0.1147]* | +0.0223 [+0.0117, +0.0329]* | -0.0012 [-0.0105, +0.0083] | +0.0069 [-0.0021, +0.0153] |

**share_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0137 | 340 | 277 | -0.0107 [-0.0189, -0.0029]* | -0.0050 [-0.0132, +0.0025] | -0.0113 [-0.0191, -0.0032]* | -0.0321 [-0.0415, -0.0230]* | -0.0333 [-0.0448, -0.0214]* | +0.0784 [+0.0664, +0.0904]* | +0.0368 [+0.0276, +0.0461]* | +0.0144 [+0.0059, +0.0227]* | +0.0022 [-0.0062, +0.0102] |
| (0.0137,0.027] | 356 | 321 | -0.0152 [-0.0230, -0.0075]* | -0.0108 [-0.0189, -0.0030]* | -0.0142 [-0.0219, -0.0063]* | -0.0408 [-0.0510, -0.0310]* | -0.0173 [-0.0277, -0.0068]* | +0.0772 [+0.0632, +0.0909]* | +0.0240 [+0.0161, +0.0325]* | +0.0101 [+0.0026, +0.0179]* | -0.0091 [-0.0187, +0.0004] |
| (0.027,0.0444] | 318 | 305 | -0.0231 [-0.0321, -0.0144]* | -0.0244 [-0.0334, -0.0148]* | -0.0229 [-0.0315, -0.0142]* | -0.0737 [-0.0858, -0.0614]* | +0.0021 [-0.0104, +0.0146] | +0.0593 [+0.0460, +0.0740]* | +0.0340 [+0.0252, +0.0435]* | +0.0193 [+0.0107, +0.0278]* | -0.0277 [-0.0380, -0.0178]* |
| >0.0444 | 331 | 316 | -0.0147 [-0.0229, -0.0068]* | -0.0150 [-0.0229, -0.0077]* | -0.0149 [-0.0222, -0.0077]* | -0.0453 [-0.0559, -0.0353]* | -0.0111 [-0.0269, +0.0044] | +0.1002 [+0.0850, +0.1152]* | +0.0260 [+0.0171, +0.0347]* | +0.0065 [-0.0017, +0.0146] | -0.0110 [-0.0210, -0.0008]* |

**share_measurement**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.767 | 337 | 317 | -0.0242 [-0.0326, -0.0158]* | -0.0241 [-0.0326, -0.0160]* | -0.0211 [-0.0288, -0.0135]* | -0.0444 [-0.0554, -0.0336]* | -0.0247 [-0.0377, -0.0122]* | +0.0758 [+0.0607, +0.0910]* | +0.0331 [+0.0238, +0.0426]* | +0.0046 [-0.0039, +0.0131] | -0.0029 [-0.0133, +0.0070] |
| (0.767,0.814] | 339 | 330 | -0.0273 [-0.0349, -0.0195]* | -0.0300 [-0.0379, -0.0224]* | -0.0266 [-0.0342, -0.0194]* | -0.0907 [-0.1013, -0.0800]* | -0.0020 [-0.0159, +0.0116] | +0.0642 [+0.0509, +0.0776]* | +0.0328 [+0.0246, +0.0416]* | +0.0219 [+0.0143, +0.0291]* | -0.0385 [-0.0476, -0.0288]* |
| (0.814,0.86] | 333 | 313 | -0.0135 [-0.0215, -0.0059]* | -0.0125 [-0.0208, -0.0045]* | -0.0119 [-0.0197, -0.0038]* | -0.0462 [-0.0567, -0.0363]* | -0.0217 [-0.0342, -0.0090]* | +0.0754 [+0.0608, +0.0905]* | +0.0280 [+0.0197, +0.0366]* | +0.0155 [+0.0078, +0.0229]* | -0.0116 [-0.0219, -0.0022]* |
| >0.86 | 336 | 256 | +0.0018 [-0.0069, +0.0096] | +0.0124 [+0.0044, +0.0196]* | -0.0030 [-0.0119, +0.0055] | -0.0082 [-0.0175, -0.0001]* | -0.0127 [-0.0250, -0.0012]* | +0.1003 [+0.0862, +0.1145]* | +0.0263 [+0.0179, +0.0347]* | +0.0078 [-0.0001, +0.0157] | +0.0087 [+0.0007, +0.0164]* |

**share_vital**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0.0845 | 341 | 253 | -0.0002 [-0.0087, +0.0079] | +0.0125 [+0.0043, +0.0198]* | -0.0035 [-0.0121, +0.0050] | -0.0051 [-0.0143, +0.0035] | -0.0141 [-0.0272, -0.0017]* | +0.1047 [+0.0906, +0.1180]* | +0.0264 [+0.0175, +0.0355]* | +0.0078 [-0.0008, +0.0164] | +0.0129 [+0.0047, +0.0210]* |
| (0.0845,0.146] | 350 | 334 | -0.0110 [-0.0190, -0.0028]* | -0.0132 [-0.0209, -0.0055]* | -0.0131 [-0.0209, -0.0049]* | -0.0401 [-0.0502, -0.0305]* | -0.0165 [-0.0293, -0.0032]* | +0.0838 [+0.0698, +0.0984]* | +0.0252 [+0.0170, +0.0334]* | +0.0123 [+0.0048, +0.0196]* | -0.0103 [-0.0197, -0.0009]* |
| (0.146,0.182] | 324 | 323 | -0.0248 [-0.0329, -0.0171]* | -0.0248 [-0.0326, -0.0167]* | -0.0233 [-0.0307, -0.0161]* | -0.0892 [-0.0995, -0.0785]* | +0.0028 [-0.0096, +0.0155] | +0.0581 [+0.0446, +0.0724]* | +0.0347 [+0.0262, +0.0436]* | +0.0224 [+0.0151, +0.0298]* | -0.0377 [-0.0468, -0.0282]* |
| >0.182 | 330 | 315 | -0.0284 [-0.0369, -0.0201]* | -0.0299 [-0.0384, -0.0214]* | -0.0236 [-0.0311, -0.0157]* | -0.0581 [-0.0687, -0.0472]* | -0.0328 [-0.0457, -0.0200]* | +0.0674 [+0.0531, +0.0819]* | +0.0345 [+0.0249, +0.0446]* | +0.0077 [-0.0007, +0.0168] | -0.0108 [-0.0211, -0.0003]* |

**share_diagnosis**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0205 [-0.0264, -0.0146]* | -0.0261 [-0.0318, -0.0202]* | -0.0210 [-0.0264, -0.0154]* | -0.0879 [-0.0955, -0.0797]* | -0.0089 [-0.0182, +0.0001] | +0.0637 [+0.0543, +0.0732]* | +0.0342 [+0.0282, +0.0405]* | +0.0205 [+0.0150, +0.0262]* | -0.0391 [-0.0461, -0.0315]* |
| (0,0.0441] | 329 | 259 | -0.0011 [-0.0097, +0.0070] | +0.0048 [-0.0042, +0.0129] | -0.0041 [-0.0138, +0.0048] | -0.0057 [-0.0153, +0.0035] | -0.0211 [-0.0351, -0.0071]* | +0.0879 [+0.0727, +0.1034]* | +0.0268 [+0.0185, +0.0359]* | +0.0092 [+0.0008, +0.0180]* | +0.0132 [+0.0049, +0.0213]* |
| >0.0441 | 336 | 257 | -0.0207 [-0.0295, -0.0121]* | -0.0062 [-0.0147, +0.0019] | -0.0163 [-0.0248, -0.0081]* | -0.0065 [-0.0164, +0.0022] | -0.0224 [-0.0345, -0.0100]* | +0.1008 [+0.0856, +0.1160]* | +0.0249 [+0.0146, +0.0360]* | -0.0005 [-0.0097, +0.0089] | +0.0217 [+0.0119, +0.0308]* |

**n_knowledge**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=21 | 347 | 331 | -0.0233 [-0.0312, -0.0152]* | -0.0246 [-0.0321, -0.0165]* | -0.0193 [-0.0261, -0.0117]* | -0.0618 [-0.0723, -0.0515]* | -0.0275 [-0.0393, -0.0151]* | +0.0658 [+0.0520, +0.0796]* | +0.0313 [+0.0229, +0.0410]* | +0.0107 [+0.0026, +0.0190]* | -0.0186 [-0.0283, -0.0082]* |
| (21,25] | 407 | 389 | -0.0188 [-0.0270, -0.0108]* | -0.0221 [-0.0297, -0.0143]* | -0.0158 [-0.0236, -0.0084]* | -0.0695 [-0.0797, -0.0596]* | +0.0022 [-0.0100, +0.0143] | +0.0729 [+0.0605, +0.0859]* | +0.0317 [+0.0238, +0.0400]* | +0.0210 [+0.0134, +0.0286]* | -0.0256 [-0.0344, -0.0166]* |
| (25,30] | 290 | 265 | -0.0107 [-0.0197, -0.0022]* | -0.0072 [-0.0155, +0.0013] | -0.0153 [-0.0242, -0.0062]* | -0.0335 [-0.0441, -0.0234]* | -0.0131 [-0.0276, +0.0009] | +0.0856 [+0.0699, +0.1018]* | +0.0275 [+0.0187, +0.0366]* | +0.0100 [+0.0013, +0.0185]* | -0.0027 [-0.0124, +0.0067] |
| >30 | 301 | 218 | -0.0081 [-0.0164, -0.0002]* | +0.0044 [-0.0036, +0.0122] | -0.0118 [-0.0204, -0.0036]* | -0.0147 [-0.0243, -0.0056]* | -0.0266 [-0.0401, -0.0143]* | +0.0956 [+0.0801, +0.1106]* | +0.0289 [+0.0193, +0.0383]* | +0.0054 [-0.0033, +0.0139] | +0.0090 [+0.0004, +0.0170]* |

**n_analyte**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=23 | 367 | 328 | -0.0267 [-0.0350, -0.0187]* | -0.0224 [-0.0313, -0.0139]* | -0.0221 [-0.0289, -0.0142]* | -0.0402 [-0.0509, -0.0298]* | -0.0380 [-0.0507, -0.0249]* | +0.0750 [+0.0609, +0.0900]* | +0.0335 [+0.0244, +0.0432]* | +0.0037 [-0.0042, +0.0123] | +0.0008 [-0.0088, +0.0104] |
| (23,29] | 388 | 344 | -0.0153 [-0.0232, -0.0075]* | -0.0183 [-0.0260, -0.0104]* | -0.0175 [-0.0255, -0.0100]* | -0.0634 [-0.0738, -0.0529]* | +0.0026 [-0.0095, +0.0147] | +0.0681 [+0.0550, +0.0819]* | +0.0341 [+0.0262, +0.0426]* | +0.0227 [+0.0152, +0.0299]* | -0.0203 [-0.0291, -0.0110]* |
| (29,33] | 294 | 274 | -0.0164 [-0.0262, -0.0073]* | -0.0127 [-0.0209, -0.0048]* | -0.0139 [-0.0237, -0.0049]* | -0.0478 [-0.0582, -0.0377]* | -0.0196 [-0.0345, -0.0058]* | +0.0781 [+0.0632, +0.0929]* | +0.0220 [+0.0128, +0.0313]* | +0.0105 [+0.0017, +0.0195]* | -0.0161 [-0.0257, -0.0060]* |
| >33 | 296 | 258 | -0.0025 [-0.0106, +0.0054] | +0.0026 [-0.0060, +0.0108] | -0.0072 [-0.0155, +0.0013] | -0.0353 [-0.0464, -0.0250]* | -0.0060 [-0.0186, +0.0061] | +0.0987 [+0.0839, +0.1134]* | +0.0285 [+0.0190, +0.0378]* | +0.0119 [+0.0036, +0.0206]* | -0.0090 [-0.0200, +0.0013] |

**n_prior_dx**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0205 [-0.0264, -0.0146]* | -0.0261 [-0.0318, -0.0202]* | -0.0210 [-0.0264, -0.0154]* | -0.0879 [-0.0955, -0.0797]* | -0.0089 [-0.0182, +0.0001] | +0.0637 [+0.0543, +0.0732]* | +0.0342 [+0.0282, +0.0405]* | +0.0205 [+0.0150, +0.0262]* | -0.0391 [-0.0461, -0.0315]* |
| (0,3] | 390 | 334 | -0.0103 [-0.0189, -0.0017]* | -0.0022 [-0.0103, +0.0054] | -0.0083 [-0.0167, +0.0003] | -0.0007 [-0.0096, +0.0076] | -0.0169 [-0.0294, -0.0041]* | +0.0918 [+0.0775, +0.1065]* | +0.0283 [+0.0198, +0.0375]* | +0.0082 [-0.0004, +0.0169] | +0.0250 [+0.0159, +0.0333]* |
| >3 | 275 | 204 | -0.0119 [-0.0205, -0.0034]* | +0.0013 [-0.0085, +0.0100] | -0.0130 [-0.0221, -0.0044]* | -0.0138 [-0.0240, -0.0042]* | -0.0286 [-0.0427, -0.0151]* | +0.0981 [+0.0812, +0.1147]* | +0.0223 [+0.0117, +0.0329]* | -0.0012 [-0.0105, +0.0083] | +0.0069 [-0.0021, +0.0153] |

**rel_trajectory_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 1039 | 907 | -0.0202 [-0.0253, -0.0154]* | -0.0212 [-0.0262, -0.0162]* | -0.0189 [-0.0237, -0.0140]* | -0.0600 [-0.0663, -0.0530]* | -0.0139 [-0.0213, -0.0064]* | +0.0732 [+0.0653, +0.0816]* | +0.0309 [+0.0259, +0.0363]* | +0.0148 [+0.0099, +0.0197]* | -0.0180 [-0.0242, -0.0116]* |
| >0 | 306 | 229 | -0.0009 [-0.0108, +0.0085] | +0.0123 [+0.0033, +0.0202]* | -0.0048 [-0.0153, +0.0048] | -0.0050 [-0.0145, +0.0037] | -0.0198 [-0.0344, -0.0054]* | +0.0981 [+0.0833, +0.1122]* | +0.0272 [+0.0163, +0.0381]* | +0.0045 [-0.0057, +0.0143] | +0.0122 [+0.0039, +0.0199]* |

**rel_recurrence_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 680 | 679 | -0.0205 [-0.0264, -0.0146]* | -0.0261 [-0.0318, -0.0202]* | -0.0210 [-0.0264, -0.0154]* | -0.0879 [-0.0955, -0.0797]* | -0.0089 [-0.0182, +0.0001] | +0.0637 [+0.0543, +0.0732]* | +0.0342 [+0.0282, +0.0405]* | +0.0205 [+0.0150, +0.0262]* | -0.0391 [-0.0461, -0.0315]* |
| (0,3] | 390 | 334 | -0.0103 [-0.0189, -0.0017]* | -0.0022 [-0.0103, +0.0054] | -0.0083 [-0.0167, +0.0003] | -0.0007 [-0.0096, +0.0076] | -0.0169 [-0.0294, -0.0041]* | +0.0918 [+0.0775, +0.1065]* | +0.0283 [+0.0198, +0.0375]* | +0.0082 [-0.0004, +0.0169] | +0.0250 [+0.0159, +0.0333]* |
| >3 | 275 | 204 | -0.0119 [-0.0205, -0.0034]* | +0.0013 [-0.0085, +0.0100] | -0.0130 [-0.0221, -0.0044]* | -0.0138 [-0.0240, -0.0042]* | -0.0286 [-0.0427, -0.0151]* | +0.0981 [+0.0812, +0.1147]* | +0.0223 [+0.0117, +0.0329]* | -0.0012 [-0.0105, +0.0083] | +0.0069 [-0.0021, +0.0153] |

**rel_comorbid_with**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 862 | 783 | -0.0170 [-0.0226, -0.0117]* | -0.0201 [-0.0259, -0.0144]* | -0.0177 [-0.0231, -0.0123]* | -0.0693 [-0.0768, -0.0616]* | -0.0098 [-0.0182, -0.0014]* | +0.0702 [+0.0613, +0.0795]* | +0.0341 [+0.0285, +0.0400]* | +0.0180 [+0.0126, +0.0236]* | -0.0258 [-0.0327, -0.0186]* |
| (0,2] | 159 | 144 | -0.0229 [-0.0358, -0.0092]* | -0.0096 [-0.0213, +0.0024] | -0.0170 [-0.0285, -0.0043]* | -0.0021 [-0.0157, +0.0109] | -0.0309 [-0.0493, -0.0112]* | +0.0896 [+0.0690, +0.1106]* | +0.0172 [+0.0037, +0.0308]* | +0.0006 [-0.0126, +0.0135] | +0.0260 [+0.0139, +0.0373]* |
| >2 | 324 | 229 | -0.0092 [-0.0172, -0.0016]* | +0.0018 [-0.0064, +0.0097] | -0.0097 [-0.0181, -0.0017]* | -0.0117 [-0.0213, -0.0031]* | -0.0221 [-0.0355, -0.0092]* | +0.0969 [+0.0817, +0.1116]* | +0.0258 [+0.0163, +0.0358]* | +0.0037 [-0.0046, +0.0122] | +0.0098 [+0.0011, +0.0181]* |

**rel_co_complaint**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 873 | 697 | -0.0209 [-0.0267, -0.0150]* | -0.0185 [-0.0245, -0.0125]* | -0.0186 [-0.0238, -0.0133]* | -0.0543 [-0.0622, -0.0467]* | -0.0277 [-0.0355, -0.0198]* | +0.0619 [+0.0533, +0.0711]* | +0.0324 [+0.0266, +0.0385]* | +0.0137 [+0.0082, +0.0195]* | -0.0137 [-0.0205, -0.0069]* |
| 2 | 400 | 354 | -0.0088 [-0.0160, -0.0015]* | -0.0057 [-0.0136, +0.0017] | -0.0123 [-0.0199, -0.0048]* | -0.0369 [-0.0472, -0.0272]* | +0.0084 [-0.0050, +0.0209] | +0.1080 [+0.0938, +0.1219]* | +0.0224 [+0.0146, +0.0308]* | +0.0084 [+0.0007, +0.0161]* | -0.0088 [-0.0184, +0.0009] |
| 6 | 65 | 64 | +0.0112 [-0.0080, +0.0278] | +0.0046 [-0.0148, +0.0230] | +0.0051 [-0.0158, +0.0247] | -0.0183 [-0.0440, +0.0077] | +0.0037 [-0.0384, +0.0442] | +0.1159 [+0.0817, +0.1498]* | +0.0447 [+0.0197, +0.0672]* | +0.0229 [+0.0022, +0.0412]* | +0.0101 [-0.0155, +0.0355] |
| 12 | n<10 | |  |  |  |  |  |  |  |  |  |

**rel_baseline_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 768 | 755 | -0.0219 [-0.0276, -0.0165]* | -0.0263 [-0.0320, -0.0208]* | -0.0210 [-0.0262, -0.0156]* | -0.0778 [-0.0851, -0.0703]* | -0.0086 [-0.0175, -0.0002]* | +0.0666 [+0.0577, +0.0761]* | +0.0340 [+0.0285, +0.0400]* | +0.0191 [+0.0136, +0.0244]* | -0.0293 [-0.0360, -0.0220]* |
| (0,25] | 287 | 242 | -0.0126 [-0.0218, -0.0033]* | +0.0002 [-0.0093, +0.0087] | -0.0112 [-0.0213, -0.0014]* | -0.0034 [-0.0136, +0.0053] | -0.0301 [-0.0443, -0.0146]* | +0.0917 [+0.0753, +0.1089]* | +0.0260 [+0.0151, +0.0372]* | +0.0029 [-0.0071, +0.0128] | +0.0198 [+0.0098, +0.0290]* |
| >25 | 290 | 233 | -0.0028 [-0.0121, +0.0055] | +0.0064 [-0.0020, +0.0143] | -0.0061 [-0.0149, +0.0023] | -0.0108 [-0.0202, -0.0019]* | -0.0180 [-0.0311, -0.0059]* | +0.0988 [+0.0850, +0.1123]* | +0.0237 [+0.0141, +0.0332]* | +0.0045 [-0.0046, +0.0134] | +0.0064 [-0.0028, +0.0154] |

**rel_medical_total**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=63 | 353 | 327 | -0.0265 [-0.0346, -0.0185]* | -0.0241 [-0.0322, -0.0158]* | -0.0206 [-0.0279, -0.0129]* | -0.0466 [-0.0572, -0.0359]* | -0.0390 [-0.0516, -0.0267]* | +0.0724 [+0.0585, +0.0867]* | +0.0327 [+0.0232, +0.0425]* | +0.0057 [-0.0027, +0.0147] | -0.0047 [-0.0143, +0.0053] |
| (63,75] | 352 | 327 | -0.0183 [-0.0261, -0.0101]* | -0.0176 [-0.0261, -0.0093]* | -0.0187 [-0.0263, -0.0113]* | -0.0677 [-0.0788, -0.0562]* | +0.0013 [-0.0111, +0.0132] | +0.0680 [+0.0544, +0.0823]* | +0.0351 [+0.0273, +0.0433]* | +0.0207 [+0.0135, +0.0280]* | -0.0242 [-0.0338, -0.0144]* |
| (75,87] | 320 | 291 | -0.0136 [-0.0225, -0.0051]* | -0.0157 [-0.0239, -0.0080]* | -0.0148 [-0.0240, -0.0058]* | -0.0529 [-0.0635, -0.0429]* | -0.0141 [-0.0289, -0.0001]* | +0.0746 [+0.0595, +0.0894]* | +0.0210 [+0.0129, +0.0297]* | +0.0118 [+0.0034, +0.0202]* | -0.0206 [-0.0301, -0.0115]* |
| >87 | 320 | 251 | -0.0036 [-0.0115, +0.0043] | +0.0045 [-0.0035, +0.0125] | -0.0078 [-0.0163, +0.0004] | -0.0208 [-0.0308, -0.0109]* | -0.0083 [-0.0211, +0.0042] | +0.1024 [+0.0888, +0.1160]* | +0.0307 [+0.0211, +0.0398]* | +0.0116 [+0.0032, +0.0203]* | +0.0057 [-0.0039, +0.0149] |

**rel_medical_member_of**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=21 | 354 | 317 | -0.0274 [-0.0354, -0.0191]* | -0.0227 [-0.0315, -0.0141]* | -0.0222 [-0.0295, -0.0143]* | -0.0387 [-0.0495, -0.0283]* | -0.0402 [-0.0526, -0.0268]* | +0.0745 [+0.0599, +0.0899]* | +0.0339 [+0.0247, +0.0439]* | +0.0034 [-0.0049, +0.0124] | +0.0029 [-0.0071, +0.0126] |
| (21,28] | 483 | 424 | -0.0160 [-0.0230, -0.0091]* | -0.0164 [-0.0232, -0.0094]* | -0.0166 [-0.0238, -0.0099]* | -0.0581 [-0.0673, -0.0487]* | -0.0016 [-0.0124, +0.0091] | +0.0692 [+0.0574, +0.0814]* | +0.0324 [+0.0253, +0.0400]* | +0.0203 [+0.0133, +0.0268]* | -0.0171 [-0.0254, -0.0091]* |
| (28,31] | 188 | 180 | -0.0126 [-0.0242, -0.0018]* | -0.0084 [-0.0181, +0.0011] | -0.0094 [-0.0201, +0.0003] | -0.0483 [-0.0610, -0.0358]* | -0.0212 [-0.0384, -0.0049]* | +0.0739 [+0.0556, +0.0928]* | +0.0210 [+0.0103, +0.0317]* | +0.0122 [+0.0019, +0.0219]* | -0.0193 [-0.0312, -0.0073]* |
| >31 | 320 | 280 | -0.0047 [-0.0130, +0.0035] | -0.0023 [-0.0107, +0.0061] | -0.0107 [-0.0192, -0.0021]* | -0.0407 [-0.0514, -0.0298]* | -0.0048 [-0.0172, +0.0071] | +0.1013 [+0.0873, +0.1153]* | +0.0277 [+0.0184, +0.0371]* | +0.0109 [+0.0025, +0.0196]* | -0.0128 [-0.0235, -0.0027]* |

**rel_medical_assesses**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=12 | 457 | 400 | -0.0195 [-0.0270, -0.0119]* | -0.0166 [-0.0242, -0.0096]* | -0.0183 [-0.0254, -0.0110]* | -0.0415 [-0.0512, -0.0325]* | -0.0307 [-0.0419, -0.0187]* | +0.0756 [+0.0634, +0.0886]* | +0.0315 [+0.0231, +0.0405]* | +0.0097 [+0.0019, +0.0180]* | -0.0022 [-0.0109, +0.0066] |
| (12,15] | 459 | 401 | -0.0216 [-0.0285, -0.0147]* | -0.0210 [-0.0284, -0.0138]* | -0.0193 [-0.0263, -0.0124]* | -0.0595 [-0.0692, -0.0499]* | -0.0059 [-0.0180, +0.0057] | +0.0729 [+0.0611, +0.0850]* | +0.0292 [+0.0222, +0.0363]* | +0.0169 [+0.0102, +0.0232]* | -0.0184 [-0.0268, -0.0106]* |
| (15,16] | 142 | 133 | -0.0170 [-0.0287, -0.0054]* | -0.0131 [-0.0241, -0.0027]* | -0.0152 [-0.0269, -0.0035]* | -0.0496 [-0.0639, -0.0356]* | -0.0130 [-0.0322, +0.0064] | +0.0768 [+0.0562, +0.0955]* | +0.0238 [+0.0115, +0.0360]* | +0.0086 [-0.0033, +0.0211] | -0.0161 [-0.0300, -0.0017]* |
| >16 | 287 | 255 | -0.0002 [-0.0084, +0.0077] | +0.0028 [-0.0059, +0.0114] | -0.0059 [-0.0144, +0.0021] | -0.0366 [-0.0473, -0.0258]* | -0.0066 [-0.0195, +0.0064] | +0.0947 [+0.0796, +0.1103]* | +0.0324 [+0.0223, +0.0422]* | +0.0118 [+0.0033, +0.0201]* | -0.0112 [-0.0222, -0.0006]* |

**rel_medical_measures**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=22 | 382 | 335 | -0.0251 [-0.0334, -0.0170]* | -0.0201 [-0.0287, -0.0117]* | -0.0203 [-0.0273, -0.0122]* | -0.0372 [-0.0477, -0.0271]* | -0.0375 [-0.0503, -0.0238]* | +0.0759 [+0.0624, +0.0899]* | +0.0351 [+0.0261, +0.0447]* | +0.0036 [-0.0045, +0.0121] | +0.0025 [-0.0070, +0.0119] |
| (22,26] | 525 | 457 | -0.0190 [-0.0259, -0.0119]* | -0.0183 [-0.0251, -0.0115]* | -0.0189 [-0.0260, -0.0122]* | -0.0602 [-0.0693, -0.0514]* | -0.0044 [-0.0148, +0.0057] | +0.0732 [+0.0616, +0.0855]* | +0.0273 [+0.0201, +0.0347]* | +0.0161 [+0.0093, +0.0230]* | -0.0213 [-0.0290, -0.0133]* |
| (26,28] | 152 | 146 | -0.0051 [-0.0165, +0.0058] | -0.0047 [-0.0156, +0.0059] | -0.0048 [-0.0155, +0.0060] | -0.0409 [-0.0551, -0.0269]* | -0.0089 [-0.0294, +0.0107] | +0.0819 [+0.0599, +0.1050]* | +0.0326 [+0.0198, +0.0458]* | +0.0120 [+0.0005, +0.0229]* | -0.0170 [-0.0300, -0.0047]* |
| >28 | 286 | 246 | -0.0032 [-0.0115, +0.0050] | -0.0010 [-0.0096, +0.0072] | -0.0094 [-0.0182, -0.0006]* | -0.0413 [-0.0533, -0.0304]* | -0.0087 [-0.0208, +0.0033] | +0.0917 [+0.0771, +0.1057]* | +0.0272 [+0.0183, +0.0362]* | +0.0180 [+0.0096, +0.0265]* | -0.0075 [-0.0186, +0.0034] |

**rel_medical_is_a**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=1 | 506 | 506 | -0.0240 [-0.0304, -0.0175]* | -0.0301 [-0.0365, -0.0232]* | -0.0231 [-0.0295, -0.0166]* | -0.0938 [-0.1022, -0.0845]* | -0.0198 [-0.0297, -0.0105]* | +0.0472 [+0.0377, +0.0575]* | +0.0342 [+0.0272, +0.0415]* | +0.0202 [+0.0134, +0.0266]* | -0.0422 [-0.0498, -0.0336]* |
| (1,3] | 189 | 188 | -0.0130 [-0.0239, -0.0021]* | -0.0171 [-0.0282, -0.0061]* | -0.0168 [-0.0274, -0.0070]* | -0.0701 [-0.0841, -0.0562]* | +0.0174 [-0.0034, +0.0363] | +0.1110 [+0.0897, +0.1324]* | +0.0338 [+0.0228, +0.0458]* | +0.0184 [+0.0085, +0.0286]* | -0.0295 [-0.0419, -0.0165]* |
| (3,10] | 349 | 307 | -0.0109 [-0.0203, -0.0016]* | -0.0025 [-0.0110, +0.0055] | -0.0087 [-0.0182, +0.0004] | +0.0005 [-0.0090, +0.0092] | -0.0188 [-0.0313, -0.0056]* | +0.0909 [+0.0766, +0.1058]* | +0.0290 [+0.0196, +0.0386]* | +0.0093 [+0.0000, +0.0186]* | +0.0270 [+0.0176, +0.0363]* |
| >10 | 301 | 223 | -0.0096 [-0.0180, -0.0016]* | +0.0035 [-0.0056, +0.0119] | -0.0106 [-0.0194, -0.0023]* | -0.0109 [-0.0204, -0.0021]* | -0.0239 [-0.0374, -0.0105]* | +0.0982 [+0.0824, +0.1139]* | +0.0220 [+0.0118, +0.0320]* | -0.0004 [-0.0093, +0.0085] | +0.0084 [-0.0008, +0.0166] |

**rel_medical_may_affect**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 623 | 519 | -0.0133 [-0.0201, -0.0067]* | -0.0099 [-0.0162, -0.0037]* | -0.0114 [-0.0176, -0.0052]* | -0.0407 [-0.0484, -0.0328]* | -0.0277 [-0.0377, -0.0175]* | +0.0877 [+0.0768, +0.0985]* | +0.0275 [+0.0205, +0.0350]* | +0.0085 [+0.0017, +0.0154]* | -0.0092 [-0.0170, -0.0013]* |
| 1 | 479 | 415 | -0.0154 [-0.0226, -0.0084]* | -0.0203 [-0.0281, -0.0124]* | -0.0176 [-0.0247, -0.0108]* | -0.0550 [-0.0653, -0.0451]* | -0.0095 [-0.0200, +0.0015] | +0.0757 [+0.0641, +0.0880]* | +0.0337 [+0.0260, +0.0417]* | +0.0165 [+0.0096, +0.0237]* | -0.0170 [-0.0258, -0.0084]* |
| 2 | 172 | 162 | -0.0194 [-0.0300, -0.0087]* | -0.0066 [-0.0166, +0.0035] | -0.0186 [-0.0295, -0.0069]* | -0.0446 [-0.0602, -0.0294]* | +0.0135 [-0.0035, +0.0297] | +0.0715 [+0.0545, +0.0886]* | +0.0261 [+0.0149, +0.0375]* | +0.0130 [+0.0039, +0.0223]* | -0.0064 [-0.0198, +0.0077] |
| 3 | 54 | 54 | -0.0356 [-0.0558, -0.0155]* | -0.0243 [-0.0428, -0.0064]* | -0.0386 [-0.0587, -0.0186]* | -0.0744 [-0.0970, -0.0514]* | -0.0110 [-0.0462, +0.0234] | +0.0358 [+0.0092, +0.0634]* | +0.0302 [+0.0112, +0.0502]* | +0.0137 [-0.0073, +0.0332] | -0.0149 [-0.0358, +0.0062] |
| 4 | 11 | 11 | -0.0278 [-0.0701, +0.0130]+ | -0.0139 [-0.0517, +0.0223]+ | -0.0166 [-0.0450, +0.0113]+ | -0.0540 [-0.1061, -0.0082]*+ | -0.0345 [-0.1115, +0.0341]+ | +0.0243 [-0.0379, +0.0837]+ | +0.0667 [+0.0296, +0.1054]*+ | +0.0326 [-0.0013, +0.0650]+ | +0.0150 [-0.0308, +0.0586]+ |
| 5 | n<10 | |  |  |  |  |  |  |  |  |  |

**rel_medical_indicates**

| bin | n visits | n patients | IG/GSAT | IG/GC | Grad/GSAT | Grad/GC | GNNExp/GSAT | GNNExp/GC | bi/GSATbi | bi/GSATIG | bi/GCGrad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <=0 | 494 | 440 | -0.0231 [-0.0305, -0.0158]* | -0.0158 [-0.0237, -0.0083]* | -0.0186 [-0.0257, -0.0112]* | -0.0519 [-0.0618, -0.0423]* | -0.0222 [-0.0336, -0.0110]* | +0.0617 [+0.0504, +0.0734]* | +0.0241 [+0.0165, +0.0321]* | +0.0086 [+0.0017, +0.0154]* | -0.0128 [-0.0219, -0.0037]* |
| (0,1] | 272 | 254 | -0.0156 [-0.0250, -0.0059]* | -0.0192 [-0.0283, -0.0102]* | -0.0157 [-0.0242, -0.0072]* | -0.0566 [-0.0681, -0.0449]* | -0.0107 [-0.0236, +0.0027] | +0.0745 [+0.0583, +0.0914]* | +0.0303 [+0.0195, +0.0414]* | +0.0180 [+0.0079, +0.0280]* | -0.0193 [-0.0303, -0.0084]* |
| (1,3] | 267 | 240 | -0.0125 [-0.0215, -0.0035]* | -0.0153 [-0.0244, -0.0063]* | -0.0120 [-0.0217, -0.0026]* | -0.0493 [-0.0608, -0.0382]* | -0.0021 [-0.0171, +0.0144] | +0.0876 [+0.0718, +0.1042]* | +0.0278 [+0.0178, +0.0374]* | +0.0147 [+0.0058, +0.0232]* | -0.0145 [-0.0246, -0.0050]* |
| >3 | 312 | 243 | -0.0074 [-0.0153, +0.0008] | -0.0037 [-0.0120, +0.0051] | -0.0142 [-0.0227, -0.0055]* | -0.0310 [-0.0420, -0.0206]* | -0.0194 [-0.0317, -0.0071]* | +0.1025 [+0.0878, +0.1182]* | +0.0413 [+0.0326, +0.0509]* | +0.0119 [+0.0041, +0.0201]* | +0.0016 [-0.0082, +0.0119] |

## 7. Reproduce

`env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step2_structure.py` (runtime about 2 s). Writes `out/step2-strata.csv` (every printable cell with mean, CI, contrast) and this report.

