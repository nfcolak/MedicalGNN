# Step 1 report: per-visit tables and parity

Built by `comparison/explain/graph_types/step1_visit_tables.py` (runtime 105 s). Aggregates only; counts of 1..9 are shown as <10. Test fold, descriptive.

## Table sizes

| file | rows | columns |
|---|---|---|
| data/visit_features.csv | 1345 | 64 |
| data/visit_scores.csv | 145260 | 29 |
| data/top_kinds.csv | 144815 | 17 |

Visits: 1345 (visit_idx 0..1344); patients: 1000; keys: 9; seeds: 3; sources: 4. Reverse-relation edges in the graphs: 0 (the cei_final binding uses forward edges).

## Columns

**visit_features.csv**: `visit_idx`, `patient_idx`, `true_class_id`, `true_class`, `num_visits`, `n_nodes`, `n_edges`, `n_evidence`, `n_kind_patient`, `n_kind_visit`, `n_kind_complaint`, `n_kind_measurement`, `n_kind_analyte`, `n_kind_vital`, `n_kind_knowledge`, `n_kind_diagnosis`, `n_rel_has_visit`, `n_rel_index_visit_of`, `n_rel_reports_complaint`, `n_rel_measured_in`, `n_rel_instance_of`, `n_rel_observed_vital`, `n_rel_has_prior_diagnosis`, `n_rel_baseline_of`, `n_rel_trajectory_of`, `n_rel_co_complaint`, `n_rel_recurrence_of`, `n_rel_comorbid_with`, `n_rel_medical_member_of`, `n_rel_medical_assesses`, `n_rel_medical_measures`, `n_rel_medical_is_a`, `n_rel_medical_may_affect`, `n_rel_medical_interacts_with`, `n_rel_medical_indicates`, `n_prior_dx`, `density`, `conf_gsat`, `correct_gsat`, `pred_agree_gsat`, `conf_graphcare`, `correct_graphcare`, `pred_agree_graphcare`, `conf_protgnn`, `correct_protgnn`, `pred_agree_protgnn`, `conf_cei_final`, `correct_cei_final`, `pred_agree_cei_final`, `conf_cei_R1_ls0`, `correct_cei_R1_ls0`, `pred_agree_cei_R1_ls0`, `conf_cei_R2_ls0_wd`, `correct_cei_R2_ls0_wd`, `pred_agree_cei_R2_ls0_wd`, `conf_cei_R3_ls0_wd_de0`, `correct_cei_R3_ls0_wd_de0`, `pred_agree_cei_R3_ls0_wd_de0`, `conf_cei_R4_wd`, `correct_cei_R4_wd`, `pred_agree_cei_R4_wd`, `conf_cei_R5_de0`, `correct_cei_R5_de0`, `pred_agree_cei_R5_de0`

**visit_scores.csv**: `visit_idx`, `key`, `seed`, `source`, `ok`, `status`, `pred_class`, `pred_prob`, `fidelity_plus`, `fidelity_minus`, `deletion_auc`, `insertion_auc`, `deletion_mean_change`, `insertion_mean_change`, `explained_fraction`, `mass90_sparsity`, `one_minus_support_fraction`, `fidelity_plus_vs_random`, `fidelity_minus_vs_random`, `deletion_auc_vs_random`, `insertion_auc_vs_random`, `deletion_mean_change_vs_random`, `insertion_mean_change_vs_random`, `fidelity_plus_beats_random`, `fidelity_minus_beats_random`, `deletion_auc_beats_random`, `insertion_auc_beats_random`, `deletion_mean_change_beats_random`, `insertion_mean_change_beats_random`

**top_kinds.csv**: `visit_idx`, `key`, `seed`, `source`, `k`, `top_share_complaint`, `top_share_measurement`, `top_share_vital`, `top_share_diagnosis`, `mass_share_complaint`, `mass_share_measurement`, `mass_share_vital`, `mass_share_diagnosis`, `top_rate_complaint`, `top_rate_measurement`, `top_rate_vital`, `top_rate_diagnosis`

`visit_scores.status` is the summarize category of the source on that visit: explained, unsupported, failed_explainer, failed_fidelity, unavailable, no_eligible, or absent (source not in the record); `ok` = 1 exactly when it is `explained`. `top_kinds` ties at the top-k cut share the remaining slots equally (expectation over random tie orders); `k` is the record's fidelity+ k, with the recomputed k checked in `data/parity.md`.

## Dropped / not-ok visit counts per key and source

Counts over the 3 seeds x 1345 visits of each status; the last two columns count visits (of 1345) that are ok in at least 2 and in all 3 seeds (the inclusion rule of the analysis).

| key | source | explained | unsupported | failed explainer | failed fidelity | unavailable | no eligible | absent | visits ok in >=2 seeds | visits ok in 3 seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| gsat | IG | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| gsat | Grad | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| gsat | GNNExp | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| gsat | builtin | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| graphcare | IG | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| graphcare | Grad | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| graphcare | GNNExp | 4032 | <10 | 0 | 0 | 0 | 0 | 0 | 1344 | 1343 |
| graphcare | builtin | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| protgnn | IG | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| protgnn | Grad | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| protgnn | GNNExp | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| protgnn | builtin | 3593 | 0 | 0 | 0 | 0 | 442 | 0 | 1308 | 940 |
| cei_final | IG | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_final | Grad | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_final | GNNExp | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_final | builtin | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R1_ls0 | IG | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R1_ls0 | Grad | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R1_ls0 | GNNExp | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R1_ls0 | builtin | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R2_ls0_wd | IG | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R2_ls0_wd | Grad | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R2_ls0_wd | GNNExp | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R2_ls0_wd | builtin | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R3_ls0_wd_de0 | IG | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R3_ls0_wd_de0 | Grad | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R3_ls0_wd_de0 | GNNExp | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R3_ls0_wd_de0 | builtin | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R4_wd | IG | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R4_wd | Grad | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R4_wd | GNNExp | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R4_wd | builtin | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R5_de0 | IG | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R5_de0 | Grad | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R5_de0 | GNNExp | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |
| cei_R5_de0 | builtin | 4035 | 0 | 0 | 0 | 0 | 0 | 0 | 1345 | 1345 |

## Parity summary

144 (key, source, metric) pairs compared with the stored `summary.csv` values; failing pairs: 0; largest absolute difference: 7.772e-16 (tolerance 1e-09). Full table in `data/parity.md`.

## Node-count and visit-set checks

All 9 keys x 3 seeds hold 1345 records. Importance vectors checked against the node count of their visit: gsat 16140 (mismatch 0), graphcare 16137 (mismatch 0), protgnn 16140 (mismatch 0), cei_final 16140 (mismatch 0), cei_R1_ls0 16140 (mismatch 0), cei_R2_ls0_wd 16140 (mismatch 0), cei_R3_ls0_wd_de0 16140 (mismatch 0), cei_R4_wd 16140 (mismatch 0), cei_R5_de0 16140 (mismatch 0).

## Command

`env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 comparison/explain/graph_types/step1_visit_tables.py`

STATUS: complete
