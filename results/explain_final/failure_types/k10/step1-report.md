# Step 1 report: per-visit tables and parity

Built by `analysis/failure_types/step1_visit_tables.py` (runtime 64 s). Aggregates only; counts of 1..9 are shown as <10. K=10, test fold, descriptive.

## Table sizes

| file | rows | columns |
|---|---|---|
| data/visit_features.csv | 634 | 49 |
| data/visit_scores.csv | 30432 | 29 |
| data/top_kinds.csv | 30273 | 17 |

Visits: 634 (visit_idx 0..633); patients: 500; keys: 4; seeds: 3; sources: 4. Reverse-relation edges in the graphs: 0 (the cei_final binding uses forward edges).

## Columns

**visit_features.csv**: `visit_idx`, `patient_idx`, `true_class_id`, `true_class`, `num_visits`, `n_nodes`, `n_edges`, `n_evidence`, `n_kind_patient`, `n_kind_visit`, `n_kind_complaint`, `n_kind_measurement`, `n_kind_analyte`, `n_kind_vital`, `n_kind_knowledge`, `n_kind_diagnosis`, `n_rel_has_visit`, `n_rel_index_visit_of`, `n_rel_reports_complaint`, `n_rel_measured_in`, `n_rel_instance_of`, `n_rel_observed_vital`, `n_rel_has_prior_diagnosis`, `n_rel_baseline_of`, `n_rel_trajectory_of`, `n_rel_co_complaint`, `n_rel_recurrence_of`, `n_rel_comorbid_with`, `n_rel_medical_member_of`, `n_rel_medical_assesses`, `n_rel_medical_measures`, `n_rel_medical_is_a`, `n_rel_medical_may_affect`, `n_rel_medical_interacts_with`, `n_rel_medical_indicates`, `n_prior_dx`, `density`, `conf_gsat`, `correct_gsat`, `pred_agree_gsat`, `conf_graphcare`, `correct_graphcare`, `pred_agree_graphcare`, `conf_protgnn`, `correct_protgnn`, `pred_agree_protgnn`, `conf_cei_final`, `correct_cei_final`, `pred_agree_cei_final`

**visit_scores.csv**: `visit_idx`, `key`, `seed`, `source`, `ok`, `status`, `pred_class`, `pred_prob`, `fidelity_plus`, `fidelity_minus`, `deletion_auc`, `insertion_auc`, `deletion_mean_change`, `insertion_mean_change`, `explained_fraction`, `mass90_sparsity`, `one_minus_support_fraction`, `fidelity_plus_vs_random`, `fidelity_minus_vs_random`, `deletion_auc_vs_random`, `insertion_auc_vs_random`, `deletion_mean_change_vs_random`, `insertion_mean_change_vs_random`, `fidelity_plus_beats_random`, `fidelity_minus_beats_random`, `deletion_auc_beats_random`, `insertion_auc_beats_random`, `deletion_mean_change_beats_random`, `insertion_mean_change_beats_random`

**top_kinds.csv**: `visit_idx`, `key`, `seed`, `source`, `k`, `top_share_complaint`, `top_share_measurement`, `top_share_vital`, `top_share_diagnosis`, `mass_share_complaint`, `mass_share_measurement`, `mass_share_vital`, `mass_share_diagnosis`, `top_rate_complaint`, `top_rate_measurement`, `top_rate_vital`, `top_rate_diagnosis`

`visit_scores.status` is the summarize category of the source on that visit: explained, unsupported, failed_explainer, failed_fidelity, unavailable, no_eligible, or absent (source not in the record); `ok` = 1 exactly when it is `explained`. `top_kinds` ties at the top-k cut share the remaining slots equally (expectation over random tie orders); `k` is the record's fidelity+ k, with the recomputed k checked in `data/parity.md`.

## Dropped / not-ok visit counts per key and source

Counts over the 3 seeds x 634 visits of each status; the last two columns count visits (of 634) that are ok in at least 2 and in all 3 seeds (the inclusion rule of the analysis).

| key | source | explained | unsupported | failed explainer | failed fidelity | unavailable | no eligible | absent | visits ok in >=2 seeds | visits ok in 3 seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| gsat | IG | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| gsat | Grad | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| gsat | GNNExp | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| gsat | builtin | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| graphcare | IG | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| graphcare | Grad | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| graphcare | GNNExp | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| graphcare | builtin | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| protgnn | IG | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| protgnn | Grad | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| protgnn | GNNExp | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| protgnn | builtin | 1743 | 0 | 0 | 0 | 0 | 159 | 0 | 624 | 485 |
| cei_final | IG | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| cei_final | Grad | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| cei_final | GNNExp | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |
| cei_final | builtin | 1902 | 0 | 0 | 0 | 0 | 0 | 0 | 634 | 634 |

## Parity summary

64 (key, source, metric) pairs compared with the stored `summary.csv` values; failing pairs: 0; largest absolute difference: 8.882e-16 (tolerance 1e-09). Full table in `data/parity.md`.

## Node-count and visit-set checks

All 4 keys x 3 seeds hold 634 records. Importance vectors checked against the node count of their visit: gsat 7608 (mismatch 0), graphcare 7608 (mismatch 0), protgnn 7608 (mismatch 0), cei_final 7608 (mismatch 0).

## Command

`PYTHONPATH=. python -m analysis.failure_types.step1_visit_tables` (settings from the FT_* environment)

STATUS: complete
