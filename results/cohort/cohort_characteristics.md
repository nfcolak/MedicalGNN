# Cohort and graph characteristics

Aggregates computed by `analysis/cohort_characteristics.py` from the graph file, the targets file, the manifests of the two cohort folders and the bindings of the trained runs. Counts below 25 are printed as "<25". No identifiers or per-visit rows.

## Cohort flow

| Step | Visits | Patients |
|---|---|---|
| All visits with a first laboratory result (no cap on visits per patient) | 185,782 | 60,240 |
| Cohort of the graphs (at most 6 visits per patient, one decision point per visit) | 100,355 | 51,961 |
| Excluded | 85,427 | 8,279 |
| Graphs | 100,355 | 51,961 |

Cap pool of the 6-visit cohort: 136,894 visits; of these, no laboratory charttime in the window: 34,531; no eligible recorded result in the window: 2,008; end before start: 1 (counted outside the pool).

Target labelling of the graphs:

| Status / reason | Visits |
|---|---|
| no_frozen_class | 54,553 |
| labelled | 45,692 |
| outside_common_stays | 110 |
| not labelled: r_symptom | 34,678 |
| not labelled: no_class | 18,991 |
| not labelled: override_none | 352 |
| not labelled: z_code | 218 |
| not labelled: no_diagnosis | 160 |
| not labelled: unmapped_icd9 | 154 |
| not labelled: outside_common_stays | 110 |

## Graph statistics

| Fold | Graphs | Mean nodes | Mean edges | Informative edges (%) |
|---|---|---|---|---|
| train | 80,246 | 112.8 | 188.6 | 53.9 |
| validation | 10,038 | 113.9 | 190.9 | 54.0 |
| test | 10,071 | 112.7 | 188.6 | 54.0 |

All folds: 100,355 graphs, 11,326,290 nodes (112.9 per graph), 18,949,586 edges (188.8 per graph), 54.0% informative.

Mean nodes per graph by kind (train fold):

| Kind | Mean per graph |
|---|---|
| measurement | 48.19 |
| analyte | 27.45 |
| knowledge | 25.96 |
| vital | 5.86 |
| diagnosis | 2.03 |
| complaint | 1.26 |
| patient | 1.00 |
| visit | 1.00 |

Coverage of the training graphs (graph manifest):

| With | Graphs | Share (%) |
|---|---|---|
| at least one complaint | 69,594 | 86.7 |
| triage vital signs | 78,884 | 98.3 |
| a laboratory baseline from an earlier visit | 37,410 | 46.6 |
| at least one prior diagnosis | 43,117 | 53.7 |
| knowledge relations | 80,246 | 100.0 |

Complaint vocabulary: 209 surface forms kept of 5,976 distinct in the training fold (minimum count 50).

## Vocabulary and absence items per task

| K | Token vocabulary | Token minimum count | Absence items | Node features | Edge features | Relations |
|---|---|---|---|---|---|---|
| 10 | 863 | 20 | 108 | 68 | 26 | 19 |
| 20 | 965 | 20 | 121 | 68 | 26 | 19 |
| 30 | 991 | 20 | 122 | 68 | 26 | 19 |

## Cohort characteristics per task and fold

Labelled visits of the K most frequent diagnoses (ranked by training count). Age in years at the visit (mean ± SD; median [Q1, Q3]); acuity is the ESI triage level.

| K | Fold | Visits | Patients | Age mean ± SD | Age median [Q1, Q3] | Age missing (%) | Female (%) | Sex missing (%) | Acuity 1 | 2 | 3 | 4 | 5 | Acuity missing (%) | Majority class (%) | Largest : smallest class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | train | 25,170 | 19,983 | 57.3 ± 21.3 | 59 [40, 75] | 0.0 | 53.5 | 0.0 | 1,175 (4.7) | 10,558 (41.9) | 12,513 (49.7) | 785 (3.1) | <25 (0.0) | 0.5 | 14.7 | 3.2 : 1 |
| 10 | validation | 3,165 | 2,473 | 57.4 ± 21.3 | 59 [41, 75] | 0.0 | 55.7 | 0.0 | 131 (4.1) | 1,331 (42.1) | 1,581 (50.0) | 104 (3.3) | <25 (0.0) | 0.6 | 15.2 | 3.0 : 1 |
| 10 | test | 3,175 | 2,497 | 57.9 ± 20.7 | 60 [41, 74] | 0.0 | 54.8 | 0.0 | 131 (4.1) | 1,347 (42.4) | 1,600 (50.4) | 83 (2.6) | <25 (0.1) | 0.4 | 14.4 | 3.2 : 1 |
| 20 | train | 33,532 | 25,252 | 57.7 ± 21.0 | 60 [41, 75] | 0.0 | 51.8 | 0.0 | 1,914 (5.7) | 14,877 (44.4) | 15,685 (46.8) | 846 (2.5) | <25 (0.0) | 0.6 | 11.0 | 6.8 : 1 |
| 20 | validation | 4,183 | 3,142 | 57.7 ± 21.1 | 59 [41, 75] | 0.0 | 53.9 | 0.0 | 214 (5.1) | 1,863 (44.5) | 1,966 (47.0) | 112 (2.7) | <25 (0.0) | 0.7 | 11.5 | 7.0 : 1 |
| 20 | test | 4,174 | 3,151 | 58.1 ± 20.4 | 60 [43, 74] | 0.0 | 53.4 | 0.0 | 218 (5.2) | 1,862 (44.6) | 1,982 (47.5) | 90 (2.2) | <25 (0.0) | 0.5 | 10.9 | 6.4 : 1 |
| 30 | train | 36,592 | 27,253 | 57.5 ± 21.0 | 60 [41, 75] | 0.0 | 52.4 | 0.0 | 2,184 (6.0) | 15,999 (43.7) | 17,295 (47.3) | 888 (2.4) | <25 (0.0) | 0.6 | 10.1 | 739.2 : 1 |
| 30 | validation | 4,534 | 3,388 | 57.5 ± 21.1 | 59 [41, 75] | 0.0 | 54.3 | 0.0 | 239 (5.3) | 2,004 (44.2) | 2,140 (47.2) | 123 (2.7) | <25 (0.0) | 0.6 | 10.6 | 481.0 : 1 |
| 30 | test | 4,566 | 3,404 | 58.0 ± 20.4 | 60 [42, 74] | 0.0 | 54.0 | 0.0 | 252 (5.5) | 2,014 (44.1) | 2,179 (47.7) | 97 (2.1) | <25 (0.0) | 0.5 | 10.0 | 91.2 : 1 |
