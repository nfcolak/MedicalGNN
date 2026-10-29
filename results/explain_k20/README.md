# Explanation results, top-20 (test fold)

Explanations of the trained final runs of GSAT, GraphCare, ProtGNN and CEI-GNN mp2 at K = 20, and of five CEI-GNN
training-recipe variants. Every number is on the test fold; nothing here was used to choose, drop or rerun a model. The
recipe variants were explained after the test explanation results of the final runs had been seen, so they are a
diagnostic, not a model selection: the reported CEI-GNN stays the final run.

## Setup

| Item | Setting |
|---|---|
| Task | K = 20, protocol `fair_v10t`, test fold (4,174 visits, 3,151 patients) |
| Runs | 4 methods x 3 seeds (1234, 2025, 7) = 12 final runs; 5 CEI-GNN recipe variants x 3 seeds = 15 runs |
| Sample | 1,000 random test patients (sample seed 1234) with all their test visits = 1,345 visits, identical in all 27 runs |
| Explainers | Integrated Gradients from the not-recorded clinical baseline, GradExplainer, GNNExplainer (50 epochs per visit), each model's built-in explanation (CEI-GNN evidence ledger, GSAT node attention, GraphCare visit attention, ProtGNN prototype subgraph) |
| Headline | `deletion_mean_change_vs_random`: how much more the predicted-class probability drops when the top-ranked evidence is deleted than when random evidence of the same size is deleted (10 curve points, 50 random draws); higher is better |
| Code | explanation code at commit 6782d4b (source hash 411ae930d115 in every summary) |

CEI-GNN recipe variants (training: `comparison/nasim/recipe_control.py`; LS = label smoothing 0.1, WD = decoupled weight
decay 0.01, DE = DropEdge 0.2 of the final recipe):

| Arm | Removed | Trained |
|---|---|---|
| R1_ls0 | LS | `--date 20261012` |
| R2_ls0_wd | LS, WD | `--date 20261012` |
| R3_ls0_wd_de0 | LS, WD, DE (the baselines' recipe) | `--date 20261012` |
| R4_wd | WD | `--date 20261025 --arms R4_wd,R5_de0` |
| R5_de0 | DE | `--date 20261025 --arms R4_wd,R5_de0` |

Test macro-F1 of every arm against the final runs, with seed-and-patient intervals:
`results/baseline_fidelity_k20/analysis_R/summary.md`.

## Main results

- With Integrated Gradients and GradExplainer, GSAT and GraphCare get the highest scores (0.30-0.33), ProtGNN is in
  between (0.26-0.28) and CEI-GNN is lowest (0.25-0.26).
- CEI-GNN's evidence ledger (0.276) scores higher than any post-hoc explanation of CEI-GNN and close to GSAT's attention
  (0.292), and puts 90% of its importance on 25% of the nodes (mass90 sparsity 0.75; GSAT attention 0.10). The built-in
  explanations of GraphCare and ProtGNN are not better than random.
- Only label smoothing changes CEI-GNN's explanation scores. Without it every score rises by 0.014-0.046 and the ledger
  is the most faithful built-in explanation of all models (+0.030 [+0.025, +0.035] over GSAT's attention, paired
  patient-cluster bootstrap); test macro-F1 drops by 0.009. Weight decay and DropEdge change neither the scores nor
  macro-F1. Label smoothing lowers the mean predicted-class probability from 0.73 to 0.65, and the scores are
  probability drops.
- Graph types: in none of 29 predefined subgroups does the final CEI-GNN explain better than GSAT or GraphCare with
  Integrated Gradients or GradExplainer. The gap follows the confidence of the two models; against GraphCare it is
  largest on a patient's first visit and almost closes with two or three visits.

## Files

| File | Content |
|---|---|
| `final_methods/summary.{md,csv,json}` | explanation summary of the 12 final runs (`comparison/explain/summarize.py`) |
| `cei_recipe_variants/<arm>/summary.{md,csv,json}` | the same summary for each CEI-GNN recipe variant |
| `comparison_table.{md,json}` | all nine models in one table (performance on all test visits and every explanation score), per-explainer ranks, per-class F1 and explanation scores of the final CEI-GNN vs R1_ls0 (`comparison/explain/recipe_tables.py`) |
| `graph_types/` | paired graph-type analysis of CEI-GNN vs GSAT and GraphCare (`comparison/explain/graph_types/`, steps 1-6) |

All files are aggregates; no patient identifiers, rows or raw values.

## Reproduce (repo root, `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3` before each command)

```
# explanations of the 12 final runs, then their summary (inference only, long)
-m comparison.explain.run --ks 20 --sample-patients 1000 --jobs 4 --out <new dir> --execute
-m comparison.explain.summarize --explanations <that dir> --out <new dir>

# one CEI-GNN recipe variant (date 20261012 for R1-R3, 20261025 for R4 and R5)
-m comparison.explain.run --runs comparison/standardized/clinical_runs_nasim_s2_<date>/<arm> --ks 20 \
    --methods cei_mp2 --sample-patients 1000 --jobs 3 --out <new dir> --execute

# comparison table
-m comparison.explain.recipe_tables --final-summary <summary of the final runs> --final-records <explanations of the final runs> \
    --variant R1_ls0=<summary> ... --variant R5_de0=<summary> --ls0-records <explanations of R1_ls0> --out <new dir>

# graph types: set MEDGNN_EXPLAIN_ROOT to the folder that holds the 6 explanation folders (names as in KEYS of
# step1_visit_tables.py), then run steps 1-6; per-visit tables go to comparison/standardized/graph_types_k20/data (ignored)
comparison/explain/graph_types/step1_visit_tables.py   # ... step6_check.py
```
