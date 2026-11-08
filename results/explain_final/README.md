# Explanation results, top-10, top-20 and top-30 (test fold)

Explanations of the trained final runs of GSAT, GraphCare, ProtGNN and CEI-GNN mp2 at K = 10, 20 and 30. Every number is
on the test fold; nothing here was used to choose, drop or rerun a model. The folder covers K = 10, 20 and 30 and
holds matched evidence budgets, a ProtGNN prototype check, controls (weight randomisation, perturbation stability, seed
agreement, similar patients, relation shuffling) and a failure-type analysis. The five CEI-GNN training-recipe variants
(45 further runs, same protocol and explanation code) are in `cei_recipe_variants/`; they are a diagnostic made after the
test explanation results were seen, and the reported CEI-GNN stays the final model.

## Setup

| Item | Setting |
|---|---|
| Task | K = 10, 20, 30, protocol `fair_v10t`, test fold (3,175 / 4,174 / 4,566 visits) |
| Runs | 4 methods x 3 K x 3 seeds (1234, 2025, 7) = 36 runs, explained from the trained checkpoints; nothing retrained |
| Sample | 500 random test patients (sample seed 1234) with all their test visits = 634 / 672 / 671 visits at K = 10 / 20 / 30, identical for all methods at a K |
| Replay check | every run reproduced its stored test predictions before it was explained (exact hash, or 1e-4 with the same argmax): 36 of 36, 0 failed visits |
| Explainers | Integrated Gradients from the not-recorded clinical baseline, GradExplainer, GNNExplainer (50 epochs per visit), each model's built-in explanation (CEI-GNN evidence ledger, GSAT node attention, GraphCare visit attention, ProtGNN prototype subgraph) |
| Headline | `deletion_mean_change_vs_random`: how much more the predicted-class probability drops when the top-ranked evidence is deleted than when random evidence of the same size is deleted (10 curve points, 50 random draws); higher is better |
| Controls | weight randomisation and perturbation stability on the first 40 subjects of each run |
| Code | explanation code identity `5f7518e193aa` (122 files, git head `58a4def`) in every run manifest |

The trained graph inputs were reproduced from the repository rebuild with `analysis/trained_artifact.py` (the renamed
`reviewed_by` provenance label restored; all four SHA-256 values equal those recorded by the 36 trained runs), and the run
copies made with `analysis/trained_runs.py` (paths only).

## Main results

- With Integrated Gradients and GradExplainer, GSAT and GraphCare get the highest scores at every K (K = 20: 0.30-0.34).
  CEI-GNN is lowest at K = 20 (0.25-0.26, ProtGNN 0.26-0.28); at K = 10 (Grad) and K = 30 (IG and Grad) ProtGNN is lowest.
- CEI-GNN's evidence ledger scores 0.270 / 0.275 / 0.275 (K = 10 / 20 / 30), higher than any post-hoc explanation of
  CEI-GNN and below GSAT's attention (0.305 / 0.289 / 0.283); it puts 90% of its importance on 25-30% of the nodes
  (mass90 sparsity 0.70-0.75; GSAT attention 0.10).
- The built-in explanations of GraphCare (-0.016 / -0.012 / +0.017 +- 0.055) and ProtGNN (-0.007 / -0.009 / -0.007) are not
  clearly better than random. ProtGNN's has no eligible evidence in 8% / 11% / 20% of visit-runs; those visits drop out of
  its built-in averages.
- ProtGNN prototype check (K = 10, 20, 30; numbers for K = 20 here, the others are in `fair_report/k<K>/`): deleting the prototype subgraph lowers the explained prototype's activation only
  slightly more than deleting the same number of random nodes (+0.011 [+0.007, +0.016]) and lowers the class probability
  less than random nodes (-0.021 [-0.025, -0.018]); the subgraph (4.6 nodes, 2.2 removable) is not the evidence the class
  probability depends on.
- Controls (K = 20): built-in explanations of all four methods are close to uncorrelated with those of a model with
  re-drawn weights (Spearman -0.09 to -0.05); GNNExplainer is not (0.44-0.77). Explanations are stable under small noise
  (Spearman 0.96-1.00). The built-in explanations of GraphCare and ProtGNN agree across seeds no more than chance; those
  of CEI-GNN and GSAT do (lift 0.48 and 0.59).
- Relation shuffling (K = 20, trained models, inference only): dropping all relations changes the predicted class in 85%
  (GSAT), 55% (ProtGNN), 26% (CEI-GNN) and 14% (GraphCare) of visits. This is dependence on the relations, not proof of
  meaningful use.
- Failure types (K = 20): CEI-GNN is below GSAT with IG and Grad in all six diagnosis groups; the gap depends on the
  confidence of the two models and does not vanish in cells where both are equally confident.

## CEI-GNN recipe variants

`cei_recipe_variants/comparison.{md,csv}` lists the final model and the five variants side by side at each K (test macro-F1,
accuracy, confidence, explanation scores, relation-shuffle flip rate, seed agreement, weight randomisation). The variants
remove label smoothing (R1), label smoothing and weight decay (R2), all three of label smoothing, weight decay and DropEdge
(R3), weight decay only (R4) or DropEdge only (R5). Each variant has the same set of reports as the final models in
`cei_recipe_variants/<variant>/` (`summary`, `fair_report`, `audit_report`, `seed_agreement`, `similar_patients`,
`relation_shuffle`, `failure_types`). For the failure types the variant replaces CEI-GNN and the other three methods are the
final runs of the same K.

- Removing label smoothing raises every explanation score of CEI-GNN at all three K and lifts the ledger above GSAT's
  attention, at a test macro-F1 change of +0.003 / -0.009 / -0.008 (K = 10 / 20 / 30). Weight decay or DropEdge alone changes no score
  beyond seed noise.
- With label smoothing kept, the failure-type pattern equals the final model's (clearly below GSAT's IG and Grad in nearly
  every diagnosis group); without it, CEI-GNN is clearly below GSAT in fewer groups and confidence-matched cells.
- Removing DropEdge raises the dependence on the relations: dropping all relations changes the predicted class in 43-64% of
  visits for R3 and R5, against 26-36% for the final model and R1, R2, R4.
- The K = 20 variants were trained on another machine and received as run folders; the K = 10 and K = 30 variants were
  trained locally with unchanged repository code. All variant runs record the same explanation code identity. The
  target-binding sidecar hash (it embeds absolute paths) was compared with the local copy instead of the original machine's
  value; labels, cohort lineage and targets checksum were verified as for the final runs.

## Files

| File | Content |
|---|---|
| `final_methods/summary.{md,csv,json}` | explanation summary of the 36 runs, all K (`comparison/explain/summarize.py`) |
| `fair_report/k<K>/fair_report.{md,json}` | matched evidence budgets, ProtGNN prototype check, paired comparisons between methods (`comparison/explain/fair_report.py`) |
| `audit_report/k<K>/audit_report.{md,json}` | weight randomisation and perturbation stability (`comparison/explain/audit_report.py`) |
| `seed_agreement/k<K>/seed_agreement.{md,json}` | agreement of the top-ranked evidence across seeds (`comparison/explain/seed_agreement.py`) |
| `similar_patients/k<K>/similar_patients.{md,json}` | agreement across similar patients (`comparison/explain/similar_patients.py`) |
| `relation_shuffle/relation_shuffle.{md,json}` | relation shuffling of the trained models (`comparison/explain/relation_shuffle.py`) |
| `failure_types/k<K>/` | explanation score by kind of visit: `strata-report.md`, `strata-cells.csv`, `step1-report.md` (`analysis/failure_types/`) |
| `cei_recipe_variants/comparison.{md,csv}`, `cei_recipe_variants/<variant>/` | the five recipe variants (see above) |
| `predictive_confidence_test.md` | mean predicted-class probability of the four methods on the test fold, per K |
| `predictive_per_seed_test.md` | test-fold AUROC and AUPRC per seed, plain and patient-weighted |

All files are aggregates; no patient identifiers, rows or raw values. Not included: the per-visit records of the 36 runs
and the decoded patient examples (`comparison/explain/decode_examples.py`), which hold individual-level MIMIC-derived
data.

## Reproduce (repo root, `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3` before each command)

```
# the variants use the same commands with their own runs root; the failure types need a root that holds the variant and
# the three other methods side by side (symbolic links to the final runs are enough)
# explanations of the 36 final runs, then their summary (inference only, long)
-m comparison.explain.run --runs <runs root> --ks 10,20,30 --sample-patients 500 --audit-subjects 40 --jobs 4 --keep-going --out <new dir> --execute
-m comparison.explain.summarize --explanations <that dir> --out <new dir>

# per K: fairness, controls, agreement
-m comparison.explain.fair_report --explanations <that dir>/top<K> --out <new dir>
-m comparison.explain.audit_report --explanations <that dir>/top<K> --out <new dir>
-m comparison.explain.seed_agreement --explanations <that dir>/top<K> --graphs <graphs.jsonl> --out <new dir>
-m comparison.explain.similar_patients --explanations <that dir>/top<K> --graphs <graphs.jsonl> --runs <runs root> --out <new dir>

# relation shuffling of the trained models
-m comparison.explain.relation_shuffle run --runs <runs root> --out <probes dir> --sample-patients 500 --draws 3 --execute
-m comparison.explain.relation_shuffle report --probes <probes dir> --out <new dir>

# failure types, all K (per-visit tables go to <work>/data, not committed)
-m analysis.failure_types.run --explanations <that dir> --summary <summary dir> --runs <runs root> --work <new dir>
```
