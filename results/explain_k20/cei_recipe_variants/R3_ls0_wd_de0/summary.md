# Explanation summary, fold: test (report fold)

Protocol: fair_v10t. Intervention contract version 6, record schema 3. Mean ± SD over seeds of the per-run visit means; fidelity numbers are drops of the predicted-class probability; `*_mean_change` is the curve AUC divided by the explained fraction; the headline is the difference of `*_mean_change` to random removal. No identifiers.

Cohort sample: size: 1000; seed: 1234; cap: null.

Explanation code: 411ae930d115 (3 runs, git head 6782d4b17e9599c92d990fd5f9cf382ba55d625f).

Run set: checked against the driver plan.

## K = 20 · fold: test (report fold) · intervention contract v6

| method | source | runs | visits | explained | failed | deletion_mean_change_vs_random | insertion_mean_change_vs_random | fidelity_plus | fidelity_minus | deletion_mean_change | insertion_mean_change | deletion_auc | insertion_auc | explained_fraction | mass90_sparsity | one_minus_support_fraction | deletion_auc_vs_random | insertion_auc_vs_random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cei_gnn_mp2 | GNNExplainer | 3 | 4035 | 4035 | 0 | 0.2450 ± 0.0056 | -0.2163 ± 0.0025 | 0.4661 ± 0.0097 | 0.0407 ± 0.0013 | 0.5079 ± 0.0060 | 0.0471 ± 0.0015 | 0.5079 ± 0.0060 | 0.0471 ± 0.0015 | 1.0000 ± 0.0000 | 0.1018 ± 0.0004 | 0.0000 ± 0.0000 | 0.2450 ± 0.0056 | -0.2163 ± 0.0025 |
| cei_gnn_mp2 | GradExplainer | 3 | 4035 | 4035 | 0 | 0.2856 ± 0.0056 | -0.2035 ± 0.0058 | 0.5409 ± 0.0163 | 0.0653 ± 0.0100 | 0.5485 ± 0.0072 | 0.0599 ± 0.0039 | 0.5485 ± 0.0072 | 0.0599 ± 0.0039 | 1.0000 ± 0.0000 | 0.4089 ± 0.0115 | 0.0000 ± 0.0000 | 0.2856 ± 0.0056 | -0.2035 ± 0.0058 |
| cei_gnn_mp2 | IntegratedGradExplainer (clinical baseline) | 3 | 4035 | 4035 | 0 | 0.2927 ± 0.0076 | -0.2068 ± 0.0054 | 0.5571 ± 0.0152 | 0.0554 ± 0.0098 | 0.5550 ± 0.0076 | 0.0570 ± 0.0043 | 0.5540 ± 0.0075 | 0.0569 ± 0.0043 | 0.9983 ± 0.0000 | 0.3220 ± 0.0080 | 0.0017 ± 0.0000 | 0.2922 ± 0.0076 | -0.2065 ± 0.0053 |
| cei_gnn_mp2 | builtin | 3 | 4035 | 4035 | 0 | 0.3221 ± 0.0020 | -0.2366 ± 0.0033 | 0.5800 ± 0.0178 | 0.0025 ± 0.0033 | 0.5850 ± 0.0044 | 0.0267 ± 0.0013 | 0.5850 ± 0.0044 | 0.0267 ± 0.0013 | 1.0000 ± 0.0000 | 0.7109 ± 0.0244 | 0.0000 ± 0.0000 | 0.3221 ± 0.0020 | -0.2366 ± 0.0033 |

### Visit accounting

| method | source | visits | explained | unsupported | explainer failed | fidelity failed | built-in unavailable | no eligible evidence | with random reference | IG complete | constant importance | tied at cut | largest tie group |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cei_gnn_mp2 | GNNExplainer | 4035 | 4035 | 0 | 0 | 0 | 0 | 0 | 4035 |  | 0.000 | 0.000 | 1.00 |
| cei_gnn_mp2 | GradExplainer | 4035 | 4035 | 0 | 0 | 0 | 0 | 0 | 4035 |  | 0.000 | 0.000 | 1.00 |
| cei_gnn_mp2 | IntegratedGradExplainer (clinical baseline) | 4035 | 4035 | 0 | 0 | 0 | 0 | 0 | 4035 | 0.999 | 0.000 | 0.000 | 1.02 |
| cei_gnn_mp2 | builtin | 4035 | 4035 | 0 | 0 | 0 | 0 | 0 | 4035 |  | 0.000 | 0.000 | 1.00 |

### Paired comparison (visits every source of the method has)

| method | source | runs | paired visits | deletion_mean_change_vs_random | insertion_mean_change_vs_random | fidelity_plus | fidelity_minus | deletion_mean_change | insertion_mean_change | deletion_auc | insertion_auc | explained_fraction | mass90_sparsity | one_minus_support_fraction | deletion_auc_vs_random | insertion_auc_vs_random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cei_gnn_mp2 | GNNExplainer | 3 | 4035 | 0.2450 ± 0.0056 | -0.2163 ± 0.0025 | 0.4661 ± 0.0097 | 0.0407 ± 0.0013 | 0.5079 ± 0.0060 | 0.0471 ± 0.0015 | 0.5079 ± 0.0060 | 0.0471 ± 0.0015 | 1.0000 ± 0.0000 | 0.1018 ± 0.0004 | 0.0000 ± 0.0000 | 0.2450 ± 0.0056 | -0.2163 ± 0.0025 |
| cei_gnn_mp2 | GradExplainer | 3 | 4035 | 0.2856 ± 0.0056 | -0.2035 ± 0.0058 | 0.5409 ± 0.0163 | 0.0653 ± 0.0100 | 0.5485 ± 0.0072 | 0.0599 ± 0.0039 | 0.5485 ± 0.0072 | 0.0599 ± 0.0039 | 1.0000 ± 0.0000 | 0.4089 ± 0.0115 | 0.0000 ± 0.0000 | 0.2856 ± 0.0056 | -0.2035 ± 0.0058 |
| cei_gnn_mp2 | IntegratedGradExplainer (clinical baseline) | 3 | 4035 | 0.2927 ± 0.0076 | -0.2068 ± 0.0054 | 0.5571 ± 0.0152 | 0.0554 ± 0.0098 | 0.5550 ± 0.0076 | 0.0570 ± 0.0043 | 0.5540 ± 0.0075 | 0.0569 ± 0.0043 | 0.9983 ± 0.0000 | 0.3220 ± 0.0080 | 0.0017 ± 0.0000 | 0.2922 ± 0.0076 | -0.2065 ± 0.0053 |
| cei_gnn_mp2 | builtin | 3 | 4035 | 0.3221 ± 0.0020 | -0.2366 ± 0.0033 | 0.5800 ± 0.0178 | 0.0025 ± 0.0033 | 0.5850 ± 0.0044 | 0.0267 ± 0.0013 | 0.5850 ± 0.0044 | 0.0267 ± 0.0013 | 1.0000 ± 0.0000 | 0.7109 ± 0.0244 | 0.0000 ± 0.0000 | 0.3221 ± 0.0020 | -0.2366 ± 0.0033 |

### Caveats: cei_gnn_mp2 (K = 20)

- Intervention: node deletion in the graph record (intervention contract version 6): the record without the selected evidence nodes and their derived concept nodes, re-encoded with the run's own preprocessing and scored by the model's own forward; combo states: the medical:indicates edges of one (state node, rule_id) of a combo rule form a group; deleting any anchor deletes the group; when the latest index value of a combo token is deleted the rule is evaluated again on the next-latest value with the graph builder's rule check and the artifact's threshold table; an input missing from the record is reported and the group is deleted; a rule that holds on the surviving values gets its edges (and its state node when the record has none), so deleting a latest value that failed a rule can add a state the record did not have; prior visits the deletion emptied are dropped; prior visits empty in the record are kept; trajectory chains are not relinked; this is the record with the nodes deleted, not the record the graph builder would produce without the events: prior labs of an analyte whose index labs are all deleted stay (with the analyte and its knowledge nodes), and a baseline_of edge is not re-anchored; nodes are ranked by |importance|, ties at a cut averaged over seeded orders
- Built-in explanation: the exact additive accounting of the explained logit (node + half of each incident edge/pair contribution), ranked by |signed contribution|; the shares of negative eligible nodes and of mass on protected nodes are recorded; class-specific
- Explainers: gradient: GradExplainer and the clinical IG explain x = [numeric row | token vector]; node type, edges and edge metadata stay fixed; node importance = sum of |attribution| over every column of x; ig: clinical IG from the not-recorded baseline (token block zero, the pad embedding); per visit: from the start step count up the candidates until the relative completeness error is within the tolerance, else the largest candidate (completeness_met false); the vendored zero-baseline IG is kept in the records only; gnnexplainer: 50 epochs (paper 100-300, PyG default 100), seeded per visit, edge: size 0.005 * sum(sigmoid(M)), entropy 1.0 * mean; feature: size 1.0 * mean(sigmoid(F)) over the columns, entropy 0.1 * mean; node importance = mean of the incident edge masks; a fractional edge mask is not a removal: hard 0/1 masks equal deletion and keeping, soft values are compared by rank only; edge_mask_scope: within-visit pair contributions, the absence block and the node-local rows are not reachable by an edge mask; the message-passing weighted means make a common scale of a receiver's incoming masks partly invisible; ple_term: outside_ig: the network's own hard-gated PLE term (has_value > 0), recomputed from the explained features; inside_ig: inside IG only: the PLE basis computed with has_value treated as 1 and multiplied by the explained has_value (equal to the network's term at has_value 0 and 1, linear in between)
- a source whose importance is zero on every eligible evidence node is counted as no_eligible_evidence and left out of the means
- an edge mask (0) deletes the edge from the edge rows and from every message-passing layer (weighted means, relation groups, hub max, mp_dedup copy split and mirror rule); a fractional mask scales the edge once, and under a weighted mean a common scale of a receiver's incoming masks is partly invisible. Within-visit pair contributions, the absence block and the node-local rows are not reachable by an edge mask; a typed reverse edge of a bidirectional view is a separate edge.
- a node that votes against the class is ranked by its magnitude
- Replay rule that passed: stored_predictions_tolerance
- Built-in explanation source: method: cei_gnn_mp2; algorithm: exact additive accounting of the logit (native_importance); settings: ranking: |signed contribution|; class_specific: true
- Settings: curve_points: 10; random_repeats: 50; tie_draws: 10; random_seed: 1234; gnnexplainer_epochs: 50; gnnexplainer_seed_rule: per visit: the first 8 hex digits of sha256('<random_seed>:<sample_id>') as an integer; the same seed drives the tie draws and the random reference; gnnexplainer_feature_penalty: mean; gnnexplainer_node_reduction: mean; vendored_ig_steps: 32; ig: enabled: true; baseline: not_recorded; tolerance: 0.05; step_candidates: 8; 16; 32; 64; 128; 256; escalation: per visit: from the start step count up the candidates until the relative completeness error is within the tolerance, else the largest candidate (completeness_met false); pilot: 30 decision-fold patients, one visit each; start_steps: 32; fidelity_skipped_sources: IntegratedGradExplainerZeroBaseline; true_class_explanations_for_misclassified: true; audit: subjects: 0; stability_draws: 2; stability_sigma: 0.05
- Cohort sample: size: 1000; seed: 1234; cap: null

Direction: higher is better for fidelity_plus and the deletion columns; lower is better for fidelity_minus and the insertion columns; a negative insertion difference to random (`*_vs_random`) is better than random, a positive deletion difference is. The headline is the width-matched difference to random (`deletion_mean_change_vs_random`, `insertion_mean_change_vs_random`).

Left out of every table: `IntegratedGradExplainerZeroBaseline` (vendored zero-baseline integrated gradients (no completeness check)); it stays in the records.
