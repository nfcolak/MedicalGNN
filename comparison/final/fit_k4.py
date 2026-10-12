"""Fit the CEI-GNN v3 K4 state on the TRAIN sample of one graph and one top-k label count (plan mode unless --execute).

Equivalent of how cei_v3_state/v3_state/K4.json was made: cei.studies.cei_v3_study.fit_v3_state with the
default reader (read_v3_fit_inputs: select_top_labels(top_k_labels) -> resolve_train_dev_ids(fair_v10t: train = EVERY TRAIN-fold row
of the kept classes, no dev split, dev_limit None; --train-limit N = the old seeded sample; seed 1234) -> fit_preprocessing,
token_min_count 20). The state therefore matches the TRAIN ids of core.train protocol fair_v10t (the CEI adapter guard compares
train_sample_ids_sha256 with the run's ids and keeps refusing states fitted on the old full-TRAIN-minus-dev set). Writes <output-root>/v3_state/K4.json once, never overwrites.
"K4" = four PLE knots per measurement ("knots"), not the number of diagnosis labels ("labels", --top-k-labels).
One state belongs to exactly one top-k: the TRAIN sample differs per top-k, so fit one state per --top-k-labels
into its own output root (e.g. <root>/top17). The state records top_k_labels and train_sample_ids_sha256
(sorted TRAIN sample ids); the CEI adapter refuses a state of another top-k or sample.
Run from the repo root.
"""
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--artifact', required=True)
ap.add_argument('--targets', required=True)
ap.add_argument('--output-root', required=True)
ap.add_argument('--top-k-labels', type=int, default=10, help='number of diagnosis labels kept (default 10)')
ap.add_argument('--train-limit', type=int, default=None, help='default: every TRAIN-fold row of the kept classes (fair_v10t); N = seeded sample of N rows')
ap.add_argument('--execute', action='store_true')
a = ap.parse_args()
from cei.studies import cei_v3_study as study
KNOTS = 4
artifact, targets = Path(a.artifact).resolve(), Path(a.targets).resolve()
root = Path(a.output_root).resolve()
target_file = study.v3_state_path(root, KNOTS)
plan = {'status': 'not_executed' if not a.execute else 'executing', 'k': KNOTS, 'knots': KNOTS,
        'artifact': str(artifact),
        'artifact_exists': artifact.is_dir(), 'targets': str(targets), 'targets_exist': targets.is_file(),
        'writes': str(target_file), 'output_exists': target_file.exists(),
        'protocol': 'fair_v10t', 'train_limit': a.train_limit, 'dev_limit': None, 'sample_seed': study.SAMPLE_SEED,
        'top_k_labels': a.top_k_labels, 'token_min_count': 20}
print(json.dumps(plan, indent=2))
if a.execute:
    if not (plan['artifact_exists'] and plan['targets_exist']):
        sys.exit('missing artifact or targets')
    print(study.fit_v3_state(root, KNOTS, reader=lambda: study.read_v3_fit_inputs(
        artifact, targets, top_k_labels=a.top_k_labels, train_limit=a.train_limit, dev_limit=None)))
