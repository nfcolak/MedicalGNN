"""Primary-diagnosis (seq_num == 1) target producer, policy primary_dx_v3.

Copy of the local_labels_v2 build() logic with the same hash guards against the
historical binding, the same frozen 30-class order and the same inherited
category reconstruction. Differences: only the stay's seq_num == 1 diagnosis can
produce a label, and three category overrides are applied after the inherited
map. Git-ignored inputs are resolved from --source-root; tracked inputs
(data/s1_clean/merge_ed.py) come from this repo.
"""
from pathlib import Path
import json
import pandas as pd
from data.s3_labels import icd_mapping
from data.s3_labels.icd_mapping import ICD_MAPPING_POLICY, load_icd_map, normalize_icd10
from data.s3_labels.labels import REPO, sha256, _load_labels, _load_disease_merges, _json_save
from data.s3_labels.local_labels_v2 import literal_ids

GRAPH_REL = 'comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2'
LABELS_REL = 'comparison/standardized/native_inputs/protgsat_snapshot_v1/contract.json'
HISTORICAL_REL = 'comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_v1/binding_manifest.json'

LABEL_POLICY_VERSION = 'primary_dx_v3'
# Applied after the inherited category map, on usable seq_num == 1 rows only.
OVERRIDES = {
    'cat3_to_label': {'S76': 'Limb injury or pain'},
    'cat3_icd10_prefix_to_label': {'T81': {'prefix': 'T814', 'label': 'Skin or soft-tissue infection',
                                           'otherwise': 'no class'}},
    'cat3_no_class': ['I25'],
}


def _apply_overrides(frame, labels):
    label = frame.label.copy()
    for cat3, target in OVERRIDES['cat3_to_label'].items():
        if target not in labels:
            raise ValueError('Override label outside frozen classes: ' + target)
        label[frame.cat3 == cat3] = target
    for cat3, rule in OVERRIDES['cat3_icd10_prefix_to_label'].items():
        if rule['label'] not in labels:
            raise ValueError('Override label outside frozen classes: ' + rule['label'])
        in_cat = frame.cat3 == cat3
        label[in_cat] = ''
        label[in_cat & frame.icd10_code.str.startswith(rule['prefix'])] = rule['label']
    for cat3 in OVERRIDES['cat3_no_class']:
        label[frame.cat3 == cat3] = ''
    return label


def build(output, graph_root=None, source_root=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_root = Path(source_root if source_root is not None else REPO).resolve()
    raw_root = source_root / 'data/raw/Original CSVs'
    labels_path = source_root / LABELS_REL
    graph_root = Path(graph_root if graph_root is not None else source_root / GRAPH_REL).resolve()
    from data.s2_events.repair_metadata import LOCK, PENDING, validate_index_ready
    if (graph_root / PENDING).exists() or (graph_root / LOCK).exists():
        raise ValueError('Graph/index metadata is locked or repair is incomplete')
    historical_path = source_root / HISTORICAL_REL
    historical = json.loads(historical_path.read_text())
    source_hashes = {
        'raw_diagnosis_sha256': sha256(raw_root / 'diagnosis.csv'),
        'icd_mapping_sha256': sha256(raw_root / 'icd9_to_icd10_mapping.csv'),
        'merge_policy_sha256': sha256(REPO / 'data/s1_clean/merge_ed.py'),
    }
    if any(historical[k] != v for k, v in source_hashes.items()):
        raise ValueError('Original ontology source changed; refuse new-outcome reconstruction')
    labels = _load_labels(labels_path)
    graph_manifest_hash = sha256(graph_root / 'manifest.json')
    gm = json.loads((graph_root / 'manifest.json').read_text())
    if gm['reference_class_order_only'] != labels or historical['labels'] != labels:
        raise ValueError('Frozen graph/class order mismatch')
    if gm['status'] == 'index_ready':
        validate_index_ready(graph_root, gm)
        binding_scope = 'cohort_and_index_only'
        graph_hash = None
    elif gm['status'] == 'completed':
        graph_hash = sha256(graph_root / 'graphs.jsonl')
        if graph_hash != gm['graphs_sha256']:
            raise ValueError('Graph bytes differ from their manifest')
        binding_scope = 'completed_graph_and_cohort'
    else:
        raise ValueError('Graph requires completed or verified index_ready status')
    cohort_hash = sha256(graph_root / 'cohort.csv')
    if cohort_hash != gm['cohort_sha256']:
        raise ValueError('Cohort differs from graph/index metadata')
    d = pd.read_csv(raw_root / 'diagnosis.csv', dtype=str, keep_default_na=False)
    literal_ids(d)
    if not d.icd_version.isin(['9', '10']).all() or not d.seq_num.str.fullmatch(r'[1-9][0-9]*').all():
        raise ValueError('Invalid diagnosis version/priority')
    d['icd_code'] = d.icd_code.str.strip().str.upper()
    d['icd_title'] = d.icd_title.str.strip()
    d['seq_num'] = d.seq_num.astype('int64')
    titles = d[d.icd_version == '10'].drop_duplicates('icd_code').set_index('icd_code').icd_title.to_dict()
    crosswalk, _ = load_icd_map(raw_root / 'icd9_to_icd10_mapping.csv')
    d['icd10_code'] = [normalize_icd10(v, c, crosswalk) or '' for c, v in zip(d.icd_code, d.icd_version)]
    d['title'] = d.icd10_code.map(lambda c: titles.get(c, '').split(',')[0].strip())
    d['cat3'] = d.icd10_code.str[:3]
    label_usable = ~d.cat3.str[:1].isin(['V', 'W', 'X', 'Y', 'Z', 'R', ''])
    usable = label_usable & ~d.title.isin(['', 'nan'])  # ontology reconstruction only
    # Inherited merge_ed ontology reconstruction, not a new event-cohort vote.
    cat_title = d[usable].groupby('cat3').title.agg(lambda s: s.value_counts().index[0]).to_dict()
    merges = _load_disease_merges()
    category_map = {k: merges.get(v, v) for k, v in cat_title.items() if merges.get(v, v) in labels}
    d['label'] = d.cat3.map(category_map).fillna('')
    d['label'] = _apply_overrides(d, labels)
    # Primary diagnosis only: other seq_num rows never produce a label.
    primary = d[d.seq_num == 1]
    if primary.groupby(['subject_id', 'stay_id']).size().gt(1).any():
        raise ValueError('Stay has several seq_num == 1 diagnosis rows')
    selected = primary[label_usable.loc[primary.index] & primary.label.isin(labels)]
    groups = selected[['subject_id', 'stay_id', 'label']].copy()
    groups['n_selected_diseases'] = 1
    groups['target'] = groups.label.map({label: i for i, label in enumerate(labels)}).astype('int64')
    cohort = pd.read_csv(graph_root / 'cohort.csv', dtype=str, keep_default_na=False)
    literal_ids(cohort)
    if cohort.sample_id.duplicated().any() or cohort.stay_id.duplicated().any():
        raise ValueError('Duplicate cohort identity')
    if cohort.groupby('subject_id').split.nunique().gt(1).any():
        raise ValueError('Patient crosses canonical folds')
    # Cross-check ALL diagnosis pairs, not only labelled rows, before the two-key join.
    pairs = d[['subject_id', 'stay_id']].drop_duplicates()
    check = cohort[['subject_id', 'stay_id']].merge(pairs, on='stay_id', how='inner', suffixes=('_graph', '_diagnosis'))
    if check.subject_id_graph.ne(check.subject_id_diagnosis).any():
        raise ValueError('Graph/diagnosis subject mismatch')
    bound = cohort.merge(groups, on=['subject_id', 'stay_id'], how='left', validate='one_to_one')
    bound['target'] = bound.target.fillna(-1).astype('int64')
    bound['label'] = bound.label.fillna('')
    bound['n_selected_diseases'] = bound.n_selected_diseases.fillna(0).astype('int64')
    bound['label_status'] = bound.n_selected_diseases.map(lambda n: 'labelled' if n == 1 else 'no_frozen_class')
    matched = bound[bound.target >= 0]
    if (sha256(graph_root / 'manifest.json') != graph_manifest_hash
            or sha256(graph_root / 'cohort.csv') != cohort_hash):
        raise ValueError('Graph/index binding changed while producing targets')
    output.mkdir(parents=True)
    bound.to_csv(output / 'targets.csv', index=False)
    (output / 'targets.csv').chmod(0o600)
    _json_save(output / 'frozen_category_map.json', {'category_to_label': category_map, 'category_to_historical_title': cat_title,
        'overrides': OVERRIDES, 'source_hashes': source_hashes, 'labels': labels,
        'policy': 'Reconstructed inherited merge_ed.py ontology on the identical hash-pinned historical raw source, then category overrides; no event-cohort vote or top-N reselection.'})
    manifest = {'schema_version': 'event_graph_target_binding_v1', 'status': 'completed',
        'label_policy_version': LABEL_POLICY_VERSION, 'category_overrides': OVERRIDES,
        'graph_artifact': str(graph_root), 'graph_manifest_sha256': graph_manifest_hash,
        'graph_sha256': graph_hash, 'cohort_sha256': cohort_hash,
        'target_binding_scope': binding_scope,
        'labels': labels, 'labels_path': str(labels_path), 'labels_sha256': sha256(labels_path), **source_hashes,
        'source_code_sha256': sha256(Path(__file__)), 'historical_binding_sha256': sha256(historical_path),
        'icd_mapping_policy': ICD_MAPPING_POLICY,
        'icd_mapping_source_sha256': sha256(Path(icd_mapping.__file__)),
        'target_policy': "Exact literal subject_id+stay_id; the label is the frozen class of the stay's seq_num == 1 diagnosis only (after overrides S76->'Limb injury or pain', T81 only T814->'Skin or soft-tissue infection', I25 no class); an unusable or unmapped primary code gets target=-1. The label comes from the primary code's 3-character category; a missing ICD-10 title for the exact mapped code does not block it. ICD strings preserve leading zeros.",
        'counts': {'cohort_samples': len(bound), 'matched_samples': len(matched), 'unmatched_samples': int((bound.target < 0).sum()),
            'status_counts': {str(k): int(v) for k, v in bound.label_status.value_counts().items()},
            'by_split': {split: {'samples': int((matched.split == split).sum()), 'patients': int(matched.loc[matched.split == split, 'subject_id'].nunique())} for split in ['train', 'validation', 'test']},
            'class_counts': {str(i): int((matched.target == i).sum()) for i in range(len(labels))}},
        'limitations': ['temporal_clean=false; storetime is an availability proxy.', 'Inherited label ontology was historically frequency-derived, not train-only fitted.', 'Visit-level single-disease target policy, not reproduction of all legacy demographic/complete-case cohort filters.', 'Diagnosis targets never become graph covariates.', 'ICD-9 codes reach ICD-10 through an approximate GEM pick; only the 3-character category is used.', 'seq_num is MIMIC-IV-ED pseudo-order; 1 usually but not always the reason for the visit.'],
        'artifact_files': {name: sha256(output / name) for name in ['targets.csv', 'frozen_category_map.json']}}
    _json_save(output / 'binding_manifest.json', manifest)
    print(json.dumps(manifest['counts']), flush=True)
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=REPO,
                        help='Root holding git-ignored inputs (raw CSVs, event_inputs, native contract)')
    parser.add_argument('--graph-root', type=Path, default=None,
                        help='Explicit completed graph or repaired index_ready cohort to bind; never overwrites old targets')
    args = parser.parse_args()
    build(args.output, graph_root=args.graph_root, source_root=args.source_root)
