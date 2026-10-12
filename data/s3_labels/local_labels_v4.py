"""Primary-diagnosis (seq_num == 1) target producer, policy primary_dx_v4.

Copy of local_labels_v3 with the same hash guards, frozen 30-class order, inherited
category ontology and category overrides. What differs from v3:

* ICD-9 -> ICD-10 labelling uses data/s3_labels/icd_mapping_v2 (reviewed tie-break
  overrides, 'gem_override_v1_stem98short'); codes without a defensible single target
  map to None and get no label (label_reason 'unmapped_icd9'). The inherited category
  ontology (cat3 -> frozen class) is still reconstructed with the OLD GEM pick so the
  class set and category map stay exactly the v3 ones; only which 3-char category a
  diagnosis code lands in changes.
* The Top-10 TRAIN class list is pinned and asserted against the recomputed one.
* --restrict-stays: stays outside the file get target -1, label_status
  'outside_common_stays' (all arms then share one stay set).
* targets.csv gets label_reason (LAB-19); frozen_category_map.json and the manifest
  carry the effective category_to_label (LAB-14) and class_display_names (cosmetic,
  class strings themselves stay frozen).
"""
from pathlib import Path
import json
import sys
import pandas as pd
from data.s3_labels import icd_mapping, icd_mapping_v2
from data.s3_labels.icd_mapping import ICD_MAPPING_POLICY, load_icd_map, normalize_icd10
from data.s3_labels.icd_mapping_v2 import (ICD_MAPPING_POLICY_V2, load_icd_map_v2, normalize_icd10_v2,
                                           overrides_sha256)
from data.s3_labels.labels import REPO, sha256, _load_labels, _load_disease_merges, _json_save
from data.s3_labels.local_labels_v2 import literal_ids
from data.s3_labels.local_labels_v3 import OVERRIDES, _apply_overrides

LABEL_POLICY_VERSION = 'primary_dx_v4'
# Tracked inputs (this repo): the frozen 30-class order + historical source hashes, and the CMS GEM table.
LABEL_REFERENCE_REL = 'data/reference/label_reference_v1.json'
ICD_MAPPING_REL = 'data/reference/icd9_to_icd10_mapping.csv'
# Cosmetic names for tables; class strings in targets/contract stay frozen.
CLASS_DISPLAY_NAMES = {
    'Hypokalemia': 'Fluid, electrolyte or acid-base disorder (E87)',
    'Non-ST elevation (NSTEMI) myocardial infarction': 'Acute myocardial infarction (I21)',
}
# fair_v3 TRAIN Top-10 (exact frozen class strings), alphabetical.
PINNED_TOP10 = [
    'Back or spine pain', 'Diabetes mellitus', 'GI bleed', 'Head injury', 'Heart failure',
    'Limb injury or pain', 'Lower respiratory disease', 'Major depressive disorder',
    'Skin or soft-tissue infection', 'UTI or pyelonephritis',
]


def _train_top10(bound, labels, k=10):
    """Top-k TRAIN classes by count, ties by class index (as select_top_labels)."""
    train = bound[(bound.split == 'train') & (bound.target >= 0)]
    counts = train.target.value_counts().to_dict()
    ranked = sorted(counts, key=lambda t: (-counts[t], t))[:k]
    return sorted(labels[t] for t in ranked), {labels[t]: int(c) for t, c in counts.items()}


def _read_restrict(path):
    ids = {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}
    if not ids:
        raise ValueError('--restrict-stays file has no stay ids')
    return ids


def _effective_category_map(category_map):
    """Category map with OVERRIDES applied; the T81 prefix rule stays a separate field."""
    effective = dict(category_map)
    for cat3, target in OVERRIDES['cat3_to_label'].items():
        effective[cat3] = target
    for cat3 in OVERRIDES['cat3_icd10_prefix_to_label']:
        effective.pop(cat3, None)
    for cat3 in OVERRIDES['cat3_no_class']:
        effective.pop(cat3, None)
    prefix_rules = {cat3: {'prefix': r['prefix'], 'label': r['label']}
                    for cat3, r in OVERRIDES['cat3_icd10_prefix_to_label'].items()}
    return effective, prefix_rules


def _label_from_effective(frame, effective, prefix_rules):
    label = frame.cat3.map(effective).fillna('')
    for cat3, rule in prefix_rules.items():
        hit = (frame.cat3 == cat3) & frame.icd10_code.str.startswith(rule['prefix'])
        label[hit] = rule['label']
    return label


def _primary_reason(row_frame, labels):
    """label_reason (LAB-19) for each seq_num == 1 row."""
    first = row_frame.cat3.str[:1]
    reason = pd.Series('no_class', index=row_frame.index)
    reason[(row_frame.label_base != '') & (row_frame.label == '')] = 'override_none'
    reason[first.isin(['V', 'W', 'X', 'Y'])] = 'external_cause'
    reason[first == 'Z'] = 'z_code'
    reason[first == 'R'] = 'r_symptom'
    reason[(row_frame.icd10_code == '') & (row_frame.icd_version == '9')] = 'unmapped_icd9'
    reason[(row_frame.icd10_code == '') & (row_frame.icd_version == '10')] = 'empty_code'
    reason[row_frame.usable_label & row_frame.label.isin(labels)] = 'labelled'
    return reason


def build(output, graph_root, raw_root, restrict_stays=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw_root = Path(raw_root).resolve()
    reference_path = REPO / LABEL_REFERENCE_REL
    icd_mapping_path = REPO / ICD_MAPPING_REL
    graph_root = Path(graph_root).resolve()
    from data.s2_events.repair_metadata import LOCK, PENDING, validate_index_ready
    if (graph_root / PENDING).exists() or (graph_root / LOCK).exists():
        raise ValueError('Graph/index metadata is locked or repair is incomplete')
    historical = json.loads(reference_path.read_text())
    source_hashes = {
        'raw_diagnosis_sha256': sha256(raw_root / 'diagnosis.csv'),
        'icd_mapping_sha256': sha256(icd_mapping_path),
        'merge_policy_sha256': sha256(REPO / 'data/s1_clean/merge_ed.py'),
    }
    if any(historical[k] != v for k, v in source_hashes.items()):
        raise ValueError('Original ontology source changed; refuse new-outcome reconstruction')
    labels = _load_labels(reference_path)
    missing = [c for c in PINNED_TOP10 + list(CLASS_DISPLAY_NAMES) if c not in labels]
    if missing:
        raise ValueError('Pinned/display class outside frozen classes: ' + repr(missing))
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

    # Inherited ontology: reconstructed with the ORIGINAL GEM pick (identical to v3), so the
    # class set / category map cannot drift with the new mapping.
    crosswalk, _ = load_icd_map(icd_mapping_path)
    old_icd10 = pd.Series([normalize_icd10(v, c, crosswalk) or '' for c, v in zip(d.icd_code, d.icd_version)],
                          index=d.index)
    old_cat3 = old_icd10.str[:3]
    old_title = old_icd10.map(lambda c: titles.get(c, '').split(',')[0].strip())
    old_usable = ~old_cat3.str[:1].isin(['V', 'W', 'X', 'Y', 'Z', 'R', '']) & ~old_title.isin(['', 'nan'])
    cat_title = (pd.DataFrame({'cat3': old_cat3[old_usable], 'title': old_title[old_usable]})
                 .groupby('cat3').title.agg(lambda s: s.value_counts().index[0]).to_dict())
    merges = _load_disease_merges()
    category_map = {k: merges.get(v, v) for k, v in cat_title.items() if merges.get(v, v) in labels}

    # Labelling mapping: v2 (reviewed overrides); None = no defensible single ICD-10 target.
    crosswalk2, approx2 = load_icd_map_v2(icd_mapping_path)
    d['icd10_code'] = [normalize_icd10_v2(v, c, crosswalk2) or '' for c, v in zip(d.icd_code, d.icd_version)]
    d['cat3'] = d.icd10_code.str[:3]
    d['usable_label'] = ~d.cat3.str[:1].isin(['V', 'W', 'X', 'Y', 'Z', 'R', ''])
    d['label_base'] = d.cat3.map(category_map).fillna('')
    d['label'] = _apply_overrides(d.assign(label=d.label_base), labels)
    effective_map, prefix_rules = _effective_category_map(category_map)
    if not _label_from_effective(d, effective_map, prefix_rules).equals(d.label):
        raise ValueError('Effective category_to_label does not reproduce the override labels')

    # Primary diagnosis only: other seq_num rows never produce a label.
    primary = d[d.seq_num == 1].copy()
    if primary.groupby(['subject_id', 'stay_id']).size().gt(1).any():
        raise ValueError('Stay has several seq_num == 1 diagnosis rows')
    primary['label_reason'] = _primary_reason(primary, labels)
    selected = primary[primary.usable_label & primary.label.isin(labels)]
    groups = selected[['subject_id', 'stay_id', 'label']].copy()
    groups['n_selected_diseases'] = 1
    groups['target'] = groups.label.map({label: i for i, label in enumerate(labels)}).astype('int64')
    reasons = primary[['subject_id', 'stay_id', 'label_reason']]
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
    bound = bound.merge(reasons, on=['subject_id', 'stay_id'], how='left', validate='one_to_one')
    bound['target'] = bound.target.fillna(-1).astype('int64')
    bound['label'] = bound.label.fillna('')
    bound['n_selected_diseases'] = bound.n_selected_diseases.fillna(0).astype('int64')
    bound['label_reason'] = bound.label_reason.fillna('no_diagnosis')
    bound['label_status'] = bound.n_selected_diseases.map(lambda n: 'labelled' if n == 1 else 'no_frozen_class')

    # Pinned Top-10 must equal the recomputed TRAIN Top-10 (before any restriction).
    top10, train_counts = _train_top10(bound, labels)
    if top10 != sorted(PINNED_TOP10):
        raise ValueError(f'Recomputed TRAIN Top-10 differs from the pinned list: {top10}')
    unrestricted_counts = {'labelled_stays': int((bound.target >= 0).sum()), 'train_class_counts': train_counts}

    restriction = None
    if restrict_stays is not None:
        keep = _read_restrict(restrict_stays)
        absent = keep - set(cohort.stay_id)
        if absent:
            raise ValueError(f'--restrict-stays has {len(absent)} stay ids outside the cohort, e.g. {sorted(absent)[:3]}')
        outside = ~bound.stay_id.isin(keep)
        dropped_labelled = int((outside & (bound.target >= 0)).sum())
        bound.loc[outside, ['target', 'n_selected_diseases']] = [-1, 0]
        bound.loc[outside, ['label', 'label_reason']] = ['', 'outside_common_stays']
        bound.loc[outside, 'label_status'] = 'outside_common_stays'
        r_top10, r_counts = _train_top10(bound, labels)
        if r_top10 != sorted(PINNED_TOP10):
            print('WARNING: restricted TRAIN Top-10 differs from the pinned list: ' + repr(r_top10),
                  file=sys.stderr, flush=True)
        restriction = {'restrict_stays_file': str(restrict_stays), 'restrict_stays_sha256': sha256(Path(restrict_stays)),
                       'stays_in_file': len(keep), 'cohort_stays_outside': int(outside.sum()),
                       'labelled_stays_dropped': dropped_labelled,
                       'restricted_train_class_counts': r_counts,
                       'restricted_top10_equals_pinned': r_top10 == sorted(PINNED_TOP10)}

    matched = bound[bound.target >= 0]
    if (sha256(graph_root / 'manifest.json') != graph_manifest_hash
            or sha256(graph_root / 'cohort.csv') != cohort_hash):
        raise ValueError('Graph/index binding changed while producing targets')
    output.mkdir(parents=True)
    bound.to_csv(output / 'targets.csv', index=False)
    (output / 'targets.csv').chmod(0o600)
    icd_policy_fields = {
        'icd_mapping_policy': ICD_MAPPING_POLICY_V2,
        'icd_overrides_sha256': overrides_sha256(),
        'icd_mapping_source_sha256': sha256(Path(icd_mapping_v2.__file__)),
        'ontology_icd_mapping_policy': ICD_MAPPING_POLICY,
        'ontology_icd_mapping_source_sha256': sha256(Path(icd_mapping.__file__)),
    }
    _json_save(output / 'frozen_category_map.json', {
        'category_to_label': effective_map, 'category_prefix_to_label': prefix_rules,
        'no_class_categories': OVERRIDES['cat3_no_class'],
        'inherited_category_to_label': category_map, 'category_to_historical_title': cat_title,
        'overrides': OVERRIDES, 'source_hashes': source_hashes, 'labels': labels,
        'policy': 'Inherited merge_ed.py ontology reconstructed on the identical hash-pinned historical raw source with the original GEM pick (class set frozen), category overrides applied into category_to_label (T81 prefix rule separate); diagnosis codes reach their 3-char category via icd_mapping_v2; no event-cohort vote or top-N reselection.'})
    manifest = {'schema_version': 'event_graph_target_binding_v1', 'status': 'completed',
        'label_policy_version': LABEL_POLICY_VERSION, 'category_overrides': OVERRIDES,
        'category_to_label': effective_map, 'category_prefix_to_label': prefix_rules,
        'class_display_names': CLASS_DISPLAY_NAMES, 'pinned_top10': PINNED_TOP10,
        'pinned_top10_recomputed_equal': True,
        'graph_artifact': str(graph_root), 'graph_manifest_sha256': graph_manifest_hash,
        'graph_sha256': graph_hash, 'cohort_sha256': cohort_hash,
        'target_binding_scope': binding_scope,
        'labels': labels, 'label_reference_path': LABEL_REFERENCE_REL, 'label_reference_sha256': sha256(reference_path),
        'icd_mapping_path': ICD_MAPPING_REL, **source_hashes,
        'source_code_sha256': sha256(Path(__file__)),
        **icd_policy_fields,
        'restriction': restriction,
        'target_policy': "Exact literal subject_id+stay_id; the label is the frozen class of the stay's seq_num == 1 diagnosis only (after overrides S76->'Limb injury or pain', T81 only T814->'Skin or soft-tissue infection', I25 no class); an unusable or unmapped primary code gets target=-1 (label_reason says why). ICD-9 reaches ICD-10 through icd_mapping_v2 (reviewed tie-break overrides; no defensible single target = no label). The label comes from the primary code's 3-character category; a missing ICD-10 title for the exact mapped code does not block it. ICD strings preserve leading zeros. Stays outside --restrict-stays get target=-1, label_status outside_common_stays.",
        'counts': {'cohort_samples': len(bound), 'matched_samples': len(matched), 'unmatched_samples': int((bound.target < 0).sum()),
            'status_counts': {str(k): int(v) for k, v in bound.label_status.value_counts().items()},
            'label_reason_counts': {str(k): int(v) for k, v in bound.label_reason.value_counts().items()},
            'unrestricted': unrestricted_counts,
            'by_split': {split: {'samples': int((matched.split == split).sum()), 'patients': int(matched.loc[matched.split == split, 'subject_id'].nunique())} for split in ['train', 'validation', 'test']},
            'class_counts': {str(i): int((matched.target == i).sum()) for i in range(len(labels))}},
        'limitations': ['temporal_clean=false; storetime is an availability proxy.', 'Inherited label ontology was historically frequency-derived, not train-only fitted.', 'Visit-level single-disease target policy, not reproduction of all legacy demographic/complete-case cohort filters.', 'Diagnosis targets never become graph covariates.', 'ICD-9 codes reach ICD-10 through an approximate GEM pick with reviewed overrides; only the 3-character category is used.', 'seq_num is MIMIC-IV-ED pseudo-order; 1 usually but not always the reason for the visit.'],
        'artifact_files': {name: sha256(output / name) for name in ['targets.csv', 'frozen_category_map.json']}}
    _json_save(output / 'binding_manifest.json', manifest)
    print(json.dumps(manifest['counts']), flush=True)
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='Folder with the decompressed MIMIC-IV-ED / hosp CSVs (diagnosis.csv is read)')
    parser.add_argument('--graph-root', type=Path, required=True,
                        help='Completed graph or repaired index_ready cohort to bind; never overwrites old targets')
    parser.add_argument('--restrict-stays', type=Path, default=None,
                        help='File with one stay_id per line; other cohort stays get target -1 / outside_common_stays')
    args = parser.parse_args()
    build(args.output, graph_root=args.graph_root, raw_root=args.raw_root, restrict_stays=args.restrict_stays)
