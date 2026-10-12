"""T2 (SC4, SC5): graph manifests carry the threshold semantics and the inherited cutoff rule;
verify_artifact re-derives the visit age.

Synthetic inputs only: a tiny event artifact (the fixture of test_clinical_data_integrity), tiny
raw CSVs, the rule CSV. No patient data, no real event index, no test fold.
"""
import csv
import json
import os
import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

from core.contracts import VISIT_MEMBERSHIP_FILENAME
from data.s2_events import repair_metadata
from data.s4_graph import build, build_delay
from data.s4_graph.build import verify_artifact
from data.s4_graph.graph import build_graph_with_visit_membership, threshold_semantics
from test_clinical_data_integrity import event_artifact, write_csv
from test_v5_threshold_states import SAMPLE, THRESHOLDS, FakeStore, potassium

THRESHOLDS_CSV = Path(__file__).resolve().parents[1] / 'data/s2_events/knowledge_thresholds_v2.csv'

OLD_DECISION_POLICY = ('Inherited v1 cutoff, unchanged: first nonempty recorded lab result of the '
                       'encounter. Triage evidence is admitted under an unverified arrival-time '
                       'assumption. Checking intime <= cutoff verifies only proxy bounds.')
LAB_SCOPE_FIELDS = {
    'cutoff_rule': 'representable-results-only',
    'lab_scope': 'representable-v5',
    'decision_policy': 'First REPRESENTABLE recorded lab result (synthetic inherited text).',
}
# What the pre-v5 event manifests really look like (checked against the three real ones below):
# a lab_scope, a decision_policy, no cutoff_rule.
OLDER_LAB_SCOPE_FIELDS = {
    'lab_scope': 'all-numeric-known-unit',
    'decision_policy': 'Older inherited text that a v4 graph manifest must not copy.',
}
# env MEDGNN_EVENT_INPUTS: directory with the real older event manifests; unset or missing -> those tests skip
EVENT_INPUTS = Path(os.environ.get('MEDGNN_EVENT_INPUTS') or 'MEDGNN_EVENT_INPUTS-not-set')
OLDER_REAL_MANIFESTS = ('first_recorded_lab_all_visits_v2', 'first_recorded_lab_all_visits_v2_max6',
                        'first_recorded_lab_plus60_max6_20261005')
# Subject 1: 40 + 2020 - 2018 = 42. Subject 2: 90 + 2020 - 2015 = 95, capped at 91.
EXPECTED_AGES = {'1': 42.0, '2': 91.0}


def old_delay_policy(manifest):
    """The pre-T2 build_delay text, spelled out so 'byte-for-byte' is checked against a literal."""
    return ('Cutoff = min(first nonempty recorded lab result storetime + %d min, ED outtime), '
            'inherited from the first_lab_plus_delay artifact (inherits_index_sha256 %s); every '
            'stay shares the same fixed delay. Triage evidence is admitted under an unverified '
            'arrival-time assumption. Checking intime <= cutoff verifies only proxy bounds. Not an '
            'early-triage time.' % (manifest['delay_minutes'], manifest['inherits_index_sha256']))


# ------------------------------------------------------------------ synthetic build inputs
def make_inherited(root, lab_scope):
    """Index-ready event artifact (8 samples, 2 subjects). `lab_scope` adds the v5 manifest keys."""
    inherited = event_artifact(root, ['1', '2'])
    # The shared fixture reuses one source id for every event; the readback needs unique ones.
    db = sqlite3.connect(inherited / 'events.sqlite')
    db.execute('UPDATE events SET source=id')
    db.commit()
    db.close()
    manifest_path = inherited / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(cutoff_policy='first_lab_plus_delay', delay_minutes=60,
                    inherits_index_sha256='a' * 64, cutoff_clipped_at_outtime=0)
    if lab_scope == 'older':
        manifest.update(OLDER_LAB_SCOPE_FIELDS)
    elif lab_scope:
        manifest.update(LAB_SCOPE_FIELDS)
    manifest_path.write_text(json.dumps(manifest))
    repair_metadata.repair(inherited, execute=True, offline=True)
    return inherited


def make_raw(root, inherited, anchor_year_subjects=('1', '2')):
    root.mkdir()
    cohort = list(csv.DictReader((inherited / 'cohort.csv').open()))
    visits = {r['stay_id']: '2020-01-%02d 00:00:00' % (int(r['stay_id']) % 100 + 1) for r in cohort}
    write_csv(root / 'edstays.csv', ['subject_id', 'stay_id', 'intime', 'gender', 'race', 'arrival_transport'],
              [dict(subject_id=r['subject_id'], stay_id=r['stay_id'], intime=visits[r['stay_id']],
                    gender='F', race='unknown', arrival_transport='walk') for r in cohort])
    patients = {'1': ('40', '2018'), '2': ('90', '2015')}
    write_csv(root / 'patients.csv', ['subject_id', 'anchor_age', 'anchor_year'],
              [dict(subject_id=s, anchor_age=patients[s][0],
                    anchor_year=patients[s][1] if s in anchor_year_subjects else '')
               for s in ('1', '2')])
    fields = ['subject_id', 'stay_id', 'chiefcomplaint', 'acuity', 'temperature', 'heartrate',
              'resprate', 'o2sat', 'sbp', 'dbp']
    write_csv(root / 'triage.csv', fields,
              [dict(zip(fields, [r['subject_id'], r['stay_id'], 'pain', '2', '98.6', '80', '20',
                                 '99', '120', '70'])) for r in cohort])
    return root


def make_args(tmp_path, inherited, raw, name, level_source='csv', age_policy='anchor'):
    knowledge = write_csv(tmp_path / (name + '-knowledge.csv'),
                          ['source_token', 'relation', 'target_token', 'source_uri',
                           'source_version', 'reviewed_by'],
                          [dict(source_token='lab:1', relation='measures', target_token='knowledge:1',
                                source_uri='urn:fixture', source_version='1', reviewed_by='fixture')])
    return Namespace(output=tmp_path / name, inherit_from=inherited, raw_root=raw,
                     canonical=inherited / 'canonical_split.json', knowledge=knowledge,
                     thresholds=THRESHOLDS_CSV if level_source == 'row-range' else None,
                     with_diagnosis=False, max_nodes=100, max_edges=100, complaint_min_count=1,
                     limit=None, level_source=level_source, diagnosis_policy='v3',
                     age_policy=age_policy)


@pytest.fixture
def plain_inputs(tmp_path):
    inherited = make_inherited(tmp_path / 'plain', lab_scope=False)
    return tmp_path, inherited, make_raw(tmp_path / 'raw-plain', inherited)


@pytest.fixture
def v5_inputs(tmp_path):
    inherited = make_inherited(tmp_path / 'v5', lab_scope=True)
    return tmp_path, inherited, make_raw(tmp_path / 'raw-v5', inherited)


@pytest.fixture
def older_inputs(tmp_path):
    inherited = make_inherited(tmp_path / 'older', lab_scope='older')
    return tmp_path, inherited, make_raw(tmp_path / 'raw-older', inherited)


def read_manifest(args):
    return json.loads((args.output / 'manifest.json').read_text())


# ------------------------------------------------------------------ SC4: threshold_semantics
def test_row_range_manifest_carries_threshold_semantics(plain_inputs):
    tmp_path, inherited, raw = plain_inputs
    args = make_args(tmp_path, inherited, raw, 'rr', level_source='row-range')
    build.run(args)
    manifest = read_manifest(args)
    assert manifest['status'] == 'completed'
    assert manifest['threshold_semantics'] == threshold_semantics()
    assert manifest['threshold_semantics']['state_gate'] == 'level_agreement'
    assert manifest['threshold_semantics']['own_range_rule'] == 'both bounds present and lower < upper'


def test_csv_manifest_has_no_threshold_semantics(plain_inputs):
    tmp_path, inherited, raw = plain_inputs
    args = make_args(tmp_path, inherited, raw, 'csv')
    build.run(args)
    manifest = read_manifest(args)
    assert manifest['status'] == 'completed'
    assert 'threshold_semantics' not in manifest


# ------------------------------------------------------------------ SC4: inherited cutoff rule (F, build.py)
def test_build_copies_cutoff_rule_lab_scope_and_decision_policy_from_a_lab_scope_manifest(v5_inputs):
    tmp_path, inherited, raw = v5_inputs
    args = make_args(tmp_path, inherited, raw, 'f-v5')
    build.run(args)
    manifest = read_manifest(args)
    assert manifest['cutoff_rule'] == LAB_SCOPE_FIELDS['cutoff_rule']
    assert manifest['lab_scope'] == LAB_SCOPE_FIELDS['lab_scope']
    assert manifest['decision_policy'] == LAB_SCOPE_FIELDS['decision_policy']


def test_build_without_inherited_lab_scope_keeps_decision_policy_and_keys(plain_inputs):
    tmp_path, inherited, raw = plain_inputs
    args = make_args(tmp_path, inherited, raw, 'f-old')
    build.run(args)
    manifest = read_manifest(args)
    assert manifest['decision_policy'] == OLD_DECISION_POLICY
    assert not {'cutoff_rule', 'lab_scope', 'threshold_semantics'} & set(manifest)


def test_build_on_an_older_manifest_with_lab_scope_keeps_decision_policy_and_adds_no_keys(older_inputs):
    """R1-A: pre-v5 event manifests carry lab_scope and a decision_policy too; they are not v5 artifacts."""
    tmp_path, inherited, raw = older_inputs
    args = make_args(tmp_path, inherited, raw, 'f-older')
    build.run(args)
    manifest = read_manifest(args)
    assert manifest['status'] == 'completed'
    assert manifest['decision_policy'] == OLD_DECISION_POLICY
    assert not {'cutoff_rule', 'lab_scope', 'threshold_semantics'} & set(manifest)


# ------------------------------------------------------------------ R1-A: the shared v5 predicate
@pytest.mark.parametrize('manifest, expected', [
    ({}, False),
    ({'lab_scope': 'all-numeric-known-unit', 'decision_policy': 'x'}, False),
    ({'lab_scope': None, 'cutoff_rule': None}, False),
    ({'lab_scope': 'representable-v5'}, True),
    ({'cutoff_rule': 'representable-results-only'}, True),
    ({'cutoff_rule': 'representable-results-only', 'lab_scope': 'representable-v5'}, True),
])
def test_is_v5_event_manifest_is_true_only_for_the_v5_markers(manifest, expected):
    assert build.is_v5_event_manifest(manifest) is expected


@pytest.mark.parametrize('name', OLDER_REAL_MANIFESTS)
def test_the_real_older_event_manifests_are_not_v5_artifacts(name):
    path = EVENT_INPUTS / name / 'manifest.json'
    if not path.is_file():
        pytest.skip('real older event manifest not available: %s' % path)
    manifest = json.loads(path.read_text())  # read-only
    assert manifest.get('lab_scope') and manifest.get('decision_policy')  # the trap this guards
    assert build.is_v5_event_manifest(manifest) is False


# ------------------------------------------------------------------ SC4: inherited cutoff rule (L1/L3/L4, build_delay.py)
def delay_manifest(lab_scope):
    manifest = {'cutoff_policy': 'first_lab_plus_delay', 'delay_minutes': 60,
                'inherits_index_sha256': 'a' * 64, 'cutoff_clipped_at_outtime': 0}
    if lab_scope == 'older':
        manifest.update(OLDER_LAB_SCOPE_FIELDS)
    elif lab_scope:
        manifest.update(LAB_SCOPE_FIELDS)
    return manifest


def test_delay_policy_text_is_the_inherited_decision_policy_when_lab_scope_is_present():
    assert build_delay.policy_text(delay_manifest(True)) == LAB_SCOPE_FIELDS['decision_policy']


def test_delay_policy_text_without_lab_scope_is_the_old_text_byte_for_byte():
    manifest = delay_manifest(False)
    assert build_delay.policy_text(manifest) == old_delay_policy(manifest)


def test_delay_policy_text_on_an_older_manifest_with_lab_scope_is_the_old_text_byte_for_byte():
    manifest = delay_manifest('older')
    assert build_delay.policy_text(manifest) == old_delay_policy(manifest)


def test_delay_policy_text_on_the_real_plus60_manifest_is_the_pre_change_text():
    """R1-A: the real first_recorded_lab_plus60_max6_20261005 manifest (read-only) still yields the legacy text."""
    path = EVENT_INPUTS / 'first_recorded_lab_plus60_max6_20261005' / 'manifest.json'
    if not path.is_file():
        pytest.skip('real older event manifest not available: %s' % path)
    manifest = json.loads(path.read_text())
    assert manifest['lab_scope'] == 'all-numeric-known-unit' and 'cutoff_rule' not in manifest
    assert build_delay.policy_text(manifest) == old_delay_policy(manifest)


def test_delay_extra_without_lab_scope_keeps_its_old_keys():
    assert set(build_delay.extra(delay_manifest(False))) == {
        'cutoff_policy', 'delay_minutes', 'inherits_index_sha256', 'cutoff_clipped_at_outtime',
        'early_triage_eligible'}


def run_delay_build(tmp_path, inherited, raw, name, **options):
    """build.run exactly as build_delay.main calls it (same hooks), minus the delay-only validator."""
    manifest = json.loads((inherited / 'manifest.json').read_text())
    args = make_args(tmp_path, inherited, raw, name, **options)
    build.run(args, decision_policy=build_delay.policy_text(manifest),
              manifest_extra=build_delay.extra(manifest))
    return read_manifest(args)


def test_delay_build_copies_cutoff_rule_lab_scope_and_decision_policy(v5_inputs):
    tmp_path, inherited, raw = v5_inputs
    manifest = run_delay_build(tmp_path, inherited, raw, 'l4-v5', level_source='row-range',
                               age_policy='visit')
    assert manifest['cutoff_rule'] == LAB_SCOPE_FIELDS['cutoff_rule']
    assert manifest['lab_scope'] == LAB_SCOPE_FIELDS['lab_scope']
    assert manifest['decision_policy'] == LAB_SCOPE_FIELDS['decision_policy']
    assert manifest['cutoff_policy'] == 'first_lab_plus_delay'
    assert manifest['delay_minutes'] == 60
    assert manifest['threshold_semantics'] == threshold_semantics()


def test_delay_build_without_lab_scope_keeps_old_policy_text_and_keys(plain_inputs):
    tmp_path, inherited, raw = plain_inputs
    manifest = run_delay_build(tmp_path, inherited, raw, 'l-old')
    assert manifest['decision_policy'] == old_delay_policy(json.loads((inherited / 'manifest.json').read_text()))
    assert not {'cutoff_rule', 'lab_scope', 'threshold_semantics'} & set(manifest)


def test_delay_build_on_an_older_manifest_with_lab_scope_keeps_old_policy_text_and_keys(older_inputs):
    tmp_path, inherited, raw = older_inputs
    manifest = run_delay_build(tmp_path, inherited, raw, 'l-older')
    assert manifest['decision_policy'] == old_delay_policy(json.loads((inherited / 'manifest.json').read_text()))
    assert not {'cutoff_rule', 'lab_scope', 'threshold_semantics'} & set(manifest)


# ------------------------------------------------------------------ SC5: age readback (synthetic store)
class AgeStore(FakeStore):
    """FakeStore whose patient node age (`arrival`) and re-derived age (`visit_age`) are set apart."""

    def __init__(self, records, node_age, derived_age):
        super().__init__(records)
        self.node_age, self.derived_age = node_age, derived_age

    def arrival(self, stay):
        return {'gender': 'F', 'age': self.node_age}

    def visit_age(self, stay):
        return self.derived_age


def write_age_artifact(root, node_age, derived_age):
    records = [dict(r, timing_basis='storetime') for r in potassium(3.2)]
    store = AgeStore(records, node_age, derived_age)
    graph, membership = build_graph_with_visit_membership(
        store, SAMPLE, [], set(), thresholds=THRESHOLDS, level_source='row-range')
    dump = dict(sort_keys=True, separators=(',', ':'))
    (root / 'graphs.jsonl').write_text(json.dumps(graph, **dump) + '\n')
    (root / VISIT_MEMBERSHIP_FILENAME).write_text(json.dumps(membership, **dump) + '\n')
    return store


def verify(root, store, age_policy):
    return verify_artifact(root, [SAMPLE], store, set(), None, THRESHOLDS, None, 'row-range',
                           age_policy=age_policy)


def test_visit_age_readback_passes_with_the_correct_age_and_records_the_count(tmp_path):
    store = write_age_artifact(tmp_path, 42.0, 42.0)
    result = verify(tmp_path, store, 'visit')
    assert result['status'] == 'verified'
    assert result['counts']['ages_checked'] == 1


def test_visit_age_readback_rejects_a_wrong_age(tmp_path):
    store = write_age_artifact(tmp_path, 43.0, 42.0)  # off by one year
    with pytest.raises(ValueError, match='age'):
        verify(tmp_path, store, 'visit')


def test_visit_age_readback_rejects_a_missing_patient_age(tmp_path):
    store = write_age_artifact(tmp_path, None, 42.0)
    with pytest.raises(ValueError, match='age'):
        verify(tmp_path, store, 'visit')


def test_visit_age_readback_rejects_a_graph_age_the_source_cannot_derive(tmp_path):
    store = write_age_artifact(tmp_path, 42.0, None)  # e.g. anchor_year dropped from the source
    with pytest.raises(ValueError, match='age'):
        verify(tmp_path, store, 'visit')


def test_anchor_policy_readback_is_unchanged_and_never_asks_the_store_for_a_visit_age(tmp_path):
    store = write_age_artifact(tmp_path, 40.0, 'never used')
    result = verify(tmp_path, store, 'anchor')
    assert result['status'] == 'verified'
    assert 'ages_checked' not in result['counts']


def test_default_age_policy_is_anchor(tmp_path):
    store = write_age_artifact(tmp_path, 40.0, 'never used')
    result = verify_artifact(tmp_path, [SAMPLE], store, set(), None, THRESHOLDS, None, 'row-range')
    assert 'ages_checked' not in result['counts']


# ------------------------------------------------------------------ SC5: age readback end to end (real store)
def test_visit_build_records_the_ages_checked_and_the_derived_ages(plain_inputs):
    tmp_path, inherited, raw = plain_inputs
    args = make_args(tmp_path, inherited, raw, 'age-visit', age_policy='visit')
    build.run(args)
    manifest = read_manifest(args)
    verification = json.loads((args.output / 'verification.json').read_text())
    assert manifest['status'] == 'completed'
    assert verification['counts']['ages_checked'] == 8
    assert manifest['counts']['ages_checked'] == 8
    ages = {}
    for line in (args.output / 'graphs.jsonl').read_text().splitlines():
        graph = json.loads(line)
        patient = next(n for n in graph['nodes'] if n['kind'] == 'patient')
        ages.setdefault(patient['age'], 0)
        ages[patient['age']] += 1
    assert ages == {EXPECTED_AGES['1']: 7, EXPECTED_AGES['2']: 1}


def test_anchor_build_verification_has_no_age_count(plain_inputs):
    tmp_path, inherited, raw = plain_inputs
    args = make_args(tmp_path, inherited, raw, 'age-anchor', age_policy='anchor')
    build.run(args)
    verification = json.loads((args.output / 'verification.json').read_text())
    assert 'ages_checked' not in verification['counts']


def test_visit_build_without_an_anchor_year_fails_the_readback(tmp_path):
    inherited = make_inherited(tmp_path / 'inh', lab_scope=False)
    raw = make_raw(tmp_path / 'raw', inherited, anchor_year_subjects=('1',))  # subject 2: no anchor_year
    args = make_args(tmp_path, inherited, raw, 'age-missing', age_policy='visit')
    with pytest.raises(ValueError, match='age'):
        build.run(args)
    assert read_manifest(args)['status'] == 'failed'
