"""R1 (SC3): build.verify_artifact re-checks threshold indicates edges with the same
level / ref_upper / level_source / kind the builder used.

Synthetic store and threshold rows only: no patient data, no event index, no test fold.
"""
import json

import pytest

from data.s4_graph.build import verify_artifact
from data.s4_graph.graph import build_graph_with_visit_membership
from core.contracts import VISIT_MEMBERSHIP_FILENAME
from test_v5_threshold_states import (BICARB, COMBO_THRESHOLDS, DKA, HHS, HR, HYPO, K, QSOFA, SAMPLE, SBP,
                                      SIRS, THRESHOLDS, FakeStore, dka, lab, level_of, potassium, rule)

TAMPER_TOKEN = 'knowledge:state:tampered'
# CSV comparison holds for K 3.2 ('> 3.0'), the row-range level rule ('> needs high') does not.
TAMPER_RULE = rule('state_k_tamper', 'state', K, 'mEq/L', '>', 3.0, TAMPER_TOKEN)


def storetime(records):
    """The readback requires the storetime availability proxy on every measurement."""
    return [dict(r, timing_basis='storetime') for r in records]


def write_artifact(root, records, level_source, thresholds=THRESHOLDS, edit=None):
    graph, membership = build_graph_with_visit_membership(
        FakeStore(storetime(records)), SAMPLE, [], set(), thresholds=thresholds, level_source=level_source)
    if edit is not None:
        edit(graph)
    dump = dict(sort_keys=True, separators=(',', ':'))
    (root / 'graphs.jsonl').write_text(json.dumps(graph, **dump) + '\n')
    (root / VISIT_MEMBERSHIP_FILENAME).write_text(json.dumps(membership, **dump) + '\n')
    return graph


def verify_with_store(root, records, level_source, thresholds=THRESHOLDS):
    return verify_artifact(root, [SAMPLE], FakeStore(storetime(records)), set(), None, thresholds,
                           None, level_source)


def test_row_range_anion_gap_with_own_upper_below_the_csv_threshold_passes_the_readback(tmp_path):
    records = dka(11.0, 3.0, 10.0)  # own range 3-10: 11 is above own upper, below CSV 12
    graph = write_artifact(tmp_path, records, 'row-range')
    assert DKA in graph['coverage']['threshold_states']
    result = verify_with_store(tmp_path, records, 'row-range')
    assert result['status'] == 'verified'


def test_row_range_state_and_level_pass_the_readback(tmp_path):
    records = potassium(3.2)
    graph = write_artifact(tmp_path, records, 'row-range')
    assert HYPO in graph['coverage']['threshold_states']
    assert verify_with_store(tmp_path, records, 'row-range')['status'] == 'verified'


def test_csv_graph_still_passes_the_csv_readback(tmp_path):
    records = potassium(3.4) + [r for r in dka(15.0, 8.0, 20.0)]
    write_artifact(tmp_path, records, 'csv')
    assert verify_with_store(tmp_path, records, 'csv')['status'] == 'verified'


def test_tampered_state_edge_on_a_contradicting_level_is_rejected(tmp_path):
    records = potassium(3.2)  # level low; the honest graph fires hypokalemia

    def retarget(graph):
        knowledge = next(n for n in graph['nodes'] if n['token'] == HYPO)
        knowledge['token'] = TAMPER_TOKEN
        edge = next(e for e in graph['edges'] if e['relation'] == 'medical:indicates')
        edge['rule_id'] = 'state_k_tamper'
        graph['coverage']['threshold_states'] = [TAMPER_TOKEN]

    thresholds = THRESHOLDS + [TAMPER_RULE]
    write_artifact(tmp_path, records, 'row-range', thresholds=thresholds, edit=retarget)
    with pytest.raises(ValueError, match='condition does not hold'):
        verify_with_store(tmp_path, records, 'row-range', thresholds=thresholds)


def test_csv_only_edge_passes_the_csv_readback_and_fails_the_row_range_readback(tmp_path):
    # Control: an honest csv graph fires the '> 3.0' rule on K 3.2; only the row-range semantics reject it.
    records = potassium(3.2)
    thresholds = THRESHOLDS + [TAMPER_RULE]
    write_artifact(tmp_path, records, 'csv', thresholds=thresholds)
    assert verify_with_store(tmp_path, records, 'csv', thresholds=thresholds)['status'] == 'verified'
    with pytest.raises(ValueError, match='condition does not hold'):
        verify_with_store(tmp_path, records, 'row-range', thresholds=thresholds)


# ------------- N1: COMBO rows fire on normal nodes by design; the readback must pass the rule's kind
def combo_records():
    # All three nodes are normal under the level rows, yet each combo cut-off holds on the CSV value.
    return [lab('hr', HR, 'bpm', 95.0), lab('sbp', SBP, 'mmHg', 95.0), lab('bic', BICARB, 'mEq/L', 24.0)]


def test_row_range_combo_firing_on_a_normal_node_passes_the_readback(tmp_path):
    records = combo_records()
    graph = write_artifact(tmp_path, records, 'row-range', thresholds=COMBO_THRESHOLDS)
    assert {n['token']: n['level'] for n in graph['nodes'] if n['kind'] == 'measurement'} == {
        HR: 'normal', SBP: 'normal', BICARB: 'normal'}
    assert {SIRS, QSOFA, HHS} <= set(graph['coverage']['threshold_states'])
    result = verify_with_store(tmp_path, records, 'row-range', thresholds=COMBO_THRESHOLDS)
    assert result['status'] == 'verified'


def test_row_range_sirs_combo_alone_on_a_normal_heart_rate_passes_the_readback(tmp_path):
    records = [lab('hr', HR, 'bpm', 95.0)]
    graph = write_artifact(tmp_path, records, 'row-range', thresholds=COMBO_THRESHOLDS)
    assert level_of(graph, HR) == 'normal' and SIRS in graph['coverage']['threshold_states']
    assert verify_with_store(tmp_path, records, 'row-range', thresholds=COMBO_THRESHOLDS)['status'] == 'verified'



@pytest.mark.parametrize('records,rules', [(potassium(3.2), THRESHOLDS), (combo_records(), COMBO_THRESHOLDS)])
def test_expected_pairs_equal_the_edges_the_builder_fired(records, rules):
    # fv10-d1-03: the readback compares the full set of fired (node, rule) pairs (missing edges too).
    from data.s4_graph.build import _expected_threshold_pairs
    graph, _ = build_graph_with_visit_membership(
        FakeStore(storetime(records)), SAMPLE, [], set(), thresholds=rules, level_source='row-range')
    gender = next((n.get('gender') for n in graph['nodes'] if n['kind'] == 'patient'), None)
    fired = {(e['source'], e['rule_id']) for e in graph['edges'] if e['relation'] == 'medical:indicates'}
    assert fired
    assert _expected_threshold_pairs(graph['nodes'], rules, gender, 'row-range') == fired
    assert _expected_threshold_pairs(graph['nodes'], rules, gender, 'row-range') != set()
