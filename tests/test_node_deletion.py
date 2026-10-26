"""Node deletion at the graph-record level (core/explain/node_deletion.py), the operator behind the
fidelity metrics of intervention contract version 6.

Synthetic graph record shaped like the graph builder's output (data/s4_graph/graph.py): a global
patient, the index visit (last ordinal), two complaints, a vital, two index labs of one analyte and
one of another, a prior lab of the second analyte, prior diagnoses and knowledge nodes (a fired
threshold state, a diagnosis concept, a complaint concept). It is encoded with a hand-fitted
preprocessing and scored by the four final methods (gsat, graphcare, protgnn, cei_gnn_mp2) with
perturbed weights. The second half builds synthetic patients with the graph builder itself
(`build_graph_with_visit_membership`) from an in-memory event store and compares every deletion with
the builder run on the store without the deleted events. No data file, run or checkpoint is read.
"""
from __future__ import annotations

import copy
import csv
import itertools
import json
from argparse import Namespace
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from cei.studies.cei_v3_absence import fit_universe
from cei.studies.cei_v3_ple import fit_knots
from comparison.s7_why.masking import apply_mask
from core.contracts import VISIT_MEMBERSHIP_CONTRACT_VERSION, validate_visit_membership_record
from core.explain.node_deletion import (ComboRules, DeletionScorer, combo_source, delete_from_record,
                                        encoding_differences, load_combo_rules)
from core.registry import METHOD_REGISTRY, build_method
from core.schema import THRESHOLD_COLUMNS, read_thresholds, sha256
from data.s4_graph.graph import build_graph_with_visit_membership
from core.tensorize import (ALL_RELATIONS, CONTEXT_NUMERIC, PAYLOAD_WIDTH, PREPROCESSING_VERSION, Scaler,
                            Vocabulary, context_number, encode_graph, node_feature_layout, node_token,
                            triple_count, triple_key)
from tests.graphxai_fixtures import method_args

METHODS = ('gsat', 'graphcare', 'protgnn', 'cei_gnn_mp2')
CLASSES = 3
# message-passing options of the final cei_gnn_mp2 runs (binding method_config.mp2_options)
MP2_FINAL = dict(mp_feed='dual', mp_hub_max=1, mp_dedup=1, mp_ctx='table', mp_root='bias',
                 mp_aggr='relation', mp_layers=2, mp_optimizer='adamw', mp_lr_schedule='cosine',
                 mp_label_smoothing=0.1)

# record positions by name; ordinals 0 and 1 are prior visits, 2 is the index visit
NAMES = ('p', 'v', 'c1', 'c2', 'hr', 'm1', 'm2', 'a1', 'm3', 'a2', 'k1', 'mp', 'd1', 'k2', 'k3', 'd2')
P = {name: i for i, name in enumerate(NAMES)}
MEMBERSHIP = {'v': {2}, 'c1': {2}, 'c2': {2}, 'hr': {2}, 'm1': {2}, 'm2': {2}, 'a1': {2}, 'm3': {2},
              'a2': {1, 2}, 'k1': {2}, 'mp': {1}, 'd1': {0}, 'k2': {0}, 'k3': {2}, 'd2': {0}}


def _edge(source, target, relation, informative=False, **payload):
    return {'source': source, 'target': target, 'relation': relation, 'informative': informative,
            **payload}


def record(sample_id='del-1'):
    """(graph record, membership sidecar row) with the ids of NAMES."""
    lab = dict(time_hours=-2.0, available_hours=-1.5)
    nodes = [
        {'id': 'p', 'kind': 'patient', 'token': 'patient', 'age': 61, 'gender': 'F', 'race': 'A',
         'arrival_transport': 'WALK IN'},
        {'id': 'v', 'kind': 'visit', 'token': 'visit:index', 'acuity': 2, 'time_hours': -3.0,
         'available_hours': -3.0},
        {'id': 'c1', 'kind': 'complaint', 'token': 'cc:chest pain', 'time_hours': -3.0},
        {'id': 'c2', 'kind': 'complaint', 'token': 'cc:dyspnea', 'time_hours': -3.0},
        {'id': 'hr', 'kind': 'vital', 'token': 'vital:heartrate', 'unit': 'bpm', 'value': 121.0,
         'level': 'high', 'time_hours': -3.0},
        {'id': 'm1', 'kind': 'measurement', 'token': 'lab:potassium', 'unit': 'mEq/L', 'value': 6.4,
         'level': 'high', 'scope': 'index', **lab},
        {'id': 'm2', 'kind': 'measurement', 'token': 'lab:potassium', 'unit': 'mEq/L', 'value': 4.1,
         'level': 'normal', 'scope': 'index', **lab},
        {'id': 'a1', 'kind': 'analyte', 'token': 'lab:potassium'},
        {'id': 'm3', 'kind': 'measurement', 'token': 'lab:creatinine', 'unit': 'mg/dL', 'value': 1.9,
         'level': 'high', 'scope': 'index', **lab},
        {'id': 'a2', 'kind': 'analyte', 'token': 'lab:creatinine'},
        {'id': 'k1', 'kind': 'knowledge', 'token': 'state:hyperkalemia'},
        {'id': 'mp', 'kind': 'measurement', 'token': 'lab:creatinine', 'unit': 'mg/dL', 'value': 1.1,
         'level': 'normal', 'scope': 'prior', 'time_hours': -900.0, 'available_hours': -899.0},
        {'id': 'd1', 'kind': 'diagnosis', 'token': 'dx:I10', 'prior_encounters': 2},
        {'id': 'k2', 'kind': 'knowledge', 'token': 'kg:hypertensive disease'},
        {'id': 'k3', 'kind': 'knowledge', 'token': 'kg:chest finding'},
        {'id': 'd2', 'kind': 'diagnosis', 'token': 'dx:E11', 'prior_encounters': 1},
    ]
    edges = [
        _edge('p', 'v', 'has_visit'), _edge('v', 'p', 'index_visit_of'),
        _edge('v', 'c1', 'reports_complaint'), _edge('v', 'c2', 'reports_complaint'),
        _edge('c1', 'c2', 'co_complaint', True), _edge('c2', 'c1', 'co_complaint', True),
        _edge('v', 'hr', 'observed_vital'),
        _edge('v', 'm1', 'measured_in'), _edge('m1', 'a1', 'instance_of'),
        _edge('v', 'm2', 'measured_in'), _edge('m2', 'a1', 'instance_of'),
        _edge('v', 'm3', 'measured_in'), _edge('m3', 'a2', 'instance_of'),
        _edge('mp', 'a2', 'instance_of'),
        _edge('mp', 'm3', 'baseline_of', True, delta=0.8, interval_hours=898.0, rate_per_hour=0.0009,
              comparable_units=True),
        _edge('v', 'd1', 'has_prior_diagnosis'),
        _edge('d1', 'v', 'recurrence_of', True, prior_encounters=2, last_seen_hours=-700.0),
        _edge('m1', 'k1', 'medical:indicates', True),
        _edge('d1', 'k2', 'medical:is_a', True),
        _edge('c1', 'k3', 'medical:assesses', True),
        # the second prior diagnosis hangs only off the first (no visit edge of its own), so it is left
        # without an edge when the first is deleted: the source-masking orphan rule
        _edge('d1', 'd2', 'comorbid_with', True), _edge('d2', 'd1', 'comorbid_with', True),
    ]
    graph = {'sample_id': sample_id, 'nodes': nodes, 'edges': edges,
             'coverage': {'complaints': 2, 'index_measurements': 3, 'prior_visits': 2}}
    pairs = sorted([o, P[name]] for name, ordinals in MEMBERSHIP.items() for o in ordinals)
    membership = {'contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION, 'sample_id': sample_id,
                  'visit_ordinals': [0, 1, 2], 'membership_pairs': pairs,
                  'global_node_mask': [node['kind'] == 'patient' for node in nodes]}
    validate_visit_membership_record(graph, membership)
    return graph, membership


def by_hand(graph, membership, dropped, ordinal_map, visits=None):
    """The record without the nodes named in `dropped`, written out independently of the operator:
    their edges and membership pairs go, `visits` gives the old ordinals of a surviving concept node
    whose visits changed, `ordinal_map` renumbers the surviving ordinals."""
    positions = {node['id']: i for i, node in enumerate(graph['nodes'])}
    gone = {positions[name] for name in dropped}
    ids = {graph['nodes'][i]['id'] for i in gone}
    keep = [i for i in range(len(graph['nodes'])) if i not in gone]
    new = {old: new for new, old in enumerate(keep)}
    nodes = [graph['nodes'][i] for i in keep]
    edges = [e for e in graph['edges'] if e['source'] not in ids and e['target'] not in ids]
    changed = {positions[name]: ordinals for name, ordinals in (visits or {}).items()}
    old_pairs = [(o, i) for o, i in membership['membership_pairs'] if i not in changed]
    old_pairs += [(o, i) for i, ordinals in changed.items() for o in ordinals]
    pairs = sorted([ordinal_map[o], new[i]] for o, i in old_pairs if i in new)
    return ({**graph, 'nodes': nodes, 'edges': edges},
            {**membership, 'visit_ordinals': sorted(set(ordinal_map.values())), 'membership_pairs': pairs,
             'global_node_mask': [membership['global_node_mask'][i] for i in keep]})


SAME = {0: 0, 1: 1, 2: 2}
# the hand-written record has no combo rule; without a table the fallback combo rule is used
NO_COMBO = {'source': 'distinct_anchor_tokens', 'groups_deleted': 0, 'reevaluated': 0, 'refired': 0,
            'new_state_nodes': 0, 'not_reevaluated': []}
EVIDENCE = ('c1', 'c2', 'hr', 'm1', 'm2', 'm3', 'mp', 'd1', 'd2')
# selected -> (every node that leaves, cascaded counts, ordinal map of the surviving visits,
#              new visits (old ordinals) of a surviving concept node whose anchors changed)
CASES = {
    'one_of_two_labs': (('m2',), ('m2',), (0, 0, 0), SAME, {}),
    'lab_with_fired_state': (('m1',), ('m1', 'k1'), (0, 1, 0), SAME, {}),
    'both_labs_of_an_analyte': (('m1', 'm2'), ('m1', 'm2', 'a1', 'k1'), (1, 1, 0), SAME, {}),
    # the analyte now has only its prior lab: it no longer belongs to the index visit
    'index_lab_with_prior_lab_left': (('m3',), ('m3',), (0, 0, 0), SAME, {'a2': {1}}),
    'index_and_prior_lab': (('m3', 'mp'), ('m3', 'mp', 'a2'), (1, 0, 0), {0: 0, 2: 1}, {}),
    'prior_diagnosis_empties_its_visit': (('d1',), ('d1', 'k2', 'd2'), (0, 1, 1), {1: 0, 2: 1}, {}),
    'complaint_with_concept': (('c1',), ('c1', 'k3'), (0, 1, 0), SAME, {}),
    # the analyte loses the prior visit with its only prior lab, and that visit holds nothing else
    'prior_lab_only': (('mp',), ('mp',), (0, 0, 0), {0: 0, 2: 1}, {'a2': {2}}),
    'every_eligible_node': (EVIDENCE, EVIDENCE + ('a1', 'a2', 'k1', 'k2', 'k3'), (2, 3, 0), {2: 0}, {}),
}


# ------------------------------------------------------------------ record rules (no model)
@pytest.mark.parametrize('case', sorted(CASES))
def test_deletion_follows_the_cascade_orphan_and_visit_rules(case):
    selected, leaving, (analyte, knowledge, orphan), ordinal_map, visits = CASES[case]
    graph, membership = record()
    new_graph, new_membership, info = delete_from_record(graph, membership, [P[n] for n in selected])
    expected_graph, expected_membership = by_hand(graph, membership, leaving, ordinal_map, visits)
    assert [n['token'] for n in new_graph['nodes']] == [n['token'] for n in expected_graph['nodes']]
    assert [n['id'] for n in new_graph['nodes']] == [f'n{i}' for i in range(len(new_graph['nodes']))]
    old_id = {f'n{i}': node['id'] for i, node in enumerate(expected_graph['nodes'])}
    assert [(old_id[e['source']], old_id[e['target']], e['relation']) for e in new_graph['edges']] == [
        (e['source'], e['target'], e['relation']) for e in expected_graph['edges']]
    assert new_membership == expected_membership
    assert info == {'selected': len(selected),
                    'cascaded': {'analyte': analyte, 'knowledge': knowledge, 'orphan': orphan},
                    'combos': NO_COMBO, 'visits_dropped': 3 - len(set(ordinal_map.values())),
                    'visits_originally_empty': 0,
                    'nodes_after': len(expected_graph['nodes']), 'edges_after': len(expected_graph['edges'])}
    validate_visit_membership_record(new_graph, new_membership)
    assert new_graph['coverage'] == graph['coverage'] and new_graph['sample_id'] == graph['sample_id']


def test_the_inputs_are_never_modified_and_the_empty_set_is_the_record_itself():
    graph, membership = record()
    before = copy.deepcopy((graph, membership))
    for selected in ([P['m1'], P['m2']], [P['d1']], [P[n] for n in EVIDENCE]):
        delete_from_record(graph, membership, selected)
    assert (graph, membership) == before
    same_graph, same_membership, info = delete_from_record(graph, membership, [])
    assert same_graph is graph and same_membership is membership
    assert info == {'selected': 0, 'cascaded': {'analyte': 0, 'knowledge': 0, 'orphan': 0},
                    'combos': NO_COMBO, 'visits_dropped': 0, 'visits_originally_empty': 0,
                    'nodes_after': 16, 'edges_after': len(graph['edges'])}


@pytest.mark.parametrize('name', ['p', 'v', 'a1', 'k1'])
def test_only_eligible_evidence_can_be_selected(name):
    graph, membership = record()
    with pytest.raises(ValueError, match='not eligible evidence'):
        delete_from_record(graph, membership, [P[name]])


def test_positions_outside_the_graph_and_ambiguous_records_are_refused():
    graph, membership = record()
    for bad in (16, -1, 2.5, True):
        with pytest.raises(ValueError, match='not a node position'):
            delete_from_record(graph, membership, [bad])
    duplicate = copy.deepcopy(graph)
    duplicate['nodes'][3]['id'] = 'c1'
    with pytest.raises(ValueError, match='duplicate node ids'):
        delete_from_record(duplicate, membership, [P['m1']])
    moved = copy.deepcopy(membership)
    moved['membership_pairs'] = sorted(moved['membership_pairs'] + [[1, P['v']]])
    with pytest.raises(ValueError, match='index visit node'):
        delete_from_record(graph, moved, [P['m1']])


def test_a_deleted_middle_measurement_leaves_its_trajectory_chain_broken():
    graph, membership = record()
    graph['nodes'].append({'id': 'mq', 'kind': 'measurement', 'token': 'lab:creatinine', 'unit': 'mg/dL',
                           'value': 1.4, 'scope': 'prior', 'time_hours': -400.0})
    graph['edges'] += [_edge('mq', 'a2', 'instance_of'),
                       _edge('mp', 'mq', 'trajectory_of', True, delta=0.3, interval_hours=500.0)]
    graph['edges'] = [dict(e, source='mq') if e['relation'] == 'baseline_of' else e for e in graph['edges']]
    membership['global_node_mask'].append(False)
    membership['membership_pairs'] = sorted(membership['membership_pairs'] + [[1, 16]])
    validate_visit_membership_record(graph, membership)
    new_graph, _, _ = delete_from_record(graph, membership, [16])
    relations = [e['relation'] for e in new_graph['edges']]
    assert 'trajectory_of' not in relations and 'baseline_of' not in relations   # no relinking mp -> m3


def with_second_prior_lab(ordinal):
    """The record plus a second prior creatinine lab `mq` in `ordinal` and a knowledge node `k4`
    anchored on the creatinine analyte, with the visits the graph builder gives them."""
    graph, membership = record()
    graph['nodes'] += [{'id': 'mq', 'kind': 'measurement', 'token': 'lab:creatinine', 'unit': 'mg/dL',
                        'value': 1.3, 'scope': 'prior', 'time_hours': -950.0, 'available_hours': -949.0},
                       {'id': 'k4', 'kind': 'knowledge', 'token': 'kg:renal function'}]
    graph['edges'] += [_edge('mq', 'a2', 'instance_of'), _edge('a2', 'k4', 'medical:measures', True)]
    analyte = {1, 2, ordinal}
    pairs = [p for p in membership['membership_pairs'] if p[1] != P['a2']]
    pairs += [[o, P['a2']] for o in analyte] + [[ordinal, 16]] + [[o, 17] for o in analyte]
    membership['membership_pairs'] = sorted(pairs)
    membership['global_node_mask'] += [False, False]
    validate_visit_membership_record(graph, membership)
    return graph, membership


# second prior lab's visit -> (deleted, ordinal map, new visits (old ordinals) of the concept nodes)
SECOND_PRIOR_LAB = {
    'same_prior_visit_kept_through_the_second_lab': (1, ('mp',), SAME, {'a2': {1, 2}, 'k4': {1, 2}}),
    'other_prior_visit_kept_the_emptied_one_dropped': (0, ('mp',), {0: 0, 2: 1},
                                                       {'a2': {0, 2}, 'k4': {0, 2}}),
    'both_prior_labs_gone': (0, ('mp', 'mq'), {0: 0, 2: 1}, {'a2': {2}, 'k4': {2}}),
}


@pytest.mark.parametrize('case', sorted(SECOND_PRIOR_LAB))
def test_concept_nodes_take_the_visits_of_their_surviving_anchors(case):
    ordinal, deleted, ordinal_map, visits = SECOND_PRIOR_LAB[case]
    graph, membership = with_second_prior_lab(ordinal)
    positions = {node['id']: i for i, node in enumerate(graph['nodes'])}
    new_graph, new_membership, info = delete_from_record(graph, membership,
                                                         [positions[n] for n in deleted])
    expected_graph, expected_membership = by_hand(graph, membership, deleted, ordinal_map, visits)
    assert [n['token'] for n in new_graph['nodes']] == [n['token'] for n in expected_graph['nodes']]
    assert new_membership == expected_membership
    assert info['cascaded'] == {'analyte': 0, 'knowledge': 0, 'orphan': 0}
    assert info['visits_dropped'] == 3 - len(set(ordinal_map.values()))
    validate_visit_membership_record(new_graph, new_membership)


def test_an_untouched_concept_membership_is_the_builders_rule():
    """Recomputing every concept node's visits on a graph that lost nothing relevant reproduces the
    recorded visits (the fixtures follow the graph builder)."""
    for graph, membership in (record(), with_second_prior_lab(0), with_second_prior_lab(1)):
        positions = {node['id']: i for i, node in enumerate(graph['nodes'])}
        _, new_membership, _ = delete_from_record(graph, membership, [positions['c2']])
        expected = by_hand(graph, membership, ('c2',), SAME)[1]
        assert new_membership == expected


def test_cascade_off_keeps_the_concept_nodes():
    graph, membership = record()
    new_graph, _, info = delete_from_record(graph, membership, [P['m1'], P['m2']], cascade=False)
    assert [n['token'] for n in new_graph['nodes']].count('lab:potassium') == 1        # the analyte
    assert 'state:hyperkalemia' in [n['token'] for n in new_graph['nodes']]
    assert info['cascaded'] == {'analyte': 0, 'knowledge': 0, 'orphan': 0}


def test_the_orphan_and_visit_renumbering_rules_are_the_source_masking_rules():
    """On a record without a combo state and without a prior visit that is empty already, deleting
    every lab is the labs mask of source masking (the two decided differences on builder graphs:
    test_deletion_and_the_labs_mask_differ_in_the_two_decided_ways)."""
    graph, membership = record()
    labs = [i for i, n in enumerate(graph['nodes']) if n['kind'] == 'measurement']
    new_graph, new_membership, _ = delete_from_record(graph, membership, labs)
    masked_graph, masked_membership, _ = apply_mask(graph, membership, 'labs')
    assert new_graph['nodes'] == masked_graph['nodes'] and new_graph['edges'] == masked_graph['edges']
    assert new_membership == masked_membership


# ------------------------------------------------------------------ encoding and models
def preprocessing(graphs):
    """A fitted-preprocessing dict of the shape `fit_preprocessing` returns, fitted on `graphs`."""
    tokens, values, triples = set(), {}, Counter()
    for graph in graphs:
        kind = {node['id']: node['kind'] for node in graph['nodes']}
        for node in graph['nodes']:
            tokens.add(node_token(node))
            if node.get('value') is not None:
                values.setdefault(node_token(node), []).append(node['value'])
            for field in CONTEXT_NUMERIC:
                if context_number(node, field) is not None:
                    values.setdefault('context.' + field, []).append(context_number(node, field))
        for e in graph['edges']:
            triples[triple_key(kind[e['source']], e['relation'], kind[e['target']])] += 1
    return {'preprocessing_version': PREPROCESSING_VERSION, 'thresholds_layer': True,
            'context_categories': {'gender': ['F', 'M'], 'race': ['A'], 'arrival_transport': ['WALK IN']},
            'vocabulary': Vocabulary(sorted(tokens), 20), 'scaler': Scaler.fit(values, min_count=1),
            'token_min_count': 20, 'triples': sorted(triples), 'triple_counts': dict(triples)}


def encoder(prep):
    return lambda graph, membership: encode_graph(graph, prep, membership)


def _perturb(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.3 * torch.randn(parameter.shape, generator=generator))
        for module in model.modules():   # the padding row stays zero, as in every trained checkpoint
            if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()
    return model.eval()


def _v3_state(prep, path, graph):
    vocabulary = prep['vocabulary']
    valued = sorted({node_token(n) for n in graph['nodes'] if n['kind'] in ('measurement', 'vital')})
    knots = fit_knots({item: np.linspace(-2.0, 2.0, 40, dtype=np.float32) for item in valued}, K=4,
                      min_values=20, token_min_count=20)
    universe = fit_universe({item: 30 for item in valued}, dict(vocabulary.index), min_graphs=20,
                            token_min_count=20)
    path.write_text(json.dumps({
        'version': 'cei_v3_state_v1', 'K': 4, 'knot_table': knots.state(),
        'knot_table_sha256': knots.sha256(), 'universe': universe.state(),
        'universe_sha256': universe.sha256(),
        'vocabulary': {'tokens': list(vocabulary.tokens), 'min_count': 20},
        'preprocessing_sha256': '0' * 64, 'node_feature_layout': node_feature_layout(prep)}))
    return path


def build_model(name, prep, tmp_path, seed=5, graph=None):
    dims = dict(num_tokens=len(prep['vocabulary']), node_dim=len(node_feature_layout(prep)),
                edge_dim=len(ALL_RELATIONS) + PAYLOAD_WIDTH, num_classes=CLASSES, hidden=8,
                token_dim=4, num_triples=triple_count(prep))
    torch.manual_seed(seed)
    if name == 'cei_gnn_mp2':
        state = _v3_state(prep, tmp_path / 'K4.json', graph if graph is not None else record()[0])
        args = Namespace(method_options={'arm': 'C', 'v3_state': str(state), 'k': 4, **MP2_FINAL},
                         seed=1234)
        model = METHOD_REGISTRY['cei_gnn_mp2'](layers=1, dropout=0.0, args=args, **dims)
    else:
        model = build_method(name, layers=2, dropout=0.0, args=method_args(), **dims)
    return _perturb(model, seed + 1000)


def model_forward(model):
    return lambda data: model(Batch.from_data_list([data]), epoch=0).logits


def reference_probs(model, prep, graph, membership):
    with torch.no_grad():
        return torch.softmax(model_forward(model)(encode_graph(graph, prep, membership)), dim=-1)


@pytest.fixture(scope='module')
def setups(tmp_path_factory):
    graph, membership = record()
    prep = preprocessing([graph])
    return {name: build_model(name, prep, tmp_path_factory.mktemp(name)) for name in METHODS}, prep


def scorer_for(model, prep, graph=None, membership=None):
    if graph is None:
        graph, membership = record()
    return DeletionScorer(model_forward(model), graph, membership, encoder(prep), y=0)


@pytest.mark.parametrize('name', METHODS)
@pytest.mark.parametrize('case', sorted(CASES))
def test_the_scorer_equals_encoding_the_reduced_record_and_running_the_adapter(setups, name, case):
    models, prep = setups
    selected, leaving, _, ordinal_map, visits = CASES[case]
    graph, membership = record()
    scorer = scorer_for(models[name], prep, graph, membership)
    got = scorer.probs([P[n] for n in selected])
    expected = reference_probs(models[name], prep,
                               *by_hand(graph, membership, leaving, ordinal_map, visits))
    assert torch.equal(got, expected), (name, case, got, expected)
    assert scorer.last_info['selected'] == len(selected)


@pytest.mark.parametrize('name', METHODS)
@pytest.mark.parametrize('case', sorted(SECOND_PRIOR_LAB))
def test_the_scorer_equals_the_reduced_record_with_recomputed_concept_visits(setups, name, case):
    models, prep = setups
    ordinal, deleted, ordinal_map, visits = SECOND_PRIOR_LAB[case]
    graph, membership = with_second_prior_lab(ordinal)
    positions = {node['id']: i for i, node in enumerate(graph['nodes'])}
    scorer = scorer_for(models[name], prep, graph, membership)
    expected = reference_probs(models[name], prep, *by_hand(graph, membership, deleted, ordinal_map, visits))
    assert torch.equal(scorer.probs([positions[n] for n in deleted]), expected)


@pytest.mark.parametrize('name', METHODS)
def test_the_empty_set_is_the_original_graph_bit_for_bit(setups, name):
    models, prep = setups
    graph, membership = record()
    scorer = scorer_for(models[name], prep, graph, membership)
    original = reference_probs(models[name], prep, graph, membership)
    assert torch.equal(scorer.probs(()), original)
    assert torch.equal(scorer.probs([]), original)
    assert scorer.last_info['selected'] == 0
    split_tensor = encode_graph(graph, prep, membership)
    scorer.check_against(split_tensor)
    assert encoding_differences(scorer.data, split_tensor) == []


@pytest.mark.parametrize('name', METHODS)
def test_the_concept_cascade_and_the_dropped_visit_reach_the_model(setups, name):
    """Deleting both potassium labs with and without the cascade, and deleting the prior diagnosis
    with and without dropping its emptied visit, give different predictions: the derived concept
    nodes are inputs of every method, the visit numbering of GraphCare (recency). CEI reads visits
    through its within-visit pairs and the index visit only, so an empty visit leaves it unchanged."""
    models, prep = setups
    graph, membership = record()
    scorer = scorer_for(models[name], prep, graph, membership)
    kept = by_hand(graph, membership, ('m1', 'm2'), SAME)
    assert not torch.equal(scorer.probs([P['m1'], P['m2']]),
                           reference_probs(models[name], prep, *kept))
    if name == 'graphcare':
        # the emptied visit is the middle one, so dropping it moves the oldest visit one step closer
        no_drop = by_hand(graph, membership, ('m3', 'mp', 'a2'), SAME)
        assert not torch.equal(scorer.probs([P['m3'], P['mp']]),
                               reference_probs(models[name], prep, *no_drop))


@pytest.mark.parametrize('name', METHODS)
def test_each_distinct_set_is_scored_once(setups, name):
    models, prep = setups
    calls = []
    forward = model_forward(models[name])
    graph, membership = record()
    scorer = DeletionScorer(lambda data: calls.append(1) or forward(data), graph, membership,
                            encoder(prep), y=0)
    assert len(calls) == 1                                   # the untouched record
    first = scorer.probs([P['m1'], P['c1']])
    info = scorer.last_info
    scorer.probs([P['m2']])
    again = scorer.probs([P['c1'], P['m1'], P['m1']])
    assert len(calls) == 3 and torch.equal(first, again) and scorer.last_info == info


def test_the_scorer_refuses_a_split_tensor_that_is_not_the_records_encoding(setups):
    models, prep = setups
    graph, membership = record()
    scorer = scorer_for(models['gsat'], prep, graph, membership)
    tampered = encode_graph(graph, prep, membership)
    tampered.x = tampered.x.clone()
    tampered.x[3, 0] += 1.0
    with pytest.raises(ValueError, match=r"differs from the split tensor in \['x'\]"):
        scorer.check_against(tampered)
    shorter = encoder(prep)(*by_hand(graph, membership, ('m2',), SAME))
    with pytest.raises(ValueError, match='differs from the split tensor'):
        scorer.check_against(shorter)


def test_the_scorer_refuses_an_encoder_that_reorders_nodes(setups):
    models, prep = setups
    graph, membership = record()

    def reordering(g, m):
        data = encode_graph(g, prep, m)
        data.node_type = data.node_type.flip(0)
        return data

    with pytest.raises(ValueError, match='node indices differ from the record'):
        DeletionScorer(model_forward(models['gsat']), graph, membership, reordering, y=0)


# ------------------------------------------------------------------ against the graph builder
# Synthetic patients built by `build_graph_with_visit_membership` from an in-memory event store, with a
# threshold table of a level row, a state row and four combo rules (SIRS by two combos, qSOFA, DKA).
# For a deleted node set S the operator is compared with the builder run on the store without S's
# events ("never recorded"), with the decided visit rule applied to the rebuild (a prior visit the
# deletion emptied is dropped, one empty in the record stays).
PROVENANCE = {'source_uri': 'https://example.org/synthetic', 'source_version': 'v1',
              'reviewed_by': 'synthetic-check'}
SIRS, QSOFA, DKA = 'knowledge:state:sirs', 'knowledge:state:qsofa', 'knowledge:state:dka'


def _rule(rule_id, kind, token, unit, comparator, threshold, target):
    return {'rule_id': rule_id, 'kind': kind, 'source_token': token, 'unit': unit,
            'comparator': comparator, 'threshold': threshold, 'sex': 'any', 'target_token': target,
            'relation': 'level' if kind == 'level' else 'indicates', **PROVENANCE,
            'evidence': 'synthetic'}


RULES = [
    _rule('lvl_k_high', 'level', 'lab:K', 'mEq/L', '>', '5.1', 'level:high'),
    _rule('state_hyperk', 'state', 'lab:K', 'mEq/L', '>', '5.5', 'knowledge:state:hyperkalemia'),
    _rule('combo_sirs_hr_rr', 'combo', 'vital:heartrate', 'bpm', '>', '90', SIRS),
    _rule('combo_sirs_hr_rr', 'combo', 'vital:resprate', 'bpm', '>', '20', SIRS),
    _rule('combo_sirs_temp_hr', 'combo', 'vital:temperature', 'C', '>', '38', SIRS),
    _rule('combo_sirs_temp_hr', 'combo', 'vital:heartrate', 'bpm', '>', '90', SIRS),
    _rule('combo_qsofa', 'combo', 'vital:sbp', 'mmHg', '<=', '100', QSOFA),
    _rule('combo_qsofa', 'combo', 'vital:resprate', 'bpm', '>=', '22', QSOFA),
    _rule('combo_dka', 'combo', 'lab:GLU', 'mg/dL', '>', '250', DKA),
    _rule('combo_dka', 'combo', 'lab:BIC', 'mEq/L', '<=', '18', DKA),
]
KNOWLEDGE_TRIPLES = [
    {'source_token': source, 'target_token': target, 'relation': relation, **PROVENANCE}
    for source, target, relation in (('cc:chest pain', 'knowledge:system:chest', 'assesses'),
                                     ('dx:I10', 'knowledge:block:hypertension', 'is_a'),
                                     ('dx:E11', 'knowledge:block:diabetes', 'is_a'),
                                     ('lab:CR', 'knowledge:system:renal', 'measures'),
                                     ('vital:heartrate', 'knowledge:system:cardiac', 'measures'))]
COMPLAINT_VOCABULARY = {'chest pain', 'dyspnea', 'fever'}
INDEX_STAY = ('2020-07-01T00:00:00', '2020-07-01T09:00:00')


def _lab(lab_id, token, value, unit, time, stay=None):
    return {'id': lab_id, 'token': token, 'value': value, 'unit': unit, 'time': time, 'available': time,
            'source': 'labevents', 'timing_basis': 'storetime', 'censored': None, 'ref_lower': None,
            'ref_upper': None, 'flag': None, **({'stay': stay} if stay else {})}


def _dx(token, occurrences):
    return {'token': token, 'occurrences': occurrences, 'source_version': 10, 'source_code': token[3:],
            'approximate_mapping': False}


PATIENTS = {
    # a prior visit without any node in the record (B), two potassium labs at the index visit, a
    # creatinine with a prior lab, SIRS through two combos, qSOFA, and DKA fired on the later of two
    # glucose labs
    'P1': dict(visits={'A': ('2020-01-01T00:00:00', '2020-01-01T05:00:00'),
                       'B': ('2020-03-01T00:00:00', '2020-03-01T05:00:00'),
                       'C': ('2020-05-01T00:00:00', '2020-05-01T05:00:00'), 'I': INDEX_STAY},
               prior=['A', 'B', 'C'], complaints=['chest pain', 'dyspnea'],
               vitals=[('heartrate', 121.0, 'bpm'), ('resprate', 24.0, 'bpm'), ('temperature', 38.6, 'C'),
                       ('sbp', 95.0, 'mmHg')],
               labs=[_lab('k1', 'lab:K', 6.4, 'mEq/L', '2020-07-01T00:30:00'),
                     _lab('k2', 'lab:K', 4.1, 'mEq/L', '2020-07-01T01:00:00'),
                     _lab('cr', 'lab:CR', 1.9, 'mg/dL', '2020-07-01T00:30:00'),
                     _lab('glu1', 'lab:GLU', 300.0, 'mg/dL', '2020-07-01T00:30:00'),
                     _lab('glu2', 'lab:GLU', 280.0, 'mg/dL', '2020-07-01T01:00:00'),
                     _lab('bic', 'lab:BIC', 15.0, 'mEq/L', '2020-07-01T00:30:00')],
               history=[_lab('crp', 'lab:CR', 1.1, 'mg/dL', '2020-05-01T01:00:00', 'C')],
               diagnoses=[_dx('dx:I10', ['A'])]),
    # SIRS through one combo only (no respiratory rate), a potassium trajectory of two prior labs with
    # its baseline edge, comorbid prior diagnoses, and DKA not fired because the LATEST glucose is low
    'P2': dict(visits={'A': ('2020-01-01T00:00:00', '2020-01-01T05:00:00'),
                       'B': ('2020-03-01T00:00:00', '2020-03-01T05:00:00'), 'I': INDEX_STAY},
               prior=['A', 'B'], complaints=['fever'],
               vitals=[('heartrate', 95.0, 'bpm'), ('temperature', 38.9, 'C')],
               labs=[_lab('k1', 'lab:K', 5.9, 'mEq/L', '2020-07-01T00:30:00'),
                     _lab('k2', 'lab:K', 5.2, 'mEq/L', '2020-07-01T01:30:00'),
                     _lab('gluA', 'lab:GLU', 320.0, 'mg/dL', '2020-07-01T00:30:00'),
                     _lab('gluB', 'lab:GLU', 200.0, 'mg/dL', '2020-07-01T01:00:00'),
                     _lab('bic', 'lab:BIC', 14.0, 'mEq/L', '2020-07-01T00:30:00')],
               history=[_lab('kp1', 'lab:K', 4.0, 'mEq/L', '2020-01-01T01:00:00', 'A'),
                        _lab('kp2', 'lab:K', 4.8, 'mEq/L', '2020-03-01T01:00:00', 'B')],
               diagnoses=[_dx('dx:I10', ['A', 'B']), _dx('dx:E11', ['B'])]),
    # a single visit with qSOFA
    'P3': dict(visits={'I': INDEX_STAY}, prior=[], complaints=['chest pain'],
               vitals=[('heartrate', 80.0, 'bpm'), ('sbp', 99.0, 'mmHg'), ('resprate', 23.0, 'bpm')],
               labs=[_lab('cr', 'lab:CR', 1.0, 'mg/dL', '2020-07-01T00:30:00')], history=[], diagnoses=[]),
}


class FakeStore:
    """The event-store calls the graph builder makes, served from a patient spec without the events
    in `skip` (('cc', text), ('vital', field), ('lab', id), ('dx', token))."""

    def __init__(self, spec, skip=(), v5=True):
        self.spec, self.skip, self.has_v5_columns = spec, set(skip), v5

    def visit(self, stay):
        start, finish = self.spec['visits'][stay]
        return {'subject': 's1', 'start': start, 'finish': finish}

    def prior_visits(self, subject, start):
        return list(self.spec['prior'])

    def arrival(self, stay):
        return {'age': 70, 'gender': 'F', 'race': 'A', 'arrival_transport': 'WALK IN', 'acuity': 2}

    def complaints(self, stay):
        return [c for c in self.spec['complaints'] if ('cc', c) not in self.skip]

    def triage_vitals(self, stay):
        return [(f, v, u, u) for f, v, u in self.spec['vitals'] if ('vital', f) not in self.skip]

    def measurements(self, stay, cutoff):
        return [dict(r, stay='I') for r in self.spec['labs'] if ('lab', r['id']) not in self.skip]

    def analyte_history(self, subject, prior, token, cutoff):
        return [r for r in self.spec['history'] if r['token'] == token and ('lab', r['id']) not in self.skip]


class FakeDiagnoses:
    def __init__(self, rows):
        self.rows = rows

    def prior_history(self, stay, prior_visits):
        return list(self.rows)


@pytest.fixture(scope='module')
def table(tmp_path_factory):
    """The threshold table as a CSV file: the builder reads it with `read_thresholds`, the operator
    with `load_combo_rules`."""
    path = tmp_path_factory.mktemp('thresholds') / 'thresholds.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=THRESHOLD_COLUMNS)
        writer.writeheader()
        writer.writerows(RULES)
    return {'path': path, 'rows': read_thresholds(path),
            'rules': load_combo_rules(path, sha256(path), level_source='csv')}


def build_patient(name, thresholds, skip=(), v5=True):
    spec = PATIENTS[name]
    sample = {'sample_id': name, 'subject_id': 's1', 'stay_id': 'I', 'cutoff': datetime(2020, 7, 1, 2),
              'split': 'test'}
    diagnoses = FakeDiagnoses([d for d in spec['diagnoses'] if ('dx', d['token']) not in set(skip)])
    return build_graph_with_visit_membership(FakeStore(spec, skip, v5), sample, KNOWLEDGE_TRIPLES,
                                             COMPLAINT_VOCABULARY, diagnoses, thresholds=thresholds)


def source_events(name, graph, selected):
    """The store events the nodes at `selected` were built from."""
    labs = [*PATIENTS[name]['labs'], *PATIENTS[name]['history']]
    events = set()
    for i in selected:
        node = graph['nodes'][i]
        if node['kind'] == 'complaint':
            events.add(('cc', node['token'][3:]))
        elif node['kind'] == 'vital':
            events.add(('vital', node['token'].split(':', 1)[1]))
        elif node['kind'] == 'diagnosis':
            events.add(('dx', node['token']))
        else:
            (lab,) = [r for r in labs if r['token'] == node['token'] and r['value'] == node['value']]
            events.add(('lab', lab['id']))
    return events


def decided_visits(original, rebuilt):
    """The rebuild with the decided visit rule: a prior visit the deletion emptied is dropped (the
    builder numbers every prior stay), one empty in the record already stays."""
    graph, membership = rebuilt

    def used(m):
        return {o for o, pos in m['membership_pairs'] if not m['global_node_mask'][pos]}

    last = len(membership['visit_ordinals']) - 1
    emptied = used(original) - used(membership) - {last}
    order = [o for o in membership['visit_ordinals'] if o not in emptied]
    remap = {o: i for i, o in enumerate(order)}
    return graph, {**membership, 'visit_ordinals': list(range(len(order))),
                   'membership_pairs': [[remap[o], pos] for o, pos in membership['membership_pairs']]}


def encoded_canon(prep, graph, membership):
    """The encoded tensors up to node and edge storage order: every node row (features, token, type,
    global flag, visits) and every edge row (endpoint rows, attributes, relation, triple)."""
    data = encode_graph(graph, prep, membership)
    visits = defaultdict(list)
    for ordinal, pos in data.visit_membership_index.t().tolist():
        visits[pos].append(ordinal)
    rows = [(tuple(data.x[i].tolist()), int(data.token[i]), int(data.node_type[i]),
             bool(data.global_node_mask[i]), tuple(sorted(visits[i]))) for i in range(data.num_nodes)]
    edges = Counter((rows[s], rows[t], tuple(data.edge_attr[j].tolist()), int(data.edge_relation[j]),
                     int(data.edge_triple[j]))
                    for j, (s, t) in enumerate(data.edge_index.t().tolist()))
    return Counter(rows), edges, int(data.num_visits)


def without(graph, relation):
    return {**graph, 'edges': [e for e in graph['edges'] if e['relation'] != relation]}


def departures(graph, selected):
    """The two decided departures from a rebuild that a deletion of `selected` meets: the prior labs
    the deletion keeps although every index lab of their analyte is deleted (the builder pulls no
    history for them), and the analytes whose baseline_of edge a rebuild would re-anchor."""
    labs = defaultdict(lambda: {'index': [], 'prior': []})
    for i, node in enumerate(graph['nodes']):
        if node['kind'] == 'measurement':
            labs[node['token']][node['scope']].append(i)
    kept_priors, reanchored = set(), set()
    for token, lab in labs.items():
        if not lab['prior']:
            continue
        if all(i in selected for i in lab['index']):
            kept_priors |= {i for i in lab['prior'] if i not in selected}
            continue
        earliest = min(lab['index'], key=lambda i: graph['nodes'][i]['time_hours'])
        last_prior = max(lab['prior'], key=lambda i: graph['nodes'][i]['time_hours'])
        if (earliest in selected or last_prior in selected) and not all(i in selected for i in lab['prior']):
            reanchored.add(token)
    return kept_priors, reanchored


def baseline_tokens(graph):
    tokens = {n['id']: n['token'] for n in graph['nodes']}
    return Counter(tokens[e['target']] for e in graph['edges'] if e['relation'] == 'baseline_of')


@pytest.fixture(scope='module')
def built(table):
    graphs = {name: build_patient(name, table['rows']) for name in PATIENTS}
    for graph, membership in graphs.values():
        validate_visit_membership_record(graph, membership)
    return graphs, preprocessing([graph for graph, _ in graphs.values()])


def test_every_deletion_is_the_rebuild_up_to_the_two_decided_departures(table, built):
    """Every single eligible node, every pair and the whole eligible set of three builder patients:
    the deletion encodes to the tensors of the builder run without those events (up to storage order),
    once the two decided departures are accounted for: the prior labs the deletion keeps are deleted
    too, and baseline_of edges a rebuild re-anchors are left out of both; the departures themselves
    are asserted as differences."""
    graphs, prep = built
    seen = Counter()
    for name, (graph, membership) in graphs.items():
        eligible = [i for i, n in enumerate(graph['nodes'])
                    if n['kind'] in ('complaint', 'vital', 'measurement', 'diagnosis')]
        for selected in [(i,) for i in eligible] + list(itertools.combinations(eligible, 2)) + [tuple(eligible)]:
            rebuilt_graph, rebuilt_membership = decided_visits(
                membership, build_patient(name, table['rows'], source_events(name, graph, selected)))
            kept_priors, reanchored = departures(graph, set(selected))
            deleted_graph, deleted_membership, info = delete_from_record(
                graph, membership, sorted(set(selected) | kept_priors), combo_rules=table['rules'])
            validate_visit_membership_record(deleted_graph, deleted_membership)
            if kept_priors:
                seen['prior labs kept'] += 1
                plain = delete_from_record(graph, membership, selected, combo_rules=table['rules'])
                assert encoded_canon(prep, *plain[:2]) != encoded_canon(prep, rebuilt_graph, rebuilt_membership)
            if reanchored:
                seen['baseline not re-anchored'] += 1
                assert all(baseline_tokens(rebuilt_graph)[t] == 1 and baseline_tokens(deleted_graph)[t] == 0
                           for t in reanchored), (name, selected)
                deleted_graph, rebuilt_graph = without(deleted_graph, 'baseline_of'), without(rebuilt_graph,
                                                                                             'baseline_of')
            assert encoded_canon(prep, deleted_graph, deleted_membership) == encoded_canon(
                prep, rebuilt_graph, rebuilt_membership), (name, [graph['nodes'][i]['token'] for i in selected])
            combos = info['combos']
            seen['combo group deleted'] += combos['groups_deleted'] > 0
            seen['combo fired again'] += combos['refired'] > 0
            seen['combo re-evaluated unchanged'] += combos['reevaluated'] > combos['refired']
            seen['state the record did not have added'] += combos['new_state_nodes'] > 0
            seen['emptied visit dropped'] += info['visits_dropped'] > 0
            seen['originally empty visit kept'] += info['visits_originally_empty'] > 0
            assert combos['not_reevaluated'] == []
            assert combos['new_state_nodes'] == len(set(_tokens(deleted_graph)) - set(_tokens(graph)))
    assert len(seen) == 8 and min(seen.values()) > 0, seen


def _tokens(graph):
    return [n['token'] for n in graph['nodes']]


def _rule_edges(graph, target):
    tokens = {n['id']: n['token'] for n in graph['nodes']}
    return sorted((tokens[e['source']], e.get('rule_id')) for e in graph['edges']
                  if e['relation'] == 'medical:indicates' and tokens[e['target']] == target)


def _position(graph, token, value=None):
    (pos,) = [i for i, n in enumerate(graph['nodes'])
              if n['token'] == token and n['kind'] != 'analyte' and (value is None or n.get('value') == value)]
    return pos


def _same_as_rebuild(table, prep, name, graph, membership, selected, result):
    rebuilt = decided_visits(membership, build_patient(name, table['rows'], source_events(name, graph, selected)))
    return encoded_canon(prep, *result[:2]) == encoded_canon(prep, *rebuilt)


def test_sirs_through_one_combo_goes_with_one_of_its_components(table, built):
    graphs, prep = built
    graph, membership = graphs['P2']
    assert _rule_edges(graph, SIRS) == [('vital:heartrate', 'combo_sirs_temp_hr'),
                                        ('vital:temperature', 'combo_sirs_temp_hr')]
    selected = [_position(graph, 'vital:heartrate')]
    result = delete_from_record(graph, membership, selected, combo_rules=table['rules'])
    assert SIRS not in _tokens(result[0]) and _rule_edges(result[0], SIRS) == []
    # both SIRS rules read the heart rate; the one without a respiratory rate fires neither before nor after
    assert result[2]['combos'] == {'source': 'threshold_table', 'groups_deleted': 1, 'reevaluated': 2,
                                   'refired': 0, 'new_state_nodes': 0, 'not_reevaluated': []}
    assert _same_as_rebuild(table, prep, 'P2', graph, membership, selected, result)


def test_sirs_through_two_combos_keeps_the_combo_without_the_deleted_component(table, built):
    graphs, prep = built
    graph, membership = graphs['P1']
    assert _rule_edges(graph, SIRS) == [('vital:heartrate', 'combo_sirs_hr_rr'),
                                        ('vital:heartrate', 'combo_sirs_temp_hr'),
                                        ('vital:resprate', 'combo_sirs_hr_rr'),
                                        ('vital:temperature', 'combo_sirs_temp_hr')]
    selected = [_position(graph, 'vital:resprate')]
    result = delete_from_record(graph, membership, selected, combo_rules=table['rules'])
    assert _rule_edges(result[0], SIRS) == [('vital:heartrate', 'combo_sirs_temp_hr'),
                                            ('vital:temperature', 'combo_sirs_temp_hr')]
    assert QSOFA not in _tokens(result[0])                    # its other combo needed the same rate
    assert result[2]['combos']['groups_deleted'] == 2
    assert _same_as_rebuild(table, prep, 'P1', graph, membership, selected, result)


def test_a_deleted_latest_value_is_replaced_by_the_next_latest(table, built):
    graphs, prep = built
    graph, membership = graphs['P1']
    assert _rule_edges(graph, DKA) == [('lab:BIC', 'combo_dka'), ('lab:GLU', 'combo_dka')]
    latest, earlier = _position(graph, 'lab:GLU', 280.0), _position(graph, 'lab:GLU', 300.0)
    result = delete_from_record(graph, membership, [latest], combo_rules=table['rules'])
    tokens = {n['id']: (n['token'], n.get('value')) for n in result[0]['nodes']}
    assert sorted(tokens[e['source']] for e in result[0]['edges'] if e.get('rule_id') == 'combo_dka') == [
        ('lab:BIC', 15.0), ('lab:GLU', 300.0)]
    assert result[2]['combos'] == {'source': 'threshold_table', 'groups_deleted': 0, 'reevaluated': 1,
                                   'refired': 1, 'new_state_nodes': 0, 'not_reevaluated': []}
    assert _same_as_rebuild(table, prep, 'P1', graph, membership, [latest], result)
    # an earlier value is not what the builder evaluated: nothing changes for the combo
    result = delete_from_record(graph, membership, [earlier], combo_rules=table['rules'])
    assert _rule_edges(result[0], DKA) == _rule_edges(graph, DKA)
    assert result[2]['combos']['reevaluated'] == 1 and result[2]['combos']['refired'] == 0


def test_a_combo_that_only_an_earlier_value_fires_appears_as_in_the_rebuild(table, built):
    graphs, prep = built
    graph, membership = graphs['P2']
    assert DKA not in _tokens(graph)
    selected = [_position(graph, 'lab:GLU', 200.0)]
    result = delete_from_record(graph, membership, selected, combo_rules=table['rules'])
    assert _rule_edges(result[0], DKA) == [('lab:BIC', 'combo_dka'), ('lab:GLU', 'combo_dka')]
    assert result[2]['combos']['refired'] == 1
    assert result[2]['combos']['new_state_nodes'] == 1          # a state the record did not have
    assert _same_as_rebuild(table, prep, 'P2', graph, membership, selected, result)


def test_without_a_table_a_combo_group_goes_but_is_never_reevaluated(table, built):
    graphs, prep = built
    graph, membership = graphs['P2']
    selected = [_position(graph, 'vital:heartrate')]
    result = delete_from_record(graph, membership, selected)
    assert SIRS not in _tokens(result[0])
    assert result[2]['combos'] == {'source': 'distinct_anchor_tokens', 'groups_deleted': 1, 'reevaluated': 0,
                                   'refired': 0, 'new_state_nodes': 0,
                                   'not_reevaluated': [{'rule_id': 'combo_sirs_temp_hr',
                                                        'reason': 'no threshold table given'}]}
    assert _same_as_rebuild(table, prep, 'P2', graph, membership, selected, result)
    graph, membership = graphs['P1']
    selected = [_position(graph, 'lab:GLU', 280.0)]
    result = delete_from_record(graph, membership, selected)
    assert DKA not in _tokens(result[0])                      # the rebuild fires it on the earlier glucose
    assert not _same_as_rebuild(table, prep, 'P1', graph, membership, selected, result)
    # a state rule firing on two labs of one token is not a combo: its edges are alternatives
    state = copy.deepcopy(graphs['P1'])
    k1, k2 = _position(state[0], 'lab:K', 6.4), _position(state[0], 'lab:K', 4.1)
    hyperk = _position(state[0], 'knowledge:state:hyperkalemia')
    state[0]['edges'].append({'source': state[0]['nodes'][k2]['id'], 'target': state[0]['nodes'][hyperk]['id'],
                              'relation': 'medical:indicates', 'informative': True, 'rule_id': 'state_hyperk'})
    result = delete_from_record(*state, [k1])
    assert 'knowledge:state:hyperkalemia' in _tokens(result[0]) and result[2]['combos']['groups_deleted'] == 0


GLU_SBP = 'knowledge:state:glucose_sbp'


def test_deletion_and_the_labs_mask_differ_in_the_two_decided_ways(tmp_path):
    """A builder patient with a combo state fired by a lab and a vital, and a prior visit that is empty
    in the record: deleting every lab deletes the combo state (combo provenance) and keeps the empty
    visit, where the labs mask of source masking keeps the state through the surviving vital and drops
    that visit."""
    path = tmp_path / 'thresholds.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=THRESHOLD_COLUMNS)
        writer.writeheader()
        writer.writerows(RULES + [_rule('combo_glu_sbp', 'combo', 'lab:GLU', 'mg/dL', '>', '250', GLU_SBP),
                                  _rule('combo_glu_sbp', 'combo', 'vital:sbp', 'mmHg', '<=', '100', GLU_SBP)])
    rules = load_combo_rules(path, sha256(path), level_source='csv')
    graph, membership = build_patient('P1', read_thresholds(path))
    assert _rule_edges(graph, GLU_SBP) == [('lab:GLU', 'combo_glu_sbp'), ('vital:sbp', 'combo_glu_sbp')]
    empty = sorted(set(membership['visit_ordinals'])
                   - {o for o, pos in membership['membership_pairs'] if not membership['global_node_mask'][pos]})
    assert len(empty) == 1
    labs = [i for i, n in enumerate(graph['nodes']) if n['kind'] == 'measurement']
    deleted_graph, deleted_membership, info = delete_from_record(graph, membership, labs, combo_rules=rules)
    masked_graph, masked_membership, _ = apply_mask(graph, membership, 'labs')
    assert GLU_SBP not in _tokens(deleted_graph)
    assert GLU_SBP in _tokens(masked_graph)
    assert _rule_edges(masked_graph, GLU_SBP) == [('vital:sbp', 'combo_glu_sbp')]
    # both drop the prior visit whose only node was a prior lab; only the mask drops the empty one
    assert info['visits_originally_empty'] == 1 and info['visits_dropped'] == 1
    assert len(deleted_membership['visit_ordinals']) == len(membership['visit_ordinals']) - 1
    assert len(masked_membership['visit_ordinals']) == len(membership['visit_ordinals']) - 2
    # without the combo state and the empty visit the two records are the same
    state = _position(masked_graph, GLU_SBP)
    assert sorted(_tokens(deleted_graph)) == sorted(t for i, t in enumerate(_tokens(masked_graph)) if i != state)


def test_a_combo_input_missing_from_the_record_is_reported(table):
    """A graph from an event store without the v5 lab columns has no censoring sign or reference bounds
    on its lab nodes: the combo is deleted, not evaluated again, and the record names the inputs."""
    graph, membership = build_patient('P1', table['rows'], v5=False)
    result = delete_from_record(graph, membership, [_position(graph, 'lab:GLU', 280.0)],
                                combo_rules=table['rules'])
    assert DKA not in _tokens(result[0])
    assert result[2]['combos']['groups_deleted'] == 1 and result[2]['combos']['reevaluated'] == 0
    assert result[2]['combos']['not_reevaluated'] == [
        {'rule_id': 'combo_dka', 'reason': 'censored, ref_upper, ref_lower of the latest lab:GLU not in the record'}]


def test_only_the_visits_the_deletion_emptied_are_dropped(table, built):
    graphs, prep = built
    graph, membership = graphs['P1']
    assert sorted({o for o, _ in membership['membership_pairs']}) == [0, 2, 3]      # visit 1 has no node
    complaint = _position(graph, 'cc:dyspnea')
    result = delete_from_record(graph, membership, [complaint], combo_rules=table['rules'])
    assert result[1]['visit_ordinals'] == [0, 1, 2, 3]
    assert (result[2]['visits_dropped'], result[2]['visits_originally_empty']) == (0, 1)
    diagnosis = _position(graph, 'dx:I10')            # the only node of visit 0
    result = delete_from_record(graph, membership, [diagnosis], combo_rules=table['rules'])
    assert result[1]['visit_ordinals'] == [0, 1, 2]
    assert sorted({o for o, _ in result[1]['membership_pairs']}) == [1, 2]   # the empty visit, renumbered
    assert (result[2]['visits_dropped'], result[2]['visits_originally_empty']) == (1, 1)
    assert _same_as_rebuild(table, prep, 'P1', graph, membership, [diagnosis], result)


def test_prior_labs_of_an_analyte_without_index_labs_stay_unlike_a_rebuild(table, built):
    graphs, prep = built
    graph, membership = graphs['P1']
    index_lab, prior_lab = _position(graph, 'lab:CR', 1.9), _position(graph, 'lab:CR', 1.1)
    result = delete_from_record(graph, membership, [index_lab], combo_rules=table['rules'])
    assert {'lab:CR', 'knowledge:system:renal'} <= set(_tokens(result[0]))      # prior lab and analyte
    assert not _same_as_rebuild(table, prep, 'P1', graph, membership, [index_lab], result)
    rebuilt = build_patient('P1', table['rows'], source_events('P1', graph, [index_lab]))[0]
    assert 'lab:CR' not in _tokens(rebuilt) and 'knowledge:system:renal' not in _tokens(rebuilt)
    both = delete_from_record(graph, membership, [index_lab, prior_lab], combo_rules=table['rules'])
    assert _same_as_rebuild(table, prep, 'P1', graph, membership, [index_lab], both)


def test_a_baseline_edge_is_not_reanchored_unlike_a_rebuild(table, built):
    graphs, prep = built
    graph, membership = graphs['P2']
    earliest = _position(graph, 'lab:K', 5.9)
    result = delete_from_record(graph, membership, [earliest], combo_rules=table['rules'])
    rebuilt = decided_visits(membership, build_patient('P2', table['rows'], source_events('P2', graph, [earliest])))
    assert baseline_tokens(graph)['lab:K'] == 1 and baseline_tokens(result[0])['lab:K'] == 0
    tokens = {n['id']: (n['token'], n.get('value')) for n in rebuilt[0]['nodes']}
    assert [(tokens[e['source']], tokens[e['target']], e['delta']) for e in rebuilt[0]['edges']
            if e['relation'] == 'baseline_of'] == [(('lab:K', 4.8), ('lab:K', 5.2), pytest.approx(0.4))]
    assert encoded_canon(prep, *result[:2]) != encoded_canon(prep, *rebuilt)
    assert encoded_canon(prep, without(result[0], 'baseline_of'), result[1]) == encoded_canon(
        prep, without(rebuilt[0], 'baseline_of'), rebuilt[1])


def test_the_combo_rules_come_from_the_threshold_table(table, tmp_path):
    rules = table['rules']
    assert list(rules.by_rule) == ['combo_dka', 'combo_qsofa', 'combo_sirs_hr_rr', 'combo_sirs_temp_hr']
    assert all(row['kind'] == 'combo' for row in rules.rows) and len(rules.rows) == 8
    assert rules.by_rule['combo_dka'][0]['threshold'] == 250.0
    assert rules.describe() == {'source': 'threshold_table', 'path': str(table['path']),
                                'sha256': sha256(table['path']), 'sha256_checked': True, 'level_source': 'csv',
                                'combo_rules': 4, 'combo_rows': 8}
    unchecked = load_combo_rules(table['path'], None, level_source='row-range')
    assert unchecked.describe()['sha256'] == sha256(table['path']) and not unchecked.sha256_checked
    with pytest.raises(ValueError, match='the recorded one is'):
        load_combo_rules(table['path'], '0' * 64, level_source='csv')
    with pytest.raises(ValueError, match='unknown level source'):
        load_combo_rules(table['path'], None, level_source='textbook')
    with pytest.raises(ValueError, match='only rows of kind combo'):
        ComboRules(table['rows'], level_source='csv')
    assert combo_source(None)['source'] == 'distinct_anchor_tokens'
    assert combo_source(rules) == rules.describe()


@pytest.mark.parametrize('name', METHODS)
def test_the_scorer_deletes_with_the_combo_rules_it_was_given(table, built, name, tmp_path):
    graphs, prep = built
    graph, membership = graphs['P1']
    model = build_model(name, prep, tmp_path, graph=graph)
    latest = _position(graph, 'lab:GLU', 280.0)
    with_table = DeletionScorer(model_forward(model), graph, membership, encoder(prep), y=0,
                                combo_rules=table['rules'])
    fallback = DeletionScorer(model_forward(model), graph, membership, encoder(prep), y=0)
    assert with_table.combo_source['source'] == 'threshold_table'
    assert fallback.combo_source['source'] == 'distinct_anchor_tokens'
    expected = delete_from_record(graph, membership, [latest], combo_rules=table['rules'])
    assert torch.equal(with_table.probs([latest]), reference_probs(model, prep, *expected[:2]))
    assert with_table.last_info == expected[2]
    assert fallback.probs([latest]).shape == with_table.probs([latest]).shape
    assert fallback.last_info['combos']['not_reevaluated'][0]['rule_id'] == 'combo_dka'
