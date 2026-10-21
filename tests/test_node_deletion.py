"""Node deletion at the graph-record level (core/explain/node_deletion.py), the operator behind the
fidelity metrics of intervention contract version 5.

Synthetic graph record shaped like the graph builder's output (data/s4_graph/graph.py): a global
patient, the index visit (last ordinal), two complaints, a vital, two index labs of one analyte and
one of another, a prior lab of the second analyte, prior diagnoses and knowledge nodes (a fired
threshold state, a diagnosis concept, a complaint concept). It is encoded with a hand-fitted
preprocessing and scored by the four final methods (gsat, graphcare, protgnn, cei_gnn_mp2) with
perturbed weights. No data file, run or checkpoint is read.
"""
from __future__ import annotations

import copy
import json
from argparse import Namespace
from collections import Counter

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from cei.studies.cei_v3_absence import fit_universe
from cei.studies.cei_v3_ple import fit_knots
from comparison.s7_why.masking import apply_mask
from core.contracts import VISIT_MEMBERSHIP_CONTRACT_VERSION, validate_visit_membership_record
from core.explain.node_deletion import DeletionScorer, delete_from_record, encoding_differences
from core.registry import METHOD_REGISTRY, build_method
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
                    'visits_dropped': 3 - len(set(ordinal_map.values())),
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
                    'visits_dropped': 0, 'nodes_after': 16, 'edges_after': len(graph['edges'])}


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


def test_deleting_every_lab_is_the_source_masking_labs_record():
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


def build_model(name, prep, tmp_path, seed=5):
    dims = dict(num_tokens=len(prep['vocabulary']), node_dim=len(node_feature_layout(prep)),
                edge_dim=len(ALL_RELATIONS) + PAYLOAD_WIDTH, num_classes=CLASSES, hidden=8,
                token_dim=4, num_triples=triple_count(prep))
    torch.manual_seed(seed)
    if name == 'cei_gnn_mp2':
        state = _v3_state(prep, tmp_path / 'K4.json', record()[0])
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
