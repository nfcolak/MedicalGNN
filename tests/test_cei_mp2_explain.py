"""CEI-GNN mp2 under the explanation runner: the message-passing edge-mask hook
(cei/layers.py), the evidence-ledger built-in and `build_cei_model` (core/explain/cei_bridge.py).

The model under test is the real EvidenceAdapterMP2 with the final run's message-passing options
on the synthetic v3 graph (fixtures from tests/test_cei_gnn_v3_core.py), extended by a parallel
copy and two mirrored edges so that mp_dedup's copy split and mirror rule are exercised.
Parameters are perturbed away from their zero-init so the neighbourhood rows and the residual
message terms are live.
"""
import copy
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch

from core.explain import graphxai_standardized as gx
from cei.options import DEFAULTS
from core.explain import runner
from core.explain.cei_bridge import (
    CEI_METHODS, INTERVENTION_SCOPES, CEIGraphXAIWrapper, build_cei_model,
)
from core.registry import METHOD_REGISTRY
from tests.test_cei_gnn_v3_core import (
    CLASSES, EDGE_DIM, HIDDEN, K, LAYOUT, NODE_DIM, NUM_TOKENS, TOKEN_DIM, TRIPLES, _graph,
    _write_state,
)

# message-passing options of the final cei_gnn_mp2 runs (binding method_config.mp2_options)
FINAL = dict(mp_feed='dual', mp_hub_max=1, mp_dedup=1, mp_ctx='table', mp_root='bias',
             mp_aggr='relation', mp_layers=2, mp_optimizer='adamw', mp_lr_schedule='cosine',
             mp_label_smoothing=0.1)
EXTRA_EDGES = ((1, 3, 3, 2),      # parallel copy of edge 2 (1 -> 3, same relation)
               (5, 3, 4, 1),      # mirror of edge 4 (3 -> 5)
               (2, 1, 1, 0))      # mirror of edge 0 (1 -> 2) in another relation
PARALLEL, MIRROR, MIRRORED, PLAIN = 6, 7, 4, 3


def _mp2_graph():
    graph = _graph()
    generator = torch.Generator().manual_seed(3)
    extra = torch.tensor(EXTRA_EDGES).t()
    graph.edge_index = torch.cat((graph.edge_index, extra[:2]), dim=1)
    graph.edge_relation = torch.cat((graph.edge_relation, extra[2]))
    graph.edge_triple = torch.cat((graph.edge_triple, extra[3]))
    graph.edge_attr = torch.cat((graph.edge_attr, torch.randn(extra.size(1), EDGE_DIM,
                                                              generator=generator)))
    return graph


def _without(graph, removed):
    graph = copy.deepcopy(graph)
    keep = torch.ones(graph.edge_index.size(1), dtype=torch.bool)
    keep[list(removed)] = False
    graph.edge_index = graph.edge_index[:, keep]
    for key in ('edge_attr', 'edge_relation', 'edge_triple'):
        setattr(graph, key, getattr(graph, key)[keep])
    return graph


def _mp2_adapter(tmp_path, *, dropout=0.0, weights=None, perturb=True, **options):
    args = Namespace(method_options={'arm': 'C', 'v3_state': str(_write_state(tmp_path)), 'k': K,
                                     **FINAL, **options}, seed=1234)
    if weights is not None:
        args.weights = weights
    torch.manual_seed(1234)
    adapter = METHOD_REGISTRY['cei_gnn_mp2'](
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
        hidden=HIDDEN, layers=1, dropout=dropout, token_dim=TOKEN_DIM, num_triples=TRIPLES,
        args=args)
    if perturb:
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.add_(0.3 * torch.randn(parameter.shape, generator=generator))
    return adapter.eval()


def _batch(graph=None):
    return Batch.from_data_list([graph if graph is not None else _mp2_graph()])


@pytest.fixture
def mp2(tmp_path):
    adapter = _mp2_adapter(tmp_path)
    wrapper = CEIGraphXAIWrapper(adapter, method='cei_gnn_mp2')
    batch = _batch()
    return adapter, wrapper, batch, wrapper.set_context(batch)


def _binding(adapter, **override):
    return {'method': 'cei_gnn_mp2', 'conv': None, 'node_dim': NODE_DIM, 'edge_dim': EDGE_DIM,
            'num_classes': CLASSES, 'vocabulary_size': NUM_TOKENS, 'num_meta_relations': TRIPLES,
            'hidden': HIDDEN, 'layers': 1, 'edge_direction': 'forward', 'seed': 1234,
            'method_config': adapter.run_config()} | override


# ------------------------------------------------------------------ mask = None
def test_without_a_mask_the_wrapper_is_the_adapter_bit_for_bit(mp2):
    adapter, wrapper, batch, x = mp2
    with torch.no_grad():
        reference = adapter(batch, epoch=0).logits
        assert torch.equal(wrapper(x, batch.edge_index), reference)
        wrapper(x, batch.edge_index, edge_mask=torch.zeros(batch.edge_index.size(1)))
        assert torch.equal(wrapper(x, batch.edge_index), reference)       # nothing left behind
    assert wrapper.verify(x, batch.edge_index) == (True, 0.0)


def test_an_all_ones_mask_is_exactly_no_mask(mp2):
    _, wrapper, batch, x = mp2
    with torch.no_grad():
        ones = wrapper(x, batch.edge_index, edge_mask=torch.ones(batch.edge_index.size(1)))
        assert torch.equal(ones, wrapper(x, batch.edge_index))


def test_the_stack_hook_is_inactive_by_default(mp2):
    adapter, _, batch, _ = mp2
    seen = []
    stack = adapter.network.mp_stack
    original = stack.forward

    def spy(*args, **kwargs):
        seen.append(kwargs.get('edge_weight'))
        return original(*args, **kwargs)

    stack.forward = spy
    try:
        with torch.no_grad():
            adapter(batch, epoch=0)
    finally:
        del stack.forward
    assert seen == [None]


# ------------------------------------------------------------------ mask 0 = deletion
@pytest.mark.parametrize('removed', [(PLAIN,), (PARALLEL,), (2,), (MIRROR,), (MIRRORED,),
                                     (MIRROR, PARALLEL), tuple(range(9))],
                         ids=['plain', 'parallel_copy', 'its_twin', 'mirror', 'mirrored',
                              'mirror_and_copy', 'all'])
def test_mask_zero_is_the_same_intervention_as_deleting_the_edge(mp2, removed):
    adapter, wrapper, batch, x = mp2
    mask = torch.ones(batch.edge_index.size(1))
    mask[list(removed)] = 0.0
    with torch.no_grad():
        masked = wrapper(x, batch.edge_index, edge_mask=mask)
        deleted = adapter(_batch(_without(_mp2_graph(), removed)), epoch=0).logits
        full = adapter(batch, epoch=0).logits
    assert torch.allclose(masked, deleted, atol=1e-6), (masked, deleted)
    assert not torch.allclose(deleted, full, atol=1e-6)        # the deletion matters


def test_the_stack_alone_gives_deletion_for_dedup_copies_and_mirrors(mp2):
    """Layer-level: the message-passing state equals the deleted graph's for every node."""
    adapter, _, batch, _ = mp2
    network = adapter.network
    graph = _mp2_graph()

    def state(graph, edge_weight=None):
        clinical, _, _ = adapter._read(_batch(graph))
        features = network.continuous_inputs(clinical)
        h0, node_time, _ = network._encode(features, clinical, clinical.edge_index)
        context = (network.relation_embedding(clinical.edge_relation)
                   + network.triple_embedding(clinical.edge_triple)
                   + network.edge_feature_projection(clinical.edge_attr))
        return network.mp_stack(h0, clinical.edge_index, context, clinical,
                                edge_weight=edge_weight, node_time=node_time)[0]

    mask = torch.ones(graph.edge_index.size(1))
    mask[[PARALLEL, MIRROR]] = 0.0
    with torch.no_grad():
        assert torch.allclose(state(graph, mask), state(_without(graph, (PARALLEL, MIRROR))),
                              atol=1e-6)


# ------------------------------------------------------------------ fractional mask, once
@pytest.mark.parametrize('value', [0.5, 0.2])
def test_a_fractional_mask_reaches_the_stack_and_the_edge_rows_exactly_once(mp2, value):
    adapter, wrapper, batch, x = mp2
    edges = batch.edge_index
    mask = torch.full((edges.size(1),), value)
    stack = wrapper.inner.network.mp_stack
    original, seen = stack.forward, []

    def spy(*args, **kwargs):
        seen.append(kwargs['edge_weight'].detach().clone())
        return original(*args, **kwargs)

    stack.forward = spy
    try:
        with torch.no_grad():
            got = wrapper(x, edges, edge_mask=mask)
    finally:
        del stack.forward
    assert len(seen) == 1 and torch.equal(seen[0], mask)        # not mask * mask
    reference = copy.deepcopy(adapter)
    reference.network.set_edge_mask(mask)                        # his public hook, independent
    with torch.no_grad():
        expected = reference.forward_continuous(x, edges, batch)
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    with torch.no_grad():
        assert not torch.allclose(got, wrapper(x, edges), atol=1e-6)


# ------------------------------------------------------------------ evidence ledger
def test_the_ledger_rows_sum_to_the_logits_for_every_class(mp2):
    adapter, _, batch, _ = mp2
    network = adapter.network
    clinical, membership, visit_graph = adapter._read(batch)
    with torch.no_grad():
        parts = network.forward_continuous(network.continuous_inputs(clinical), clinical.edge_index,
                                           clinical, membership, return_parts=True,
                                           visit_graph=visit_graph)
    keys = network.ledger_keys()
    assert 'nbr_contributions' in keys and 'absence_contributions' in keys
    total = parts['bias'] + sum(parts[key].sum(0) for key in keys)
    assert torch.allclose(total, parts['logits'][0], atol=1e-5)
    assert parts['nbr_contributions'].abs().sum() > 0                    # the mp rows are live


def test_the_builtin_is_the_exact_ledger_accounting_with_neighbourhood_rows(mp2):
    _, wrapper, batch, x = mp2
    importance = wrapper.builtin_node_importance(x, batch.edge_index)
    detail = wrapper.builtin_detail()
    with torch.no_grad():
        logits = wrapper(x, batch.edge_index)
    prediction = int(logits.argmax(-1))
    assert importance.shape == (8,) and (importance >= 0).all()
    assert detail['explained_class'] == prediction
    assert detail['accounting_logit_abs_diff'] < 1e-5
    assert set(detail['ledger_block_totals']) == set(wrapper.adapter.network.ledger_keys())
    signed = torch.tensor(detail['signed_node_contributions'])
    rebuilt = signed.sum() + detail['bias'] + detail['fixed_absence_contribution']
    assert abs(float(rebuilt) - float(logits[0, prediction])) < 1e-5
    assert detail['intervention_scope'] == INTERVENTION_SCOPES['cei_gnn_mp2']
    other = (prediction + 1) % CLASSES
    redirected = wrapper.builtin_node_importance(x, batch.edge_index, target_class=other)
    assert wrapper.builtin_detail()['explained_class'] == other
    assert not torch.allclose(importance, redirected)


def test_a_non_additive_ledger_has_no_builtin(tmp_path):
    adapter = _mp2_adapter(tmp_path, ev_combine='mlp')
    wrapper = CEIGraphXAIWrapper(adapter, method='cei_gnn_mp2')
    batch = _batch()
    x = wrapper.set_context(batch)
    with pytest.raises(NotImplementedError, match='additive'):
        wrapper.builtin_node_importance(x, batch.edge_index)


# ------------------------------------------------------------------ GraphXAI algorithms
def test_the_three_graphxai_algorithms_run_on_mp2_and_leave_no_mask_behind(mp2):
    adapter, wrapper, batch, x = mp2
    result = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=3)
    assert all(result[a]['status'] == 'success' for a in gx.ALGORITHMS), result
    assert result['GNNExplainer']['provenance']['edge_gradient_verified'] is True
    assert result['GNNExplainer']['provenance']['edge_gradient_l1'] > 0
    assert gx._model_declares_edge_mask(wrapper) is False
    assert not wrapper.inner.network.edge_aggregator.explain
    assert adapter.network.edge_aggregator._edge_mask is None     # the checkpoint copy is untouched


def test_the_whole_subject_pipeline_runs_on_mp2(mp2):
    _, wrapper, _, _ = mp2
    data = _mp2_graph()
    data.sample_id, data.subject = 'mp2-1', 'p1'
    ig = {'layout': LAYOUT, 'steps': 4, 'tolerance': 0.05, 'step_selection': {}, 'bank': None,
          'reference_draws': 0, 'keep_zero_baseline': True, 'seed': 1,
          'feature_spec': wrapper.ig_feature_spec()}
    record = runner.explain_subject(wrapper, data, method='cei_gnn_mp2', conv=None, steps=2,
                                    epochs=3, num_relation_columns=15, ig=ig, random_repeats=3)
    assert record['builtin_available'] is True
    assert record['builtin_detail']['accounting_logit_abs_diff'] < 1e-4
    for algorithm in gx.ALGORITHMS:
        assert record['graphxai'][algorithm]['status'] == 'success'
    assert record['wrapper_verification']['max_abs_logit_diff'] <= 1e-5


# ------------------------------------------------------------------ build_cei_model
def test_build_cei_model_rebuilds_mp2_from_its_recorded_options(mp2):
    adapter, _, batch, _ = mp2
    model = build_cei_model(_binding(adapter), adapter.state_dict())
    assert not model.training
    with torch.no_grad():
        assert torch.equal(model(batch, epoch=0).logits, adapter(batch, epoch=0).logits)
    rebuilt, is_adapter = runner.build_model(_binding(adapter), adapter.state_dict())
    assert is_adapter is True and not rebuilt.training


def test_build_cei_model_keeps_the_recorded_dropout_and_class_weight_policy(tmp_path):
    adapter = _mp2_adapter(tmp_path, dropout=0.3, weights='sqrt_inverse')
    binding = _binding(adapter, weights='sqrt_inverse')
    assert binding['method_config']['aux_ce_class_weights'] == 'sqrt_inverse'
    model = build_cei_model(binding, adapter.state_dict())
    batch = _batch()
    with torch.no_grad():
        first, second = model(batch, epoch=0).logits, model(batch, epoch=0).logits
        assert torch.equal(first, second) and torch.equal(first, adapter(batch, epoch=0).logits)


def test_build_cei_model_refuses_other_methods(mp2):
    adapter, *_ = mp2
    for method in ('gsat', 'protgnn', 'clinical_gnn'):
        with pytest.raises(ValueError, match='CEI explanation supports'):
            build_cei_model(_binding(adapter, method=method), adapter.state_dict())
    assert CEI_METHODS == ('cei_gnn_v3', 'cei_gnn_mp2')


def test_build_cei_model_refuses_a_mismatched_method_config(mp2, tmp_path):
    adapter, *_ = mp2
    tampered = adapter.run_config()
    tampered['parameter_count'] += 1
    with pytest.raises(ValueError, match='configuration differs'):
        build_cei_model(_binding(adapter, method_config=tampered), adapter.state_dict())
    options = adapter.run_config()
    options['mp2_options'] = dict(options['mp2_options'], mp_hub_max=0)
    with pytest.raises(ValueError, match='configuration differs'):
        build_cei_model(_binding(adapter, method_config=options), adapter.state_dict())
    v3_config = dict(adapter.run_config(), method='cei_gnn_v3')
    with pytest.raises(ValueError, match='method_config records'):
        build_cei_model(_binding(adapter, method_config=v3_config), adapter.state_dict())


ABLATION_SWITCHES = ('ev_combine', 'ev_pair', 'ev_edge', 'readout', 'pool_hidden')


def _predating(config, names=ABLATION_SWITCHES):
    """A method_config as recorded by a run trained before `names` existed."""
    old = copy.deepcopy(config)
    for section in ('architecture', 'effective_settings', 'mp2_options'):
        if isinstance(old.get(section), dict):
            for name in names:
                old[section].pop(name, None)
    return old


def test_a_run_that_predates_later_options_rebuilds_when_they_hold_their_defaults(mp2):
    """The final runs were trained before the section-4 ablation switches existed; their
    binding.json lacks them. At their defaults (the earlier behaviour) the run rebuilds."""
    adapter, _, batch, _ = mp2
    config = adapter.run_config()
    assert all(config['mp2_options'][name] == DEFAULTS[name] for name in ABLATION_SWITCHES)
    model = build_cei_model(_binding(adapter, method_config=_predating(config)), adapter.state_dict())
    with torch.no_grad():
        assert torch.equal(model(batch, epoch=0).logits, adapter(batch, epoch=0).logits)


def test_a_predating_run_still_refuses_any_recorded_difference(mp2):
    adapter, *_ = mp2
    old = _predating(adapter.run_config())
    old['mp2_options'] = dict(old['mp2_options'], mp_hub_max=0)
    with pytest.raises(ValueError, match='configuration differs'):
        build_cei_model(_binding(adapter, method_config=old), adapter.state_dict())


def test_a_later_option_away_from_its_default_is_not_dropped():
    from core.explain.cei_bridge import _comparable_run_config

    rebuilt = {'method': 'cei_gnn_mp2', 'mp2_options': {'mp_layers': 2, 'readout': 'pool'}}
    recorded = {'method': 'cei_gnn_mp2', 'mp2_options': {'mp_layers': 2}}
    assert _comparable_run_config(rebuilt, recorded) != recorded
    rebuilt['mp2_options']['readout'] = 'evidence'
    assert _comparable_run_config(rebuilt, recorded) == recorded


def test_build_cei_model_loads_the_checkpoint_strictly(mp2):
    adapter, *_ = mp2
    state = dict(adapter.state_dict(), stray=torch.zeros(1))
    with pytest.raises(RuntimeError, match='stray'):
        build_cei_model(_binding(adapter), state)


def test_the_wrapper_refuses_a_method_that_is_not_the_adapters(mp2):
    adapter, wrapper, *_ = mp2
    assert wrapper.intervention_scope == INTERVENTION_SCOPES['cei_gnn_mp2']
    with pytest.raises(ValueError, match='does not match'):
        CEIGraphXAIWrapper(adapter, method='cei_gnn_v3')
    with pytest.raises(ValueError, match='CEI explanation supports'):
        CEIGraphXAIWrapper(adapter, method='gsat')
