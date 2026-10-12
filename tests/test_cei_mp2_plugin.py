"""Regression tests of the ``cei_gnn_mp2`` option table, plugin and pre-training CLI.

Finding ids (mp2v3-*, mp2rev-*, LCL-*) are in the test names/docstrings. Synthetic tensors only.
"""
from __future__ import annotations

import warnings

import pytest

from cei.options import DEFAULTS, LEGACY, ignored_options, validate


def _options(**raw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return validate(raw)


# ------------------------------------------------------------------ ignored_options

IGNORED_CASES = [
    # mp2v3-model-04, mp2rev-layers-08: mp_aggr groups nothing
    (dict(mp_degree_norm='sym', mp_aggr='triple'), 'mp_aggr'),
    (dict(mp_degree_norm='sym', mp_aggr='mean'), 'mp_aggr'),
    (dict(mp_kind='hgt', mp_aggr='mean'), 'mp_aggr'),
    (dict(mp_kind='simplehgn', mp_aggr='triple'), 'mp_aggr'),
    # no residual, so no initial residual scale
    (dict(mp_residual='none', mp_residual_init=0.5), 'mp_residual_init'),
    # the exempt list only filters edges that would be dropped
    (dict(mp_dropedge=0.0, mp_dropedge_exempt=''), 'mp_dropedge_exempt'),
    # the KG-drop branch keeps `~g.kg` and never reads the exempt set (layers.py
    # `_edge_weights`: exempt is OR-ed in only inside `if o.dropedge > 0.0`)
    (dict(mp_dropedge=0.0, mp_kg_edge_drop=0.5, mp_dropedge_exempt='has_visit'), 'mp_dropedge_exempt'),
    # mp2v3-fixes-04: the stack resets hub_max for non-gine kinds
    (dict(mp_kind='rgcn', mp_hub_max=1, mp_hub_kinds='visit'), 'mp_hub_kinds'),
    (dict(mp_kind='hgt', mp_hub_max=1, mp_hub_kinds='visit'), 'mp_hub_kinds'),
]


@pytest.mark.parametrize('raw,name', IGNORED_CASES)
def test_ignored_options_lists_an_option_the_stack_does_not_read(raw, name):
    assert name in ignored_options(_options(**raw)), f'{name} has no effect for {raw} but is not listed'


@pytest.mark.parametrize('raw,name', [
    (dict(mp_aggr='triple'), 'mp_aggr'),                                   # gine + mean degree norm: used
    (dict(mp_degree_norm='null', mp_aggr='triple'), 'mp_aggr'),
    (dict(mp_residual_init=0.5), 'mp_residual_init'),                      # channel residual: used
    (dict(mp_dropedge_exempt='has_visit'), 'mp_dropedge_exempt'),          # DEFAULT mp_dropedge 0.2: used
    (dict(mp_kg_edge_drop=0.1, mp_dropedge=0.3, mp_dropedge_exempt='has_visit'), 'mp_dropedge_exempt'),
    (dict(mp_hub_max=1, mp_hub_kinds='visit'), 'mp_hub_kinds'),            # gine with hub_max: used
])
def test_ignored_options_does_not_list_an_option_that_is_read(raw, name):
    assert name not in ignored_options(_options(**raw))


def test_default_and_legacy_list_nothing_as_ignored():
    assert ignored_options(validate({})) == {}
    assert ignored_options(validate(dict(LEGACY))) == {}


# ------------------------------------------------------------------ shared builders

from argparse import Namespace  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from cei.layers import MPStack, _Factorized  # noqa: E402
from cei.options import check_build, effective_options  # noqa: E402
from core import NODE_KINDS  # noqa: E402
from core.registry import METHOD_REGISTRY  # noqa: E402
from core.tensorize import relation_vocabulary  # noqa: E402
from tests.test_cei_gnn_v3_core import (CLASSES, EDGE_DIM, HIDDEN, K, NODE_DIM, NUM_TOKENS,  # noqa: E402
                                        TOKEN_DIM, TRIPLES, _write_state)

STACK_WIDTH = 16
RELATIONS = tuple(relation_vocabulary('bidirectional'))


def _stack(options):
    return MPStack(STACK_WIDTH, dropout=0.0, seed=5, name='mp_layers', num_relations=len(RELATIONS),
                   num_triples=4, num_node_types=len(NODE_KINDS), relation_names=RELATIONS,
                   node_kinds=NODE_KINDS, options=options)


def _build(tmp_path, *, seed=1234, dropout=0.0, hidden=HIDDEN, **options):
    state_path = _write_state(Path(tmp_path))
    method_options = {'arm': 'C', 'v3_state': str(state_path), 'k': K, **options}
    args = Namespace(method_options=method_options, seed=seed, lr=1e-3, weight_decay=4.3e-5)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return METHOD_REGISTRY['cei_gnn_mp2'](
            num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
            hidden=hidden, layers=1, dropout=dropout, token_dim=TOKEN_DIM, num_triples=TRIPLES, args=args)


# ------------------------------------------------------------------ effective_options

RANK_CASES = [   # (mp_msg_dim, mp_rank, mp_global) on a stack of width 16
    (4, 6, 0),    # rank >= msg_dim, maps stay dense
    (8, 12, 0),   # msg_dim 8 < rank 12 < width 16, maps stay dense
    (8, 4, 0),    # factorised
    (0, 12, 0),   # msg_dim 0 = full width 16, rank 12 < 16: factorised
    (0, 20, 0),   # rank above the width: dense
    (4, 6, 1),    # glob (width x width) is factorised by rank 6 even though the message maps are not
    (4, 0, 1),
    (0, 16, 1),   # rank == width: dense everywhere
]


@pytest.mark.parametrize('msg_dim,rank,glob', RANK_CASES)
def test_effective_mp_rank_differs_from_the_full_rank_value_only_if_a_map_is_factorised(msg_dim, rank, glob):
    """the record must not claim a low-rank model that is dense."""
    base = dict(mp_msg_dim=msg_dim, mp_global=glob)
    options = _options(mp_rank=rank, **base)
    stack = _stack(options)
    factorised = any(isinstance(module, _Factorized) for module in stack.modules())
    recorded = effective_options(options, hidden=STACK_WIDTH)['mp_rank']
    full = effective_options(_options(mp_rank=0, **base), hidden=STACK_WIDTH)['mp_rank']
    assert (recorded != full) == factorised, (recorded, full, factorised)
    if factorised:
        assert recorded == rank


@pytest.mark.parametrize('kind', ['rgcn', 'diag', 'simplehgn', 'hgt'])
def test_effective_mp_rank_of_a_non_gine_kind_is_the_full_value(kind):
    """The stack ignores mp_rank for the other kinds (their context path has its own fixed rank)."""
    ranked, plain = _stack(_options(mp_kind=kind, mp_rank=5)), _stack(_options(mp_kind=kind))
    assert ({k: tuple(v.shape) for k, v in ranked.state_dict().items()}
            == {k: tuple(v.shape) for k, v in plain.state_dict().items()})
    assert (effective_options(_options(mp_kind=kind, mp_rank=5), hidden=STACK_WIDTH)['mp_rank']
            == effective_options(_options(mp_kind=kind), hidden=STACK_WIDTH)['mp_rank'])


def test_run_config_records_the_dropout_the_stack_really_uses(tmp_path):
    """mp_dropout=-1 means the network dropout; the record must show the rate, not the sentinel."""
    inherited = _build(tmp_path, dropout=0.3)
    assert inherited.network.mp_stack.layers[0].dropout_rate == 0.3
    assert inherited.run_config()['mp2_effective']['mp_dropout'] == 0.3
    own = _build(tmp_path, dropout=0.3, mp_dropout=0.1)
    assert own.network.mp_stack.layers[0].dropout_rate == 0.1
    assert own.run_config()['mp2_effective']['mp_dropout'] == 0.1


# ------------------------------------------------------------------ check_build / docstring

NO_NOISE = dict(mp_dropedge=0.0, mp_kg_edge_drop=0.0, mp_dropmsg=0.0, mp_dropnode=0.0, mp_droppath=0.0,
                row_drop=0.0, mp_dropout=-1.0)


def test_check_build_accepts_consistency_when_only_the_node_mask_is_stochastic(tmp_path):
    """LCL-02 draws a fresh mask per training forward, so the two consistency views differ."""
    from torch_geometric.data import Batch
    from core.method_base import read_clinical_batch
    from tests.test_cei_gnn_mp import _large_graph

    net = _build(tmp_path, dropout=0.0, mp_mask_rate=0.4, **NO_NOISE).network
    net.train()
    batch = Batch.from_data_list([_large_graph(20 + 3 * i, i) for i in range(4)])
    meta = read_clinical_batch(batch, method='t', node_dim=net.node_dim, edge_dim=net.edge_dim,
                               num_tokens=net.num_tokens, num_triples=net.num_triples,
                               num_relations=net.num_relations)
    visit_graph = torch.repeat_interleave(torch.arange(meta.graph_count), batch.num_visits.view(-1).long())
    features = net.continuous_inputs(meta)
    first, second = (net.forward_continuous(features, meta.edge_index, meta, batch.visit_membership_index,
                                            visit_graph=visit_graph) for _ in range(2))
    assert (first - second).abs().max() > 0, 'precondition: the two views differ'
    options = _options(mp_consistency=0.5, mp_mask_rate=0.4, **NO_NOISE)
    check_build(options, hidden=HIDDEN, dropout=0.0)   # must not raise


def test_check_build_still_refuses_consistency_without_any_noise():
    options = _options(mp_consistency=0.5, **NO_NOISE)
    with pytest.raises(ValueError, match='mp_consistency'):
        check_build(options, hidden=HIDDEN, dropout=0.0)


def test_options_docstring_counts_the_exclusive_maximum_names():
    """(mp2v3-model-05): the module docstring names the size of EXCLUSIVE_MAXIMUM."""
    import re
    import cei.options as options_module
    words = {'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
    found = re.search(r'the\s+(\w+)\s+names\s+in\s+``EXCLUSIVE_MAXIMUM``', options_module.__doc__)
    assert found and words[found.group(1)] == len(options_module.EXCLUSIVE_MAXIMUM)


# ------------------------------------------------------------------ plugin record / pre-training load

CONTROL_HIDDEN = 128   # cei_gnn_v3 CONTROL_HIDDEN: the width at which C's initial values can be shared


def _c_reference(tmp_path, hidden):
    """The tensors of the cei_gnn_v3 arm-C adapter that cei_gnn_mp2 starts from at the same seed."""
    args = Namespace(method_options={'arm': 'C', 'v3_state': str(_write_state(Path(tmp_path))), 'k': K}, seed=1234)
    torch.manual_seed(1234)
    adapter = METHOD_REGISTRY['cei_gnn_v3'](
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES, hidden=hidden,
        layers=1, dropout=0.0, token_dim=TOKEN_DIM, num_triples=TRIPLES, args=args)
    return {name: tensor.clone() for name, tensor in adapter.state_dict().items()}


@pytest.mark.parametrize('option', [
    dict(), dict(mp_feed='dual'), dict(mp_dual_read=1), dict(mp_layers=3), dict(mp_mask_rate=0.2), dict(row_drop=0.1),
    dict(edge_head_rank=200),   # >= hidden: v3's own heads are kept
    dict(shared_gate=1), dict(edge_head_rank=16), dict(kind_norm=1), dict(ple_mode='robust_cum'),
], ids=lambda option: ','.join(f'{k}={v}' for k, v in option.items()) or 'default')
def test_common_init_identical_to_c_matches_the_tensors_c_also_has(tmp_path, option):
    """the flag is True exactly when every tensor C has starts from C's initial values."""
    reference = _c_reference(tmp_path, CONTROL_HIDDEN)
    adapter = _build(tmp_path, hidden=CONTROL_HIDDEN, **option)
    own = adapter.state_dict()
    identical = all(name in own and own[name].shape == tensor.shape and torch.equal(own[name], tensor)
                    for name, tensor in reference.items())
    assert adapter.common_init_identical_to_c() is identical
    assert adapter.run_config()['common_init_identical_to_c'] is identical


def test_architecture_record_has_one_unambiguous_layer_count(tmp_path):
    """(mp2v3-model-02): v3's encoder depth 'layers' must not sit next to mp_layers."""
    architecture = _build(tmp_path).run_config()['architecture']
    assert architecture['mp_layers'] == 2
    assert 'layers' not in architecture
    assert architecture['encoder_layers'] == 1


def _write_checkpoint(tmp_path, adapter, name='pre.pt'):
    """A pre-training checkpoint plus the sidecar ``cei.pretrain`` writes."""
    import json
    from cei.plugin_cei_gnn_mp2 import checkpoint_config, checkpoint_state
    path = Path(tmp_path) / name
    torch.save(checkpoint_state(adapter.network), path)
    path.with_suffix(path.suffix + '.json').write_text(json.dumps({'config': checkpoint_config(adapter)}))
    return path


def test_a_checkpoint_loads_into_the_same_configuration(tmp_path):
    path = _write_checkpoint(tmp_path, _build(tmp_path))
    loaded = _build(tmp_path, mp_pretrain_checkpoint=str(path))
    assert loaded.pretrain_info['config_checked'] is True


@pytest.mark.parametrize('other', [dict(mp_layers=3), dict(mp_norm='graph')])
def test_a_checkpoint_is_refused_by_a_network_with_another_depth_or_norm(tmp_path, other):
    """mp_layers / mp_norm change what the tensors mean; the config check must cover them."""
    path = _write_checkpoint(tmp_path, _build(tmp_path))
    with pytest.raises(ValueError, match='another configuration'):
        _build(tmp_path, mp_pretrain_checkpoint=str(path), **other)


# ------------------------------------------------------------------ cei.pretrain

def _pretrain_setup(tmp_path, *, sample_ids=None, test_ids=('t1',), count=6):
    """Synthetic graphs + id lists for ``cei.pretrain.main``; ``sample_ids`` = the graphs' own ids."""
    import json
    from tests.test_cei_gnn_mp import _large_graph
    tmp = Path(tmp_path)
    state = _write_state(tmp)
    graphs = [_large_graph(20 + i, i) for i in range(count)]
    if sample_ids is not None:
        for graph, sample_id in zip(graphs, sample_ids):
            graph.sample_id = sample_id
    torch.save(graphs, tmp / 'g.pt')
    lists = {'ids': [f'g{i}' for i in range(count)], 'dev': ['d1'], 'val': ['v1'], 'test': list(test_ids),
             'train': ['tr1']}
    for name, values in lists.items():
        (tmp / f'{name}.json').write_text(json.dumps(values))
    return tmp, state


def _pretrain_argv(tmp, state, out, extra=()):
    return ['--execute', '--v3-state', str(state), '--k', str(K), '--num-classes', str(CLASSES),
            '--num-triples', str(TRIPLES), '--hidden', '8', '--token-dim', '4', '--graphs', str(tmp / 'g.pt'),
            '--graph-ids', str(tmp / 'ids.json'), '--dev-ids', str(tmp / 'dev.json'),
            '--validation-ids', str(tmp / 'val.json'), '--test-ids', str(tmp / 'test.json'),
            '--train-ids', str(tmp / 'train.json'), '--out', str(tmp / out), '--epochs', '2',
            '--batch-size', '3', *extra]


def test_pretrain_same_seed_gives_the_same_checkpoint_bytes(tmp_path):
    """--seed is recorded in the sidecar, so it must determine the checkpoint (dropout, init)."""
    from cei import pretrain
    tmp, state = _pretrain_setup(tmp_path)
    for name, extra in (('a.pt', ()), ('b.pt', ()), ('c.pt', ('--seed', '7'))):
        assert pretrain.main(_pretrain_argv(tmp, state, name, extra)) == 0
    first, second, other = (torch.load(tmp / name, weights_only=True) for name in ('a.pt', 'b.pt', 'c.pt'))
    # tensors, not file bytes: torch.save embeds the archive (file) name in the zip
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert any(not torch.equal(first[key], other[key]) for key in first)


def test_pretrain_refuses_graphs_whose_own_sample_id_is_a_test_id(tmp_path):
    """the guard must not trust --graph-ids alone; the graphs carry their sample_id."""
    from cei import pretrain
    tmp, state = _pretrain_setup(tmp_path, sample_ids=[f't{i}' for i in range(6)],
                                 test_ids=[f't{i}' for i in range(6)])   # --graph-ids claims g0..g5
    with pytest.raises(SystemExit):
        pretrain.main(_pretrain_argv(tmp, state, 'a.pt'))
    assert not (tmp / 'a.pt').exists()


def test_pretrain_accepts_graphs_whose_sample_id_matches_graph_ids(tmp_path):
    from cei import pretrain
    tmp, state = _pretrain_setup(tmp_path, sample_ids=[f'g{i}' for i in range(6)])
    assert pretrain.main(_pretrain_argv(tmp, state, 'a.pt')) == 0


@pytest.mark.parametrize('empty', ['dev', 'val', 'test', 'train'])
def test_pretrain_refuses_an_empty_id_list(tmp_path, empty):
    """the four lists are mandatory; an empty file would make its leakage check vacuous."""
    import json
    from cei import pretrain
    tmp, state = _pretrain_setup(tmp_path)
    (tmp / f'{empty}.json').write_text(json.dumps([]))
    with pytest.raises(SystemExit):
        pretrain.main(_pretrain_argv(tmp, state, 'a.pt'))
    assert not (tmp / 'a.pt').exists()


# ------------------------------------------------------------------ optimizer groups

@pytest.mark.parametrize('options', [dict(mp_kind='diag'), dict(mp_kind='hgt', mp_heads=2), dict(mp_kind='rgcn'),
                                     dict(mp_kind='simplehgn', mp_heads=2), dict()],
                         ids=lambda options: options.get('mp_kind', 'gine'))
def test_adamw_decays_weight_matrices_only_not_lookup_tables(tmp_path, options):
    """the comment of is_weight_matrix excludes embeddings and gate tables; g_fwd, g_rev, gk and gv
    are per-triple / per-type tables and must stay out of the decayed group like table_fwd and dir_emb."""
    adapter = _build(tmp_path, mp_optimizer='adamw', **options)
    groups = adapter.optimizer_groups(Namespace())
    decayed = {id(p) for group in groups if group['weight_decay'] > 0 for p in group['params']}
    listed = [id(p) for group in groups for p in group['params']]
    assert sorted(listed) == sorted(id(p) for p in adapter.parameters())   # every tensor in exactly one group
    tables = ('.g_fwd', '.g_rev', '.gk', '.gv')
    named = dict(adapter.named_parameters())
    present = [name for name in named if name.endswith(tables)]
    assert all(id(named[name]) not in decayed for name in present), present
    if options.get('mp_kind') == 'rgcn':   # the basis matrices V stay weight matrices
        assert any(id(p) in decayed for name, p in named.items() if name.endswith('.V'))
    # positive control, the decayed group is not empty for any kind (a leaf list that
    # lost 'weight' decays no Linear matrix of the message-passing stack and passes the negative checks above)
    stack_weights = [name for name, p in named.items()
                     if '.mp_stack.' in name and name.endswith('.weight') and p.ndim >= 2]
    assert stack_weights and any(id(named[name]) in decayed for name in stack_weights), stack_weights
