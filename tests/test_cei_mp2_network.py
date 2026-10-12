"""Characterization / regression tests for ``cei/network.py`` (``EvidenceNetworkMP2``).

Synthetic tensors only: no data file, run directory or checkpoint is opened. Fixtures come from
the v3 core and cei_mp tests.
"""
from __future__ import annotations

import warnings

import pytest
import torch
from torch_geometric.data import Batch

from core import NODE_KINDS
from core.tensorize import relation_vocabulary
from cei import options as O
from cei.network import EvidenceNetworkMP2
from tests import test_cei_gnn_v3_core as core_tests
from tests.test_cei_gnn_mp import _mp, _randomised
from tests.test_cei_gnn_v3_hooks import _StubBlock

warnings.filterwarnings('ignore')

REL = tuple(relation_vocabulary('forward'))[:core_tests.RELATIONS]
NEW = dict(mp_feed='dual', mp_hub_max=1, mp_dedup=1, mp_ctx='table', mp_root='bias',
           mp_optimizer='adamw', mp_lr_schedule='cosine')


def opts(base='DEFAULT', **kw):
    d = dict(O.DEFAULTS if base == 'DEFAULT' else O.LEGACY)
    d.update(kw)
    return O.validate(d)


def make(o, **kw):
    return _mp('C', cls=EvidenceNetworkMP2, mp2_options=dict(o), relation_names=REL, **kw)


def randomise_all(network, seed=5):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
            if 'norm' in name and name.endswith('weight') and 'gamma' not in name:
                parameter.abs_().add_(0.5)
    return network


def graph_batch(count=3):
    return Batch.from_data_list([core_tests._graph(7 + i) for i in range(count)])


def run(network, graph, **kw):
    return core_tests._run(network, graph, **kw)


# --------------------------------------------------------------------------- ledger-sum guard
def _ledger_total(network, batch, out):
    metadata = core_tests._metadata(batch)
    batch_index, src = metadata.batch_index, metadata.edge_index[0]
    graphs = int(metadata.graph_count)
    total = network.bias.expand(graphs, -1).clone()
    for key in network.ledger_keys():
        rows = out[key]
        if key.startswith('node_contributions') or key == 'nbr_contributions':
            owner = batch_index
        elif key.startswith('edge_contributions'):
            owner = batch_index[src]
        elif key.startswith('pair_contributions'):
            owner = batch_index[out['pairs'][0]] if out['pairs'].numel() else batch_index[:0]
        elif key == 'absence_contributions':
            owner = out['absence_items'][0]
        else:
            raise KeyError(key)
        total = total.index_add(0, owner, rows)
    return total


@pytest.mark.parametrize('mode', ['eval', 'train'])
@pytest.mark.parametrize('name,options', [('DEFAULT', opts()), ('LEGACY', opts('LEGACY')),
                                          ('NEW', opts(**NEW))])
def test_ledger_rows_sum_to_the_logits(name, options, mode):
    network = randomise_all(make(options))
    network.train(mode == 'train')
    batch = graph_batch(3)
    torch.manual_seed(0)
    out = run(network, batch)
    error = (_ledger_total(network, batch, out) - out['logits']).abs().max().item()
    assert error < 1e-5, f'{name}/{mode}: ledger rows miss the logits by {error:.2e}'


# ---------------------------------------------------------------------------
def test_missing_mp_dropout_is_refused_like_every_other_read_option():
    # (old mp2v2-net-06): REQUIRED_OPTIONS must list mp_dropout, which __init__ reads
    options = dict(opts())
    del options['mp_dropout']
    with pytest.raises(ValueError, match='missing'):
        make(options)


# ---------------------------------------------------------------------------
def test_shared_gate_makes_nbr_gates_class_independent():
    # (old mp2v2-net-04): shared_gate=1 gives every node-row block one scalar gate per
    # row, so the nbr block (mp_feed=dual) must not keep per-class gates
    network = randomise_all(make(opts(shared_gate=1, mp_feed='dual'))).eval()
    out = run(network, core_tests._graph(7))
    node_gates, nbr_gates = out['node_gates'], out['nbr_gates']
    assert torch.equal(node_gates[:, :1].expand_as(node_gates), node_gates)
    assert torch.allclose(nbr_gates[:, :1].expand_as(nbr_gates), nbr_gates), \
        'nbr gates differ across classes'
    denominator = out['node_denominator']
    assert torch.equal(denominator[:, :1].expand_as(denominator), denominator)
    assert out['nbr_denominator'].shape == denominator.shape
    nbr_den = out['nbr_denominator']
    assert torch.allclose(nbr_den[:, :1].expand_as(nbr_den), nbr_den)


def test_shared_gate_dual_ledger_still_sums_to_logits():
    batch = graph_batch(3)
    network = randomise_all(make(opts(shared_gate=1, mp_feed='dual'))).eval()
    out = run(network, batch)
    assert (_ledger_total(network, batch, out) - out['logits']).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
def _masked_token_visibility(rate=0.5, draws=60):
    """For every instance_of edge touching a drawn node: is the other endpoint's token visible?"""
    network = randomise_all(make(opts(mp_mask_rate=rate)))
    network.train(True)
    captured = {}
    network.node_encoder.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__('x', inputs[0].detach().clone()))
    metadata = core_tests._metadata(core_tests._graph(7))
    features = network.continuous_inputs(metadata)
    analyte = NODE_KINDS.index('analyte')
    src, dst = metadata.edge_index
    instance = (metadata.edge_relation == REL.index('instance_of')).nonzero().flatten().tolist()
    seen = {'drawn_analyte': [0, 0], 'drawn_other': [0, 0]}      # [edges, leaks]
    for _ in range(draws):
        _, _, masked = network._encode(features, metadata, metadata.edge_index)
        _, tokens, _ = torch.split(captured['x'], (network.node_dim, network.token_dim, network.hidden), -1)
        drawn = set(masked.tolist())
        for edge in instance:
            s, d = int(src[edge]), int(dst[edge])
            for node, other in ((s, d), (d, s)):
                if node in drawn:
                    bucket = seen['drawn_analyte' if int(metadata.node_type[node]) == analyte
                                  else 'drawn_other']
                    bucket[0] += 1
                    bucket[1] += int(bool(tokens[other].abs().sum() != 0))
    return seen


def test_masked_analyte_hides_its_instance_of_measurement_partners():
    # (old mp2rev-net-07): the partner of a drawn node is hidden whichever endpoint is drawn
    seen = _masked_token_visibility()
    assert seen['drawn_analyte'][0] > 0 and seen['drawn_other'][0] > 0
    assert seen['drawn_other'][1] == 0       # drawn measurement -> analyte partner hidden (holds)
    assert seen['drawn_analyte'][1] == 0, (
        f"{seen['drawn_analyte'][1]} of {seen['drawn_analyte'][0]} instance_of edges of a drawn analyte "
        'keep a visible measurement partner token')


# ---------------------------------------------------------------------------
BRANCHES = dict(mp_feed='dual', mp_branches=3, mp_branch_width=4)


def test_dual_feed_evaluates_the_h0_heads_once_for_k_branches():
    # (old mp2v2-net-03): under mp_feed=dual the node/edge/pair rows read h0 only, so
    # the K branch states must not re-evaluate them (K identical eval copies, K dropout draws in train)
    network = randomise_all(make(opts(**BRANCHES))).eval()
    calls = {name: [] for name in ('node_head', 'edge_source', 'edge_target', 'pair_projection')}
    for name, calls_of in calls.items():
        getattr(network, name).register_forward_hook(lambda module, inputs, output, c=calls_of: c.append(1))
    run(network, core_tests._graph(7))
    assert {name: len(c) for name, c in calls.items()} == {name: 1 for name in calls}


def test_dual_feed_nbr_row_is_the_mean_over_the_branches_and_nbr_head_runs_once_per_branch():
    # the h3-04 shortcut evaluates branch 0 inside _readout and branches 1.. through
    # _nbr_rows; a row from branch 0 only would still make the ledger sum (it sums whatever the rows are)
    network = randomise_all(make(opts(mp_feed='dual', mp_branches=2, mp_branch_width=4))).eval()
    states, nbr_inputs, seen = [], [], []
    original = network._message_passing
    network._message_passing = lambda *a, **k: (lambda result: (states.append(result[0]), result)[1])(original(*a, **k))
    network.nbr_head.register_forward_hook(lambda module, inputs, output: nbr_inputs.append(inputs[0].detach()))
    original_rows = network._nbr_rows
    network._nbr_rows = lambda rc, h: (seen.append(rc), original_rows(rc, h))[1]
    out = run(network, core_tests._graph(7))
    h_list = states[0]
    assert len(h_list) == 2 and len(nbr_inputs) == 2                     # nbr_head: once per branch
    assert all(torch.equal(a, b) for a, b in zip(nbr_inputs, h_list))     # ... on that branch's own state
    per_branch = [original_rows(seen[0], h) for h in h_list]
    assert not torch.allclose(per_branch[0]['nbr_contributions'], per_branch[1]['nbr_contributions']), (
        'the two branches give identical nbr rows: the test cannot tell a branch-0-only row')
    for key in ('nbr_contributions', 'nbr_gates', 'nbr_denominator'):
        expected = torch.stack([rows[key] for rows in per_branch]).mean(0)
        assert torch.allclose(out[key], expected, atol=1e-6), f'{key} is not the mean over the branches'


@pytest.mark.parametrize('mode', ['eval', 'train'])
def test_dual_feed_with_branches_ledger_sums_to_logits(mode):
    batch = graph_batch(3)
    network = randomise_all(make(opts(**BRANCHES)))
    network.train(mode == 'train')
    torch.manual_seed(0)
    out = run(network, batch)
    assert (_ledger_total(network, batch, out) - out['logits']).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
def test_extra_block_part_colliding_with_an_mp2_export_key_is_refused():
    # the collision check must see the keys the result really has ('mp_layers' is
    # added by the network itself, after the rows, so a block part of that name would be lost)
    stub = _StubBlock(core_tests.HIDDEN, core_tests.CLASSES, seed=1234, name='mp',
                      keys=('mp_contributions', 'mp_layers'))
    network = make(opts(), extra_blocks=(stub,)).eval()
    with pytest.raises(ValueError, match='collides'):
        run(network, core_tests._graph(7))


def test_extra_block_part_named_like_a_row_total_that_is_never_exported_is_accepted():
    # 'nbr_total' is an internal row key of the dual feed, not a return_parts key
    stub = _StubBlock(core_tests.HIDDEN, core_tests.CLASSES, seed=1234, name='nbr',
                      keys=('nbr_total', 'nbr_relations'))
    network = make(opts(mp_feed='dual'), extra_blocks=(stub,)).eval()
    out = run(network, core_tests._graph(7))
    assert 'nbr_total' in out and 'nbr_relations' in out


@pytest.mark.parametrize('mode', ['eval', 'train'])
@pytest.mark.parametrize('config', [
    dict(), dict(mp_feed='dual', mp_split_rows=1), dict(mp_dual_read=1, mp_jk=1),
    dict(mp_branches=2, mp_branch_width=4, mp_shrink=0.1, mp_mask_rate=0.3, mp_split_rows=1),
    dict(mp_residual='scalar', mp_feed='gates')])
def test_exported_key_set_covers_every_return_parts_key(config, mode):
    # guard: the collision check must never miss a key the network really exports
    network = randomise_all(make(opts(**config)))
    network.train(mode == 'train')
    out = run(network, graph_batch(2))
    assert set(out) <= network._exported_keys(), sorted(set(out) - network._exported_keys())


# ---------------------------------------------------------------------------
def _split_expectation(network, batch, out):
    """Independent message_share / message_share_kind from the exported split rows and the stub rows."""
    metadata = core_tests._metadata(batch)
    batch_index, src, node_type = metadata.batch_index, metadata.edge_index[0], metadata.node_type
    graphs, count = int(metadata.graph_count), network.num_node_types
    pairs = out['pairs']
    owners = {'node': batch_index, 'edge': batch_index[src], 'pair': batch_index[pairs[0]]}
    kinds = {'node': node_type, 'edge': node_type[src], 'pair': node_type[pairs[0]]}
    difference = out['logits'].new_zeros((graphs, network.num_classes))
    by_kind = out['logits'].new_zeros((graphs * count, network.num_classes))
    for kind in owners:
        rows = out[f'nbr_{kind}_contributions']
        difference = difference.index_add(0, owners[kind], rows)
        by_kind = by_kind.index_add(0, owners[kind] * count + kinds[kind], rows)
    # the stub block's total: its per-node rows summed per graph (not part of local / nbr / absence)
    block_total = out['logits'].new_zeros((graphs, network.num_classes)).index_add(
        0, batch_index, out['stub_contributions'])
    return difference, by_kind, block_total, count


def test_split_rows_message_share_excludes_the_extra_block_total_from_its_denominator():
    # bias + local + nbr + absence + extra blocks = logits; the share is taken against
    # the evidence the split covers (logits - bias - extra blocks), not against the block's total
    stub = _StubBlock(core_tests.HIDDEN, core_tests.CLASSES, seed=1234, name='stub')
    network = randomise_all(make(opts(mp_split_rows=1, mp_feed='dual'), extra_blocks=(stub,))).eval()
    batch = graph_batch(2)
    out = run(network, batch)
    difference, by_kind, block_total, count = _split_expectation(network, batch, out)
    covered = (out['logits'] - network.bias - block_total).abs().sum(-1)
    assert block_total.abs().sum() > 0
    expected = difference.abs().sum(-1) / covered
    assert torch.allclose(out['message_share'], expected, rtol=1e-4, atol=1e-6), \
        f"message_share {out['message_share'].tolist()} != {expected.tolist()}"
    expected_kind = (by_kind.abs().sum(-1) / covered.repeat_interleave(count)).view(-1, count)
    assert torch.allclose(out['message_share_kind'], expected_kind, rtol=1e-4, atol=1e-6)


def test_split_rows_with_an_extra_block_close_to_logits_once_the_block_rows_are_added():
    # guard (holds on the base): bias + local + nbr + absence + block rows = logits
    stub = _StubBlock(core_tests.HIDDEN, core_tests.CLASSES, seed=1234, name='stub')
    network = randomise_all(make(opts(mp_split_rows=1, mp_feed='dual'), extra_blocks=(stub,))).eval()
    batch = graph_batch(2)
    out = run(network, batch)
    metadata = core_tests._metadata(batch)
    batch_index, src = metadata.batch_index, metadata.edge_index[0]
    total = network.bias.expand(2, -1).clone()
    for kind, owner in (('node', batch_index), ('edge', batch_index[src]),
                        ('pair', batch_index[out['pairs'][0]])):
        for prefix in ('local', 'nbr'):
            total = total.index_add(0, owner, out[f'{prefix}_{kind}_contributions'])
    total = total.index_add(0, out['absence_items'][0], out['absence_contributions'])
    total = total.index_add(0, batch_index, out['stub_contributions'])
    assert torch.allclose(total, out['logits'], atol=1e-5)
