"""Explanation hooks of the GSAT, GraphCare and ProtGNN adapters.

`forward(..., external_edge_mask=None, token_vectors=None)` and `embed_tokens(batch)`: a mask
value of 0 is the same as deleting that edge, an all-ones mask is no mask, a fractional mask
reaches every message-passing layer of every pass exactly once, and `token_vectors` replaces the
token-embedding lookup of the forward. GraphCare's node attention of the last layer is rebuilt
from `last_visit_weight`.

The synthetic patient graph has a plain edge, a parallel copy, a mirror pair in another relation,
a 2-in-degree hub, a knowledge edge and three edges to or from the patient (GraphCare's global
node). Parameters are perturbed away from their initialisation so every path is live.
"""
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from core import NODE_KINDS
from core.registry import METHOD_REGISTRY
from core.tensorize import ALL_RELATIONS, PAYLOAD_WIDTH, ClinicalGraphData

METHODS = ('gsat', 'graphcare', 'protgnn')
NODE_DIM, NUM_TOKENS, NUM_TRIPLES, CLASSES = 6, 14, 9, 3
EDGE_DIM = len(ALL_RELATIONS) + PAYLOAD_WIDTH
HIDDEN, LAYERS, TOKEN_DIM = 16, 2, 4
KIND = {name: i for i, name in enumerate(NODE_KINDS)}
# node 0 patient (global), 1-2 visits, 3 complaint, 4 measurement, 5 analyte, 6 vital,
# 7 knowledge, 8 diagnosis, 9 measurement; rows (source, target, relation, triple)
EDGES = [
    (0, 1, 0, 1),   # 0  patient -> visit 1          (global-node edge)
    (1, 0, 1, 2),   # 1  visit 1 -> patient          (global-node edge, other relation)
    (1, 3, 2, 3),   # 2  visit 1 -> complaint        (plain)
    (1, 4, 3, 4),   # 3  visit 1 -> measurement      (twin of the parallel copy)
    (4, 5, 4, 5),   # 4  measurement -> analyte      (mirrored by edge 10)
    (1, 6, 5, 6),   # 5  visit 1 -> vital
    (5, 7, 6, 7),   # 6  analyte -> knowledge
    (2, 8, 7, 8),   # 7  visit 2 -> diagnosis
    (8, 1, 8, 1),   # 8  diagnosis -> visit 1
    (1, 4, 3, 4),   # 9  parallel copy of edge 3
    (5, 4, 9, 2),   # 10 mirror of edge 4 in another relation
    (2, 9, 3, 4),   # 11 visit 2 -> measurement 9
    (9, 5, 4, 5),   # 12 measurement 9 -> analyte  (the analyte becomes a hub)
    (0, 2, 0, 1),   # 13 patient -> visit 2          (global-node edge)
]
KINDS = ['patient', 'visit', 'visit', 'complaint', 'measurement', 'analyte', 'vital',
         'knowledge', 'diagnosis', 'measurement']
MEMBERSHIP = [(0, n) for n in (1, 3, 4, 5, 6, 7)] + [(1, n) for n in (2, 5, 7, 8, 9)]
DELETIONS = {
    'plain': (2,), 'parallel copy': (9,), 'twin of the copy': (3,), 'copy and twin': (3, 9),
    'mirror': (10,), 'mirrored': (4,), 'mirror pair': (4, 10), 'hub in-edge': (12,),
    'knowledge edge': (6,), 'global-node edge': (0,), 'every global-node edge': (0, 1, 13),
    'every edge': tuple(range(len(EDGES))),
}


def _graph(removed=()):
    generator = torch.Generator().manual_seed(100)
    keep = [i for i in range(len(EDGES)) if i not in set(removed)]
    edges = torch.tensor(EDGES, dtype=torch.long).t()
    x = torch.randn((len(KINDS), NODE_DIM), generator=generator)
    edge_attr = torch.randn((len(EDGES), EDGE_DIM), generator=generator)
    data = ClinicalGraphData(
        x=x,
        edge_index=edges[:2, keep].contiguous(),
        edge_attr=edge_attr[keep],
        visit_membership_index=torch.tensor(MEMBERSHIP, dtype=torch.long).t().contiguous(),
        num_visits=torch.tensor([2], dtype=torch.long),
        global_node_mask=torch.tensor([kind == 'patient' for kind in KINDS]),
    )
    data.token = torch.arange(len(KINDS)) % (NUM_TOKENS - 1) + 1
    data.node_type = torch.tensor([KIND[kind] for kind in KINDS], dtype=torch.long)
    data.edge_relation = edges[2, keep].contiguous()
    data.edge_triple = edges[3, keep].contiguous()
    return Batch.from_data_list([data])


def _model(method):
    torch.manual_seed(1234)
    model = METHOD_REGISTRY[method](
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
        hidden=HIDDEN, layers=LAYERS, dropout=0.0, token_dim=TOKEN_DIM,
        num_triples=NUM_TRIPLES, args=Namespace())
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    return model.eval()


def _logits(model, batch, **hooks):
    with torch.no_grad():
        return model(batch, epoch=0, **hooks).logits


def _layers(model, method):
    return list(model.predictor.layers if method == 'gsat' else model.layers)


# ------------------------------------------------------------------ edge mask
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('case', list(DELETIONS))
def test_a_zero_mask_value_is_deleting_that_edge(method, case):
    model = _model(method)
    removed = DELETIONS[case]
    mask = torch.ones(len(EDGES))
    mask[list(removed)] = 0.0
    masked = _logits(model, _graph(), external_edge_mask=mask)
    deleted = _logits(model, _graph(removed))
    assert torch.allclose(masked, deleted, rtol=0.0, atol=1e-6), (masked - deleted).abs().max()


@pytest.mark.parametrize('method', METHODS)
def test_deleting_an_edge_changes_the_prediction(method):
    """The deletion checks above are not vacuous: a plain edge has an effect in every model."""
    model = _model(method)
    assert (_logits(model, _graph()) - _logits(model, _graph((2,)))).abs().max() > 1e-4


@pytest.mark.parametrize('method', METHODS)
def test_an_all_ones_mask_is_exactly_no_mask(method):
    model = _model(method)
    batch = _graph()
    assert torch.equal(_logits(model, batch, external_edge_mask=torch.ones(len(EDGES))),
                       _logits(model, batch))
    assert torch.equal(_logits(model, batch, external_edge_mask=None, token_vectors=None),
                       _logits(model, batch))


@pytest.mark.parametrize('method', METHODS)
def test_a_mask_of_the_wrong_length_is_refused(method):
    with pytest.raises(ValueError):
        _logits(_model(method), _graph(), external_edge_mask=torch.ones(len(EDGES) - 1))


@pytest.mark.parametrize('method', METHODS)
def test_a_fractional_mask_reaches_every_layer_of_every_pass_once(method):
    model = _model(method)
    batch = _graph()
    mask = torch.linspace(0.1, 0.9, len(EDGES))
    seen = []
    handles = [layer.register_forward_pre_hook(
        lambda module, args, kwargs, index=index: seen.append(
            (index, kwargs.get('edge_attention' if method == 'gsat' else 'edge_mask'))),
        with_kwargs=True) for index, layer in enumerate(_layers(model, method))]
    try:
        _logits(model, batch, external_edge_mask=mask)
    finally:
        for handle in handles:
            handle.remove()
    if method == 'gsat':
        # extraction pass (sets the node attention), then prediction pass (lifted attention)
        expected = [mask] * LAYERS + [model.last_edge_attention * mask] * LAYERS
    elif method == 'graphcare':
        # GraphCare passes messages only on edges whose endpoints are both non-global
        source, target = batch.edge_index
        non_global = ~batch.global_node_mask
        expected = [mask[non_global[source] & non_global[target]]] * LAYERS
    else:
        expected = [mask] * LAYERS
    order = list(range(LAYERS)) * (2 if method == 'gsat' else 1)
    assert [index for index, _ in seen] == order
    for (_, delivered), value in zip(seen, expected):
        assert delivered is not None and torch.equal(delivered, value)


def test_gsat_mask_moves_the_attention_as_deletion_does():
    """The extraction pass sees the mask: GSAT's node attention under mask 0 is the attention
    of the graph without that edge."""
    model = _model('gsat')
    mask = torch.ones(len(EDGES))
    mask[2] = 0.0
    _logits(model, _graph(), external_edge_mask=mask)
    masked = model.last_node_attention.clone()
    _logits(model, _graph((2,)))
    assert torch.allclose(masked, model.last_node_attention, rtol=0.0, atol=1e-6)
    _logits(model, _graph())
    assert not torch.allclose(masked, model.last_node_attention, rtol=0.0, atol=1e-6)


# ------------------------------------------------------------------ token vectors
@pytest.mark.parametrize('method', METHODS)
def test_token_vectors_from_embed_tokens_are_exactly_no_override(method):
    model = _model(method)
    batch = _graph()
    vectors = model.embed_tokens(batch)
    assert vectors.shape == (len(KINDS), TOKEN_DIM)
    assert torch.equal(_logits(model, batch, token_vectors=vectors), _logits(model, batch))


@pytest.mark.parametrize('method', METHODS)
def test_a_changed_token_vector_changes_the_logits(method):
    model = _model(method)
    batch = _graph()
    vectors = model.embed_tokens(batch).detach().clone()
    vectors[3] += 1.0
    changed = _logits(model, batch, token_vectors=vectors)
    assert (changed - _logits(model, batch)).abs().max() > 1e-4


@pytest.mark.parametrize('method', METHODS)
def test_token_vectors_of_the_wrong_shape_are_refused(method):
    model = _model(method)
    batch = _graph()
    with pytest.raises(ValueError):
        _logits(model, batch, token_vectors=torch.zeros(len(KINDS), TOKEN_DIM + 1))
    with pytest.raises(ValueError):
        _logits(model, batch, token_vectors=torch.zeros(len(KINDS) - 1, TOKEN_DIM))


def _embedding_module(model, method):
    return model.predictor.token_embedding if method == 'gsat' else model.token_embedding


@pytest.mark.parametrize('method', METHODS)
def test_token_vectors_equal_editing_the_embedding_rows(method):
    """`token_vectors` replace the lookup in every pass: the forward with v equals the forward of
    a copy whose embedding rows of this graph's tokens are set to v (GSAT: logits and the node
    attention of the extraction pass), and the gradient with respect to v is the gradient with
    respect to those rows."""
    import copy

    model = _model(method)
    batch = _graph()
    assert len(set(batch.token.tolist())) == batch.token.numel()     # one token per node
    generator = torch.Generator().manual_seed(11)
    vectors = (model.embed_tokens(batch).detach()
               + 0.3 * torch.randn((batch.token.numel(), TOKEN_DIM), generator=generator))
    edited = copy.deepcopy(model)
    with torch.no_grad():
        _embedding_module(edited, method).weight[batch.token] = vectors
    assert torch.equal(_logits(model, batch, token_vectors=vectors), _logits(edited, batch))
    if method == 'gsat':
        _logits(model, batch, token_vectors=vectors)
        hooked = model.last_node_attention.clone()
        _logits(edited, batch)
        assert torch.equal(hooked, edited.last_node_attention)
        _logits(model, batch)
        assert not torch.equal(hooked, model.last_node_attention)    # v moved the attention

    leaf = vectors.clone().requires_grad_(True)
    model(batch, epoch=0, token_vectors=leaf).logits[0, 0].backward()
    weight = _embedding_module(edited, method).weight
    edited.zero_grad(set_to_none=True)
    edited(batch, epoch=0).logits[0, 0].backward()
    assert torch.equal(leaf.grad, weight.grad[batch.token])


@pytest.mark.parametrize('method', METHODS)
def test_token_vectors_carry_gradients_to_the_logits(method):
    model = _model(method)
    batch = _graph()
    vectors = model.embed_tokens(batch).detach().clone().requires_grad_(True)
    model(batch, epoch=0, token_vectors=vectors).logits[0, 0].backward()
    assert vectors.grad is not None and vectors.grad.abs().sum() > 0


# ------------------------------------------------------------------ GraphCare attention
def test_graphcare_node_attention_is_rebuilt_from_last_visit_weight():
    model = _model('graphcare')
    batch = _graph()
    seen = []
    handle = model.layers[-1].register_forward_pre_hook(
        lambda module, args: seen.append(args[3].detach().clone()))
    try:
        _logits(model, batch)
    finally:
        handle.remove()
    assert len(seen) == 1
    rebuilt = scatter(model.last_visit_weight, batch.visit_membership_index[1], dim=0,
                      dim_size=batch.num_nodes, reduce='sum')
    assert torch.equal(rebuilt, seen[0])
    assert rebuilt[0] == 0.0                     # the global node carries no visit membership
    assert not hasattr(model, 'last_node_attention')
