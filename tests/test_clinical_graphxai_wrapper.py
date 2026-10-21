"""Synthetic, in-memory tests of the baseline GraphXAI wrappers (gsat, protgnn, graphcare).

Reuses graphxai_fixtures.py's synthetic-graph helper and dimension conventions (same
NODE_DIM/EDGE_DIM/etc., same build_method() path) so there is one definition of what a valid
synthetic clinical graph looks like. Covers the wrapper protocol (x = numeric | token vector,
verify against the raw batch, ig_feature_spec, ig_path, inert edges, deletion_forward) and the
models' own built-in explanations.
"""
import dataclasses
import inspect

import pytest
import torch
from torch_geometric.data import Batch

from core.explain.fidelity_v2 import removable_node_mask
from core.explain.graphxai_wrapper import (
    ClinicalGraphXAIWrapper,
    GraphCareGraphXAIWrapper,
)
from core.registry import build_method
from core.tensorize import ALL_RELATIONS
from tests.graphxai_fixtures import (
    EDGE_DIM,
    NODE_DIM,
    NUM_CLASSES,
    NUM_TOKENS,
    NUM_TRIPLES,
    method_args,
    synthetic_graph,
)

GENERIC_METHODS = (
    "gsat", "protgnn",
)
BASELINES = ("gsat", "protgnn", "graphcare")
NUM_RELATIONS = len(ALL_RELATIONS)
TOKEN_DIM = 4
# ProtGNN's search settings of every final run (the adapter's defaults).
FINAL_MCTS = dict(rollout=3, min_atoms=3, max_atoms=6, expand_atoms=8, c_puct=5.0)


def one_graph():
    return synthetic_graph(
        sample_id="single",
        label=0,
        node_count=5,
        num_visits=3,
        membership=[[0, 0, 1, 2], [0, 1, 2, 3]],
        global_mask=[False, False, False, False, True],
    )


def build_adapter(name, **overrides):
    return build_method(
        name,
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_classes=NUM_CLASSES, hidden=8, layers=2, dropout=0.0,
        token_dim=TOKEN_DIM, num_triples=NUM_TRIPLES, args=method_args(**overrides),
    )


def wrap_generic(name, graph, model=None):
    model = model if model is not None else build_adapter(name)
    model.eval()
    wrapper = ClinicalGraphXAIWrapper(
        model, method=name, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_tokens=NUM_TOKENS, num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


def wrap_graphcare(graph, model=None):
    model = model if model is not None else build_adapter("graphcare")
    model.eval()
    wrapper = GraphCareGraphXAIWrapper(
        model, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


def wrap_any(name, graph, model=None):
    return wrap_graphcare(graph, model) if name == "graphcare" else wrap_generic(name, graph, model)


def _raw_logits(model, graph):
    model.eval()
    with torch.no_grad():
        return model(graph, epoch=0).logits


# --- protocol --------------------------------------------------------------------------------

@pytest.mark.parametrize("name", BASELINES)
def test_verify_passes_for_every_baseline(name):
    torch.manual_seed(hash(name) % (2**31))
    graph = one_graph()
    wrapper, x = wrap_any(name, graph)
    ok, max_diff = wrapper.verify(x, graph.edge_index)
    assert ok, f"{name}: verify() failed, max_diff={max_diff}"
    assert max_diff < 1e-5


@pytest.mark.parametrize("name", BASELINES)
def test_a_corrupted_context_makes_verify_fail(name):
    torch.manual_seed(3)
    graph = one_graph()
    wrapper, x = wrap_any(name, graph)
    ctx = wrapper._context
    wrapper._context = dataclasses.replace(
        ctx, node_type=torch.zeros_like(ctx.node_type),
        edge_relation=torch.zeros_like(ctx.edge_relation),
        edge_triple=torch.zeros_like(ctx.edge_triple))
    ok, max_diff = wrapper.verify(x, graph.edge_index)
    assert not ok and max_diff > 1e-5, f"{name}: verify passed on a corrupted context"


@pytest.mark.parametrize("name", BASELINES)
def test_a_changed_token_block_makes_verify_fail(name):
    torch.manual_seed(5)
    graph = one_graph()
    wrapper, x = wrap_any(name, graph)
    x = x.clone()
    x[:, NODE_DIM:] += 0.5
    ok, max_diff = wrapper.verify(x, graph.edge_index)
    assert not ok and max_diff > 1e-5


@pytest.mark.parametrize("name", BASELINES)
def test_x_is_numeric_then_the_adapters_token_vectors_and_round_trips(name):
    torch.manual_seed(7)
    graph = one_graph()
    wrapper, x = wrap_any(name, graph)
    model = wrapper.adapter
    assert x.dtype == torch.float32
    assert x.shape == (graph.x.size(0), NODE_DIM + TOKEN_DIM)
    assert torch.equal(x[:, :NODE_DIM], graph.x.float())
    with torch.no_grad():
        assert torch.equal(x[:, NODE_DIM:], model.embed_tokens(graph))
        assert torch.equal(wrapper(x, graph.edge_index), _raw_logits(model, graph))
    assert wrapper.ig_feature_spec() == {"token_columns": [NODE_DIM, NODE_DIM + TOKEN_DIM]}
    provenance = wrapper.ig_feature_provenance()
    assert provenance["explained_columns"] == {"numeric": [0, NODE_DIM],
                                              "token_vector": [NODE_DIM, NODE_DIM + TOKEN_DIM]}
    assert provenance["token_columns"] == [NODE_DIM, NODE_DIM + TOKEN_DIM]


@pytest.mark.parametrize("name", BASELINES)
def test_the_token_block_reaches_the_model_as_token_vectors(name):
    torch.manual_seed(9)
    graph = one_graph()
    wrapper, x = wrap_any(name, graph)
    model = wrapper.adapter
    changed = x.clone()
    changed[2, NODE_DIM:] = torch.randn(TOKEN_DIM)
    with torch.no_grad():
        expected = model(graph, epoch=0, token_vectors=changed[:, NODE_DIM:]).logits
        assert torch.equal(wrapper(changed, graph.edge_index), expected)
        assert not torch.allclose(expected, _raw_logits(model, graph), atol=1e-6)
    point = x.clone().requires_grad_(True)
    logits = wrapper(point, graph.edge_index)
    (grad,) = torch.autograd.grad(logits[0, 0], point)
    assert grad[:, NODE_DIM:].abs().sum() > 0, f"{name}: no gradient on the token block"


@pytest.mark.parametrize("name", BASELINES)
def test_protocol_attributes(name):
    graph = one_graph()
    wrapper, x = wrap_any(name, graph)
    assert wrapper.method == name
    assert wrapper.builtin_class_specific is (name == "protgnn")
    assert "token_override" not in inspect.signature(wrapper.forward).parameters
    with torch.no_grad():
        outside = wrapper(x, graph.edge_index)
        with wrapper.ig_path() as inside:
            assert inside is wrapper
            assert torch.equal(wrapper(x, graph.edge_index), outside)
    inert = wrapper.inert_edge_mask()
    if name == "graphcare":
        global_mask = graph.global_node_mask
        source, target = graph.edge_index
        assert inert.dtype == torch.bool
        assert torch.equal(inert, global_mask[source] | global_mask[target])
        assert inert.any() and not inert.all()
    else:
        assert inert is None


@pytest.mark.parametrize("name", BASELINES)
def test_deletion_forward_is_the_adapters_own_forward_on_one_graph(name):
    torch.manual_seed(11)
    graph = one_graph()
    wrapper, x = wrap_any(name, graph)
    model = wrapper.adapter
    model.train()
    got = wrapper.deletion_forward(graph)
    assert model.training, "deletion_forward must restore the training flag"
    model.eval()
    with torch.no_grad():
        assert torch.equal(got, model(Batch.from_data_list([graph]), epoch=0).logits)
        assert torch.allclose(got, wrapper(x, graph.edge_index), atol=1e-6)
    assert got.shape == (1, NUM_CLASSES)


@pytest.mark.parametrize("name", BASELINES)
def test_edge_mask_hook_changes_prediction_and_is_differentiable(name):
    torch.manual_seed(31)
    graph = one_graph()
    wrapper, x = wrap_any(name, graph)

    baseline = wrapper(x, graph.edge_index)
    edge_count = graph.edge_index.size(1)
    ones_mask = torch.ones(edge_count, requires_grad=True)
    masked = wrapper(x, graph.edge_index, edge_mask=ones_mask)
    assert torch.allclose(masked, baseline, atol=1e-5), f"{name}: all-ones mask changed output"

    target_class = int(baseline.argmax(-1).item())
    grad = torch.autograd.grad(masked[0, target_class], ones_mask, retain_graph=True)[0]
    assert grad is not None, f"{name}: edge_mask gradient disconnected"
    assert torch.isfinite(grad).all()

    zeroed = wrapper(x, graph.edge_index, edge_mask=torch.zeros(edge_count))
    assert not torch.allclose(zeroed, baseline, atol=1e-4), f"{name}: zero mask had no effect"


def test_edge_index_mismatch_is_refused():
    """Generic _batch_with() validation, exercised once via gsat."""
    torch.manual_seed(61)
    graph = one_graph()
    wrapper, x = wrap_generic("gsat", graph)
    wrong_edges = graph.edge_index.flip(0)
    with pytest.raises(ValueError, match="edge_index must match"):
        wrapper(x, wrong_edges)
    with pytest.raises(ValueError, match="x must keep the shape"):
        wrapper(x[:, :NODE_DIM], graph.edge_index)


def test_external_edge_mask_rejects_wrong_shape():
    """Adapter-level validation (every adapter checks edge_mask.numel() against its own edge
    count), exercised once via gsat."""
    torch.manual_seed(63)
    graph = one_graph()
    wrapper, x = wrap_generic("gsat", graph)
    with pytest.raises(ValueError, match="one value per edge"):
        wrapper(x, graph.edge_index, edge_mask=torch.ones(graph.edge_index.size(1) + 1))


def test_a_graph_batch_is_refused():
    graph = one_graph()
    wrapper = ClinicalGraphXAIWrapper(
        build_adapter("gsat"), method="gsat", node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_tokens=NUM_TOKENS, num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS)
    with pytest.raises(ValueError, match="one graph at a time"):
        wrapper.set_context(Batch.from_data_list([graph, one_graph()]))


# --- GSAT built-in ---------------------------------------------------------------------------

def test_gsat_builtin_is_its_own_attention_with_the_edge_attention_in_the_detail():
    torch.manual_seed(37)
    graph = one_graph()
    wrapper, x = wrap_generic("gsat", graph)
    model = wrapper.adapter
    reference = model.explain(graph).clone()
    # A masked forward just before must not leak into the built-in.
    wrapper(x, graph.edge_index, edge_mask=torch.zeros(graph.edge_index.size(1)))
    importance = wrapper.builtin_node_importance(x, graph.edge_index)
    assert isinstance(importance, list) and len(importance) == graph.x.size(0)
    assert torch.equal(torch.tensor(importance), reference)
    assert wrapper.builtin_node_importance(x, graph.edge_index, target_class=1) == importance

    detail = wrapper.builtin_detail()
    source, target = graph.edge_index
    assert detail["kind"] == "gsat_node_attention" and detail["class_specific"] is False
    assert torch.equal(torch.tensor(detail["edge_attention"]), reference[source] * reference[target])
    eligible = torch.tensor(importance)[removable_node_mask(graph.node_type)]
    assert detail["distinct_values_eligible"] == len(set(eligible.tolist()))
    assert detail["constant_over_eligible"] is (len(set(eligible.tolist())) == 1)
    assert detail["saturated_nodes"] == sum(value == 1.0 for value in importance)


def test_wrappers_without_a_builtin_still_refuse_clearly():
    graph = one_graph()
    wrapper = ClinicalGraphXAIWrapper(
        build_adapter("gsat"), method="not_a_builtin_method", node_dim=NODE_DIM,
        edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS, num_triples=NUM_TRIPLES,
        num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    with pytest.raises(NotImplementedError, match="no per-node built-in explanation"):
        wrapper.builtin_node_importance(x, graph.edge_index)
    assert wrapper.builtin_detail() is None
    assert wrapper.builtin_class_specific is False


# --- ProtGNN built-in ------------------------------------------------------------------------

def _connected(nodes, edge_index):
    nodes = set(nodes)
    adjacency = {n: set() for n in nodes}
    for a, b in edge_index.t().tolist():
        if a in nodes and b in nodes:
            adjacency[a].add(b)
            adjacency[b].add(a)
    seen, stack = set(), [next(iter(nodes))]
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(adjacency[n] - seen)
    return seen == nodes


def larger_graph(node_count=9):
    return synthetic_graph(
        sample_id="larger", label=0, node_count=node_count, num_visits=1,
        membership=[[0] * node_count, list(range(node_count))],
        global_mask=[False] * node_count,
    )


def wrap_protgnn(graph, seed=43, **settings):
    torch.manual_seed(seed)
    return wrap_generic("protgnn", graph, model=build_adapter("protgnn", **(settings or FINAL_MCTS)))


def test_protgnn_builtin_is_the_adapters_own_explain_detail():
    graph = larger_graph()
    wrapper, x = wrap_protgnn(graph)
    adapter = wrapper.adapter
    importance = wrapper.builtin_node_importance(x, graph.edge_index)
    detail = wrapper.builtin_detail()
    own = adapter.explain_detail(graph)

    assert set(importance) <= {0.0, 1.0} and len(importance) == graph.x.size(0)
    members = [i for i, value in enumerate(importance) if value == 1.0]
    assert members == own["subgraph_nodes"] == detail["subgraph_nodes"]
    assert members and _connected(members, graph.edge_index)
    for key, value in own.items():
        assert detail[key] == value, key
    assert 3 <= len(members) <= 6

    with torch.no_grad():
        assert detail["predicted_class"] == int(wrapper(x, graph.edge_index).argmax(-1))
    classes = [i for i in range(adapter.num_prototypes)
               if int(adapter.prototype_class_ids[i]) == detail["explained_class"]]
    assert detail["prototype_index"] == max(classes,
                                            key=lambda i: detail["prototype_contributions"][i])


def test_protgnn_search_provenance_is_the_models_own_settings_and_search():
    graph = larger_graph()
    wrapper, _ = wrap_protgnn(graph)
    adapter = wrapper.adapter
    detail = wrapper.builtin_detail()
    search = detail["search"]
    assert search["algorithm"].startswith("adapter MCTS")
    assert search["settings"] == {
        "rollout": adapter.rollout, "min_atoms": adapter.min_atoms,
        "max_atoms": adapter.max_atoms, "expand_atoms": adapter.expand_atoms,
        "c_puct": adapter.c_puct}
    assert search["settings"] == FINAL_MCTS
    assert search["settings_source"] == "adaptation defaults; official ProtGNN 10/5/10/10"
    assert "re-encoded" in search["subgraph_scoring"]

    # The root and the number of scored states are those of the adapter's own search.
    calls = []
    induced = adapter._induced_graph_embedding

    def counting(*args, **kwargs):
        calls.append(1)
        return induced(*args, **kwargs)

    adapter._induced_graph_embedding = counting
    try:
        adapter.explain_detail(graph)
    finally:
        del adapter._induced_graph_embedding
    assert search["states_scored"] == len(calls) >= 1
    with torch.no_grad():
        node_state, _, graph_index = adapter._node_and_graph_embeddings(graph)
        candidate = adapter._projection_candidate(graph, 0, node_state, graph_index)
        root, _ = adapter._initial_coalition(
            candidate["node_state"], candidate["edge_index"],
            adapter.prototype_vectors[detail["prototype_index"]])
    assert search["root_coalition"] == list(root)
    assert "_initial_coalition" not in vars(adapter)
    assert "_induced_graph_embedding" not in vars(adapter)

    eligible = removable_node_mask(graph.node_type).tolist()
    assert detail["subgraph_nodes_eligible"] == [n for n in detail["subgraph_nodes"] if eligible[n]]
    assert detail["subgraph_eligible_count"] == len(detail["subgraph_nodes_eligible"])


def test_protgnn_settings_other_than_the_defaults_are_named_as_recorded():
    graph = larger_graph()
    wrapper, _ = wrap_protgnn(graph, rollout=2, min_atoms=3, max_atoms=5, expand_atoms=4,
                              c_puct=1.0)
    search = wrapper.builtin_detail()["search"]
    assert search["settings"] == {"rollout": 2, "min_atoms": 3, "max_atoms": 5,
                                  "expand_atoms": 4, "c_puct": 1.0}
    assert search["settings_source"].startswith("the run's recorded settings")


def test_protgnn_builtin_searches_once_per_visit_and_class(monkeypatch):
    graph = larger_graph()
    wrapper, x = wrap_protgnn(graph, seed=49)
    adapter = wrapper.adapter
    calls = []
    real = adapter.explain_detail

    def counting(batch, target_class=None):
        calls.append(target_class)
        return real(batch, target_class)

    monkeypatch.setattr(adapter, "explain_detail", counting)
    predicted = int(wrapper(x, graph.edge_index).argmax(-1))
    other = 1 - predicted
    wrapper.builtin_node_importance(x, graph.edge_index)
    wrapper.builtin_detail()
    wrapper.builtin_node_importance(x, graph.edge_index, target_class=predicted)
    wrapper.builtin_detail(target_class=predicted)
    assert calls == [None]
    wrapper.builtin_node_importance(x, graph.edge_index, target_class=other)
    wrapper.builtin_detail(target_class=other)
    assert calls == [None, other]
    assert wrapper.builtin_detail(target_class=other)["explained_class"] == other
    wrapper.set_context(graph)   # a new context: a fresh search
    wrapper.builtin_detail()
    assert calls == [None, other, None]


def test_protgnn_builtin_is_deterministic_and_does_not_disturb_the_model():
    graph = larger_graph()
    wrapper, x = wrap_protgnn(graph, seed=47)
    wrapper.adapter.train()
    before = wrapper(x, graph.edge_index).detach().clone()
    first = wrapper.builtin_detail()
    wrapper.set_context(graph)
    second = wrapper.builtin_detail()
    assert first == second
    assert torch.equal(wrapper(x, graph.edge_index), before)
    assert wrapper.adapter.training


def test_protgnn_two_node_graph_raises_and_three_node_graph_is_explained():
    small = synthetic_graph(sample_id="two", label=0, node_count=2, num_visits=1,
                            membership=[[0, 0], [0, 1]], global_mask=[False, False])
    wrapper, x = wrap_protgnn(small)
    with pytest.raises(RuntimeError, match="no connected subgraph of at least 3 nodes"):
        wrapper.builtin_node_importance(x, small.edge_index)
    assert wrapper.builtin_detail()["subgraph_nodes"] == []

    three = synthetic_graph(sample_id="three", label=0, node_count=3, num_visits=1,
                            membership=[[0, 0, 0], [0, 1, 2]], global_mask=[False] * 3)
    wrapper, x = wrap_protgnn(three)
    assert wrapper.builtin_node_importance(x, three.edge_index) == [1.0, 1.0, 1.0]


# --- GraphCare built-in ----------------------------------------------------------------------

def graphcare_graph(membership, global_mask, num_visits):
    node_count = len(global_mask)
    return synthetic_graph(sample_id="gc", label=0, node_count=node_count,
                           num_visits=num_visits, membership=membership,
                           global_mask=global_mask)


def _fresh_attention(model, graph):
    model.eval()
    with torch.no_grad():
        model(graph, epoch=0)
    attention = torch.zeros(graph.x.size(0))
    attention.index_add_(0, graph.visit_membership_index[1], model.last_visit_weight)
    return attention


def test_graphcare_builtin_is_a_fresh_last_layer_attention_after_a_masked_forward():
    torch.manual_seed(47)
    graph = one_graph()
    wrapper, x = wrap_graphcare(graph)
    model = wrapper.adapter
    reference = _fresh_attention(model, graph)
    wrapper(x, graph.edge_index, edge_mask=torch.zeros(graph.edge_index.size(1)))
    masked_state = model.last_visit_weight.clone()
    importance = wrapper.builtin_node_importance(x, graph.edge_index)
    # every node of this graph has its own single-visit set, so the per-set mean is the value
    assert torch.equal(torch.tensor(importance, dtype=torch.float64), reference.double())
    assert not torch.equal(masked_state, model.last_visit_weight), \
        "the masked forward did not change the attention; the check would be vacuous"
    assert wrapper.builtin_node_importance(x, graph.edge_index, target_class=1) == importance
    assert importance[4] == 0.0   # the global node


def test_graphcare_single_visit_graph_is_constant_one():
    torch.manual_seed(13)
    node_count = 8
    graph = graphcare_graph([[0] * (node_count - 1), list(range(1, node_count))],
                            [True] + [False] * (node_count - 1), num_visits=1)
    wrapper, x = wrap_graphcare(graph)
    importance = wrapper.builtin_node_importance(x, graph.edge_index)
    assert importance[0] == 0.0
    assert importance[1:] == [1.0] * (node_count - 1)
    detail = wrapper.builtin_detail()
    assert detail["num_visits"] == 1
    assert detail["beta"] == [1.0]
    assert detail["constant_over_eligible"] is True
    assert detail["distinct_values_eligible"] == 1
    assert detail["largest_tie_group_eligible"] == int(removable_node_mask(graph.node_type).sum())


def test_graphcare_nodes_with_equal_visit_sets_tie_exactly():
    import types

    import numpy as np

    from core.explain.fidelity_v2 import fidelity_plus_v2

    # node 0 global; eligible nodes (types 2, 3, 5, 7) 2, 3, 5, 7 -> {0, 1}; 10, 11 -> {0};
    # 13, 15 -> {1}; the structural nodes are spread over the same three visit sets.
    visit_sets = {2: (0, 1), 3: (0, 1), 5: (0, 1), 7: (0, 1), 10: (0,), 11: (0,), 13: (1,),
                  15: (1,), 1: (0, 1), 4: (0,), 6: (1,), 8: (0, 1), 9: (0,), 12: (1,), 14: (0, 1)}
    pairs = sorted((visit, node) for node, visits in visit_sets.items() for visit in visits)
    membership = [[v for v, _ in pairs], [n for _, n in pairs]]
    # seed 9: the float32 values of the shared two-visit set differ (checked below)
    torch.manual_seed(9)
    graph = graphcare_graph(membership, [True] + [False] * 15, num_visits=2)
    wrapper, x = wrap_graphcare(graph)
    raw = _fresh_attention(wrapper.adapter, graph)
    shared = [2, 3, 5, 7]
    raw_spread = float(raw[shared].max() - raw[shared].min())
    assert raw_spread > 0, "the float32 values tie already; the check would be vacuous"

    value = wrapper.builtin_node_importance(x, graph.edge_index)
    groups = {}
    for node, visits in visit_sets.items():
        groups.setdefault(visits, []).append(node)
    for members in groups.values():
        assert len({value[node] for node in members}) == 1
        mean = float(raw[sorted(members)].double().mean())
        assert value[members[0]] == mean
    assert len({value[members[0]] for members in groups.values()}) == 3
    assert value[0] == 0.0   # the global node

    detail = wrapper.builtin_detail()
    assert detail["node_attention"] == value
    assert detail["num_visits"] == 2 and len(detail["beta"]) == 2
    assert abs(sum(detail["beta"]) - 1.0) < 1e-6
    assert detail["kind"] == "graphcare_visit_attention" and detail["layer"] == "last"
    assert detail["class_specific"] is False
    assert "visit set" in detail["value_depends_on"]
    assert raw_spread <= detail["max_within_visit_set_spread"] < 1e-6
    assert detail["constant_over_eligible"] is False
    eligible = [2, 3, 5, 7, 10, 11, 13, 15]
    assert removable_node_mask(graph.node_type).nonzero().view(-1).tolist() == eligible
    assert detail["distinct_values_eligible"] == 3
    assert detail["largest_tie_group_eligible"] == 4

    # a cut inside the shared set is a tie over its four members
    above = sum(value[node] > value[2] for node in eligible)
    scorer = types.SimpleNamespace(node_type=graph.node_type,
                                   probs=lambda deleted: torch.tensor([[0.6, 0.4]]))
    result = fidelity_plus_v2(scorer, np.asarray(value), 0, k=above + 2)
    assert result["tied_at_cut"] is True
    assert result["tie_group_size"] == 4


# --- explainers through the wrappers ----------------------------------------------------------

@pytest.mark.parametrize("name", BASELINES)
def test_real_explainers_run_through_the_wrapper(name):
    from core.explain.graphxai_standardized import explain_algorithms

    torch.manual_seed(53)
    graph = one_graph()
    wrapper, x = wrap_any(name, graph)
    batch_index = torch.zeros(graph.x.size(0), dtype=torch.long)
    result = explain_algorithms(wrapper, x, graph.edge_index, batch=batch_index, steps=4,
                                epochs=15, inert_edge_mask=wrapper.inert_edge_mask())
    for algo in ("GradExplainer", "IntegratedGradExplainer", "GNNExplainer"):
        assert result[algo]["status"] == "success", f"{name}/{algo}: {result[algo]}"
    provenance = result["GNNExplainer"]["provenance"]
    assert provenance["edge_gradient_verified"] is True
    assert provenance["disclosure"]
    assert result["GradExplainer"]["provenance"]["explained_columns"] == {
        "numeric": [0, NODE_DIM], "token_vector": [NODE_DIM, NODE_DIM + TOKEN_DIM]}
    if name == "graphcare":
        assert provenance["inert_edges"]["count"] == int(wrapper.inert_edge_mask().sum())


def test_explain_algorithms_isolates_one_algorithms_failure_from_the_others(monkeypatch):
    """One algorithm failing must not discard the other two algorithms' results."""
    import core.explain.graphxai_standardized as graphxai_standardized

    def _broken_get_explanation_graph(self, *args, **kwargs):
        raise RuntimeError("synthetic forced failure")

    monkeypatch.setattr(
        graphxai_standardized.GradExplainer,
        "get_explanation_graph",
        _broken_get_explanation_graph,
    )

    torch.manual_seed(29)
    graph = one_graph()
    wrapper, x = wrap_generic("gsat", graph)
    batch_index = torch.zeros(graph.x.size(0), dtype=torch.long)
    result = graphxai_standardized.explain_algorithms(
        wrapper, x, graph.edge_index, batch=batch_index, steps=4, epochs=15
    )
    assert result["GradExplainer"]["status"] == "failed"
    assert "synthetic forced failure" in result["GradExplainer"]["error"]
    for name in ("IntegratedGradExplainer", "GNNExplainer"):
        assert result[name]["status"] == "success", result[name]
