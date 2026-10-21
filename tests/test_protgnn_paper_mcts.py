"""ProtGNN paper MCTS: the pure port against a verbatim copy of the official search.

`_Official*` below is the official rollout, scoring loop and result rule
(zaixizhang/ProtGNN my_mcts.py @5445ffd) with `mcts_args` replaced by a settings
object and the torch `Data` plumbing replaced by a networkx graph. Synthetic data only.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from types import SimpleNamespace

import networkx as nx
import pytest
import torch
from torch_geometric.data import Batch

from core.explain.protgnn_paper_mcts import (
    OFFICIAL_MCTS, mcts_search, paper_explain, paper_explain_detail,
)
from protgnn.adapter import ProtGNNAdapter
from tests.graphxai_fixtures import (
    EDGE_DIM, NODE_DIM, NUM_CLASSES, NUM_TOKENS, NUM_TRIPLES, method_args,
    synthetic_graph,
)

MCTS_ARGS = SimpleNamespace(**OFFICIAL_MCTS)


# --- official code, copied (mcts_args -> MCTS_ARGS, Data plumbing -> networkx) ---
class _OfficialNode:
    def __init__(self, coalition, ori_graph, c_puct=10.0, W=0, N=0, P=0):
        self.coalition = coalition
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


def _official_rollout(tree_node, state_map, graph, score_func):
    cur_graph_coalition = tree_node.coalition
    if len(cur_graph_coalition) <= MCTS_ARGS.min_atoms:
        return tree_node.P

    if len(tree_node.children) == 0:
        node_degree_list = list(graph.subgraph(cur_graph_coalition).degree)
        node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=MCTS_ARGS.high2low)
        all_nodes = [x[0] for x in node_degree_list]

        if len(all_nodes) < MCTS_ARGS.expand_atoms:
            expand_nodes = all_nodes
        else:
            expand_nodes = all_nodes[:MCTS_ARGS.expand_atoms]

        for each_node in expand_nodes:
            subgraph_coalition = [node for node in all_nodes if node != each_node]

            subgraphs = [graph.subgraph(c)
                         for c in nx.connected_components(graph.subgraph(subgraph_coalition))]
            main_sub = subgraphs[0]
            for sub in subgraphs:
                if sub.number_of_nodes() > main_sub.number_of_nodes():
                    main_sub = sub

            new_graph_coalition = sorted(list(main_sub.nodes()))

            Find_same = False
            for old_graph_node in state_map.values():
                if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                    new_node = old_graph_node
                    Find_same = True

            if Find_same == False:
                new_node = _OfficialNode(new_graph_coalition, ori_graph=graph,
                                         c_puct=MCTS_ARGS.c_puct)
                state_map[str(new_graph_coalition)] = new_node

            Find_same_child = False
            for cur_child in tree_node.children:
                if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                    Find_same_child = True

            if Find_same_child == False:
                tree_node.children.append(new_node)

        scores = _official_compute_scores(score_func, tree_node.children)
        for child, score in zip(tree_node.children, scores):
            child.P = score

    sum_count = sum([c.N for c in tree_node.children])
    selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
    v = _official_rollout(selected_node, state_map, graph, score_func)
    selected_node.W += v
    selected_node.N += 1
    return v


def _official_compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition)
        else:
            score = child.P
        results.append(score)
    return results


def _official_mcts(graph, score_func):
    num_nodes = graph.number_of_nodes()
    root_coalition = sorted([i for i in range(num_nodes)])
    root = _OfficialNode(root_coalition, ori_graph=graph, c_puct=MCTS_ARGS.c_puct)
    state_map = {str(root.coalition): root}
    for rollout_id in range(MCTS_ARGS.rollout):
        _official_rollout(root, state_map, graph, score_func)

    explanations = [node for _, node in state_map.items()]
    explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
    explanations = sorted(explanations, key=lambda x: len(x.coalition))

    result_node = explanations[0]
    for result_idx in range(len(explanations)):
        x = explanations[result_idx]
        if len(x.coalition) <= MCTS_ARGS.max_atoms and x.P > result_node.P:
            result_node = x
    return result_node.coalition, result_node.P, len(state_map)


# --- synthetic graphs and mock scores ---
def _random_graph(seed, *, connected):
    rng = random.Random(seed)
    n = rng.randint(6, 30)
    edges = set()
    if connected:
        for node in range(1, n):
            edges.add((rng.randrange(node), node))
        for _ in range(rng.randint(0, n)):
            u, v = rng.sample(range(n), 2)
            edges.add((min(u, v), max(u, v)))
    else:
        cut = rng.randint(2, n - 2)
        for lo, hi in ((0, cut), (cut, n)):
            for node in range(lo + 1, hi):
                edges.add((rng.randrange(lo, node), node))
        for _ in range(rng.randint(0, n // 2)):
            u, v = rng.sample(range(n), 2)
            if (u < cut) == (v < cut) and rng.random() < 0.7:
                edges.add((min(u, v), max(u, v)))
        if rng.random() < 0.5:  # an isolated node and a self loop, as to_networkx keeps them
            edges = {e for e in edges if n - 1 not in e}
            edges.add((0, 0))
    return n, sorted(edges)


def _hash_score(coalition):
    """Deterministic, order-sensitive-free, with exact zeros to hit the re-score quirk."""
    total = sum((i + 1) * (i + 3) for i in coalition)
    if total % 5 == 0:
        return 0.0
    return ((total * 2654435761) % 1000) / 100.0 + 0.01


class _Counting:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, coalition):
        self.calls += 1
        return self.fn(coalition)


def _nx_graph(n, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return graph


def _both(n, edges, fn):
    mine = _Counting(fn)
    got = mcts_search(n, edges, mine)
    reference = _Counting(fn)
    coalition, score, states = _official_mcts(_nx_graph(n, edges), reference)
    return got, mine, (coalition, score, states), reference


def test_official_settings_are_the_mcts_parser_defaults():
    assert OFFICIAL_MCTS == {
        "rollout": 10, "c_puct": 5.0, "min_atoms": 5, "max_atoms": 10,
        "expand_atoms": 10, "high2low": False}


@pytest.mark.parametrize("connected", [True, False])
@pytest.mark.parametrize("seed", range(8))
def test_port_equals_the_official_search(seed, connected):
    n, edges = _random_graph(seed, connected=connected)
    got, mine, (coalition, score, states), reference = _both(n, edges, _hash_score)
    assert got["coalition"] == coalition
    assert got["score"] == score
    assert got["states_scored"] == mine.calls == reference.calls
    assert got["states"] == states
    assert 0 < got["states_scored"]


@pytest.mark.parametrize("seed", range(4))
def test_port_equals_the_official_search_when_every_score_is_zero(seed):
    n, edges = _random_graph(100 + seed, connected=True)
    got, mine, (coalition, score, states), reference = _both(n, edges, lambda c: 0.0)
    assert (got["coalition"], got["score"]) == (coalition, score)
    # States with P == 0 are scored again every time a parent is expanded.
    assert got["states_scored"] == mine.calls == reference.calls
    assert got["states_scored"] >= got["states"] - 1


@pytest.mark.parametrize("seed", range(4))
def test_result_rule_with_ties_takes_the_first_smallest_state(seed):
    n, edges = _random_graph(200 + seed, connected=True)
    got, _, (coalition, score, _), _ = _both(n, edges, lambda c: 1.0)
    assert (got["coalition"], got["score"]) == (coalition, score) and score == 1.0
    # Equal scores: no later state is strictly better, so the first state of the
    # smallest size (stable on discovery order) wins.
    reference_states = {}
    _official_mcts_states(_nx_graph(n, edges), reference_states)
    smallest = min(len(c) for c in reference_states)
    first_smallest = next(c for c in reference_states if len(c) == smallest)
    assert got["coalition"] == list(first_smallest)


def _official_mcts_states(graph, out):
    root = _OfficialNode(sorted(range(graph.number_of_nodes())), ori_graph=graph,
                         c_puct=MCTS_ARGS.c_puct)
    state_map = {str(root.coalition): root}
    for _ in range(MCTS_ARGS.rollout):
        _official_rollout(root, state_map, graph, lambda c: 1.0)
    for node in state_map.values():
        out[tuple(node.coalition)] = node.P


def test_result_rule_prefers_higher_score_within_max_atoms_over_smaller_state():
    # Path of 12 nodes: states are contiguous runs. Score peaks on a run of 8.
    n, edges = 12, [(i, i + 1) for i in range(11)]

    def fn(coalition):
        return 10.0 if len(coalition) == 8 else 1.0 / len(coalition)

    got, _, (coalition, score, _), _ = _both(n, edges, fn)
    assert got["coalition"] == coalition and got["score"] == score


def test_leaf_at_min_atoms_is_not_expanded():
    # Path of 6 nodes: only the root is above min_atoms. Removing each node (keeping
    # the largest component, first found on ties) gives six distinct states, by hand:
    # [1-5], [0-4], [2-5], [3-5], [0-2], [0-3]. The size-5 states are leaves, so
    # nothing is derived from them and nothing else is scored.
    n, edges = 6, [(i, i + 1) for i in range(5)]
    scored = []

    def fn(coalition):
        scored.append(tuple(coalition))
        return 1.0 + len(coalition) / 100.0

    got = mcts_search(n, edges, fn)
    assert sorted(scored) == sorted([(1, 2, 3, 4, 5), (0, 1, 2, 3, 4), (2, 3, 4, 5),
                                     (3, 4, 5), (0, 1, 2), (0, 1, 2, 3)])
    assert got["states"] == 7 and got["states_scored"] == 6
    assert len(got["coalition"]) <= OFFICIAL_MCTS["max_atoms"]


@pytest.mark.parametrize("n", [1, 4, 5])
def test_graph_at_most_min_atoms_returns_the_unscored_root(n):
    edges = [(i, i + 1) for i in range(n - 1)]
    scorer = _Counting(lambda c: 1.0)
    got = mcts_search(n, edges, scorer)
    assert got["coalition"] == list(range(n))
    assert got["score"] == 0.0 and scorer.calls == 0 and got["states"] == 1


def test_settings_override_changes_the_search():
    n, edges = _random_graph(3, connected=True)
    default = mcts_search(n, edges, _hash_score)
    small = mcts_search(n, edges, _hash_score, {"rollout": 1})
    assert small["states_scored"] <= default["states_scored"]


# --- on a tiny synthetic ProtGNN model ---
def _kwargs():
    return dict(
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_classes=NUM_CLASSES, hidden=8, layers=1, dropout=0.0, token_dim=4,
        num_triples=NUM_TRIPLES,
        args=method_args(min_atoms=2, max_atoms=4, rollout=3, expand_atoms=4,
                         prototypes_per_class=2))


def _model(seed=0):
    torch.manual_seed(seed)
    model = ProtGNNAdapter(**_kwargs())
    model.eval()
    return model


def _batch(node_count, label=0):
    graph = synthetic_graph(
        sample_id=f"g{node_count}", label=label, node_count=node_count, num_visits=1,
        membership=[0] * node_count, global_mask=[False] * node_count)
    return Batch.from_data_list([graph])


def test_graph_with_at_most_min_atoms_nodes_raises():
    model = _model()
    with pytest.raises(RuntimeError, match="needs more than 5 nodes"):
        paper_explain_detail(model, _batch(5))
    with pytest.raises(RuntimeError, match="needs more than 5 nodes"):
        paper_explain(model, _batch(4))
    assert paper_explain_detail(model, _batch(6))["subgraph_nodes"]


def test_detail_keys_prototype_and_mask_follow_the_adapter():
    model = _model(1)
    for node_count in (7, 9, 12):
        batch = _batch(node_count, label=node_count % NUM_CLASSES)
        own = model.explain_detail(batch)
        paper = paper_explain_detail(model, batch)
        assert set(paper) == set(own) | {"search"}
        assert paper["prototype_index"] == own["prototype_index"]
        assert paper["predicted_class"] == own["predicted_class"]
        assert paper["prototype_contributions"] == own["prototype_contributions"]
        search = paper["search"]
        assert search["settings"] == OFFICIAL_MCTS
        assert "5445ffd" in search["official_commit"]
        assert search["paper_text_values"] == {"iterations": 20, "expand": 10, "n_min": 5}
        assert "zero-fills" in search["subgraph_scoring"]
        assert "projected with the adapter's own search" in search["note"]
        assert search["states_scored"] > 0

        mask = paper_explain(model, batch)
        assert mask.shape == (node_count,)
        assert set(mask.tolist()) <= {0.0, 1.0}
        assert [int(i) for i in mask.nonzero().view(-1)] == paper["subgraph_nodes"]
        coalition = paper["subgraph_nodes"]
        assert paper["subgraph_similarity"] is not None
        assert 0 < len(coalition) <= node_count


def test_paper_explanation_is_deterministic_and_leaves_the_model_alone():
    model = _model(2)
    batch = _batch(10)
    with torch.no_grad():
        before = model(batch, epoch=0).logits.clone()
    first, second = paper_explain_detail(model, batch), paper_explain_detail(model, batch)
    assert first == second
    with torch.no_grad():
        assert torch.equal(model(batch, epoch=0).logits, before)
    assert not model.training


def test_paper_search_similarity_against_the_adapter_search():
    """Count of 10 random graphs where the paper search finds a subgraph at least as
    similar to the prototype as the adapter's own search. Printed for the report."""
    at_least = 0
    for index in range(10):
        model = _model(10 + index)
        batch = _batch(7 + index, label=index % NUM_CLASSES)
        own = model.explain_detail(batch)["subgraph_similarity"]
        paper = paper_explain_detail(model, batch)["subgraph_similarity"]
        at_least += int(paper >= own - 1e-9)
    print(f"paper similarity >= adapter similarity on {at_least}/10 graphs")
    assert at_least >= 6
