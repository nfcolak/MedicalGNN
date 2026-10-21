"""ProtGNN's built-in explanation with the paper / official-code MCTS.

The search is a pure-Python port of the official ProtGNN rollout, scoring loop and
result rule (zaixizhang/ProtGNN my_mcts.py @5445ffd, settings from its
`Configures.MCTSParser`). The trained prototypes are untouched: they were
projected with the adapter's own search (`ProtGNNAdapter._mcts_project`); only the
search that explains a graph follows the official code.

One deliberate deviation, recorded in every explanation's provenance: the official
code scores a coalition by zero-filling the excluded nodes' features and running
the whole graph through the model. Here a coalition is scored the way the
prototypes were projected, by re-encoding the induced subgraph with the trained
model (`_induced_graph_embedding`) and taking the same similarity,
log((d + 1) / (d + epsilon)) with d the squared L2 distance to the prototype.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping

import networkx as nx
import torch

OFFICIAL_COMMIT = "zaixizhang/ProtGNN my_mcts.py @5445ffd"

# Configures.MCTSParser @5445ffd.
OFFICIAL_MCTS = {
    "rollout": 10,
    "c_puct": 5.0,
    "min_atoms": 5,
    "max_atoms": 10,
    "expand_atoms": 10,
    "high2low": False,
}

# Values in the paper text; provenance only, the search uses OFFICIAL_MCTS.
PAPER_TEXT_VALUES = {"iterations": 20, "expand": 10, "n_min": 5}

SUBGRAPH_SCORING = (
    "induced-subgraph re-encoding with the trained model, as used when the prototypes "
    "were projected; the official code zero-fills the excluded nodes instead")
SEARCH_NOTE = (
    "prototypes were projected with the adapter's own search; only the explanation "
    "search follows the official code")


class _Node:
    __slots__ = ("coalition", "c_puct", "children", "W", "N", "P")

    def __init__(self, coalition: list[int], c_puct: float):
        self.coalition = coalition
        self.c_puct = c_puct
        self.children: list[_Node] = []
        self.W = 0.0  # sum of node value
        self.N = 0  # times of arrival
        self.P = 0.0  # property score (reward)

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0

    def U(self, n: int) -> float:
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


def _settings(settings: Mapping) -> dict:
    merged = dict(OFFICIAL_MCTS)
    merged.update(settings)
    return merged


def mcts_search(num_nodes: int, edges: Iterable[tuple[int, int]],
                score_fn: Callable[[list[int]], float],
                settings: Mapping = OFFICIAL_MCTS) -> dict:
    """Official ProtGNN MCTS over a graph, with `score_fn(coalition) -> float`.

    Returns `{"coalition", "score", "states_scored", "states"}`: the chosen sorted
    node list, its score, the number of `score_fn` calls and the number of distinct
    states reached. Quirks of the official code are kept: the root is every node
    (disconnected ones included) and is never scored itself, a state whose score is
    exactly 0 is scored again when its parent is expanded, and the result is the
    two-sort rule over all states."""
    cfg = _settings(settings)
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_nodes)))
    graph.add_edges_from((int(u), int(v)) for u, v in edges)

    calls = [0]

    def scored(coalition: list[int]) -> float:
        calls[0] += 1
        return float(score_fn(coalition))

    root = _Node(sorted(range(graph.number_of_nodes())), cfg["c_puct"])
    state_map: dict[tuple[int, ...], _Node] = {tuple(root.coalition): root}

    def rollout(tree_node: _Node) -> float:
        coalition = tree_node.coalition
        if len(coalition) <= cfg["min_atoms"]:
            return tree_node.P

        # Expand if this node has never been visited.
        if len(tree_node.children) == 0:
            degrees = list(graph.subgraph(coalition).degree)
            degrees = sorted(degrees, key=lambda item: item[1], reverse=cfg["high2low"])
            all_nodes = [item[0] for item in degrees]
            expand_nodes = (all_nodes if len(all_nodes) < cfg["expand_atoms"]
                            else all_nodes[:cfg["expand_atoms"]])

            for each_node in expand_nodes:
                # Prune the node and keep only the largest remaining component.
                remaining = [node for node in all_nodes if node != each_node]
                components = [graph.subgraph(c)
                              for c in nx.connected_components(graph.subgraph(remaining))]
                main = components[0]
                for component in components:
                    if component.number_of_nodes() > main.number_of_nodes():
                        main = component
                new_coalition = sorted(main.nodes())

                # Identical sub-graphs are merged through the state map.
                key = tuple(new_coalition)
                new_node = state_map.get(key)
                if new_node is None:
                    new_node = _Node(new_coalition, cfg["c_puct"])
                    state_map[key] = new_node
                if not any(child.coalition == new_coalition for child in tree_node.children):
                    tree_node.children.append(new_node)

            for child in tree_node.children:
                child.P = scored(child.coalition) if child.P == 0 else child.P

        sum_count = sum(child.N for child in tree_node.children)
        selected = max(tree_node.children, key=lambda c: c.Q() + c.U(sum_count))
        value = rollout(selected)
        selected.W += value
        selected.N += 1
        return value

    for _ in range(int(cfg["rollout"])):
        rollout(root)

    states = list(state_map.values())
    states = sorted(states, key=lambda node: node.P, reverse=True)
    states = sorted(states, key=lambda node: len(node.coalition))
    result = states[0]
    for state in states:
        if len(state.coalition) <= cfg["max_atoms"] and state.P > result.P:
            result = state
    return {"coalition": list(result.coalition), "score": float(result.P),
            "states_scored": calls[0], "states": len(state_map)}


def _search_provenance(cfg: Mapping, found: Mapping) -> dict:
    return {
        "algorithm": "ProtGNN MCTS (paper / official code)",
        "official_commit": OFFICIAL_COMMIT,
        "settings": dict(cfg),
        "paper_text_values": dict(PAPER_TEXT_VALUES),
        "states_scored": int(found["states_scored"]),
        "states": int(found["states"]),
        "subgraph_scoring": SUBGRAPH_SCORING,
        "note": SEARCH_NOTE,
    }


@torch.no_grad()
def paper_explain_detail(model, batch, target_class: int | None = None) -> dict:
    """Same output as `ProtGNNAdapter.explain_detail`, plus a `search` provenance dict.

    The prototype is chosen exactly as the adapter chooses it: the own-class
    prototype with the largest classifier weight x activation. Only the subgraph
    search differs (the official MCTS, `OFFICIAL_MCTS`). Raises RuntimeError when the
    graph has at most `min_atoms` nodes, where the official search returns
    immediately without exploring anything."""
    cfg = _settings(OFFICIAL_MCTS)
    was_training = model.training
    model.eval()
    try:
        node_state, graph_embedding, graph_index = model._node_and_graph_embeddings(batch)
        if graph_embedding.size(0) != 1:
            raise ValueError("ProtGNN's built-in explanation scores one graph at a time")
        node_count = int(node_state.size(0))
        if node_count <= cfg["min_atoms"]:
            raise RuntimeError(
                "ProtGNN's paper MCTS needs more than "
                f"{cfg['min_atoms']} nodes; this graph has {node_count}")
        activations, distances = model._prototype_activations(graph_embedding)
        logits = model.prototype_classifier(activations)
        model_prediction = int(logits.argmax(dim=-1).item())
        predicted = model_prediction if target_class is None else int(target_class)
        contributions = model.prototype_classifier.weight[predicted] * activations[0]
        own = torch.nonzero(model.prototype_class_ids == predicted, as_tuple=False).view(-1)
        prototype = int(own[contributions[own].argmax()].item())

        candidate = model._projection_candidate(batch, 0, node_state, graph_index)
        vector = model.prototype_vectors[prototype]

        def score(coalition):
            embedding = model._induced_graph_embedding(candidate, coalition)
            distance = (embedding - vector).square().sum()
            return float(torch.log((distance + 1.0) / (distance + model.epsilon)).item())

        found = mcts_search(
            node_count, candidate["edge_index"].detach().cpu().t().tolist(), score, cfg)
        return {
            "kind": "prototype_mcts_subgraph",
            "predicted_class": model_prediction,
            "explained_class": predicted,
            "prototype_index": prototype,
            "prototype_activation": float(activations[0, prototype].item()),
            "prototype_distance": float(distances[0, prototype].item()),
            "logit_contribution": float(contributions[prototype].item()),
            "prototype_contributions": [float(v) for v in contributions.tolist()],
            "subgraph_nodes": [int(n) for n in found["coalition"]],
            "subgraph_similarity": float(found["score"]),
            "node_count": node_count,
            "prototypes_projected_onto_training_subgraphs":
                model.projection_summary["projected_prototypes"],
            "prototypes_unmatched": model.projection_summary["unmatched_prototypes"],
            "projection_summary_source": model.projection_summary["source"],
            "search": _search_provenance(cfg, found),
        }
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def paper_explain(model, batch, target_class: int | None = None) -> torch.Tensor:
    """Binary per-node mask over the coalition the paper MCTS found."""
    detail = paper_explain_detail(model, batch, target_class)
    importance = torch.zeros(detail["node_count"])
    importance[detail["subgraph_nodes"]] = 1.0
    return importance
