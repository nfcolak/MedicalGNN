"""Checks that put every explanation on the same footing, and the ProtGNN prototype check.

Two questions came out of the first review of the explanation results:

1. Fidelity+ / fidelity- are measured at the top 20% of the eligible evidence nodes, capped at the
   explanation's own support. A dense explanation (GSAT attention, an input gradient) is therefore
   scored at about a fifth of the visit's evidence, while ProtGNN's 3-6 node subgraph is scored at
   the one or two eligible nodes it contains. `matched_budget` scores every source at the SAME
   fixed numbers of deleted (or kept) nodes, next to random deletion of the same size on the same
   visit, so a difference between sources can no longer come from the number of nodes.

2. ProtGNN decides through prototype similarities of the whole-graph embedding. Its subgraph is the
   part of the visit that looks most like the explained prototype; it is not by construction the
   evidence the class probability rests on. `protgnn_prototype_check` measures, on the visit itself,
   what deleting that subgraph does to the explained prototype's similarity, to its share of the
   class logit and to the class probability, and what deleting (or keeping) the same number of
   random eligible nodes does.

Everything goes through the visit's `DeletionScorer`, i.e. the very deletion operator of the main
metrics. Nothing here trains or changes a model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch

from core import NODE_KINDS
from core.explain import fidelity_v2 as fv2
from core.explain.node_deletion import delete_from_record

BUDGET_KS = (1, 2, 3, 5, 10)
BUDGET_RANDOM_REPEATS = 20
PROTOTYPE_RANDOM_REPEATS = 20


def matched_budget(scorer, vectors: dict[str, np.ndarray], target_class: int, *,
                   ks=BUDGET_KS, repeats: int = BUDGET_RANDOM_REPEATS,
                   tie_draws: int = fv2.TIE_DRAWS, seed: int) -> dict[str, Any]:
    """Fidelity+ / fidelity- of every source at the same fixed k, with random deletion of the same
    size on the same visit.

    A source is scored at k only when it has at least k eligible nodes with nonzero importance;
    past its support the row says `beyond_support` and nothing is padded (a 3-node subgraph is not
    topped up with nodes the model did not choose). `plus` is the probability drop when the top-k
    are deleted, `minus` when only the top-k are kept; `*_random_mean` is the same with k random
    eligible nodes (`repeats` draws), and `*_minus_random` the difference to it.
    """
    removable_count = int(fv2.removable_node_mask(scorer.node_type).sum())
    chosen = [int(k) for k in ks if int(k) <= removable_count]
    if not chosen:
        return {"status": "no_eligible_evidence", "removable_node_count": removable_count}
    reference = fv2.random_reference(scorer, target_class, chosen, repeats=repeats, seed=seed)
    sources: dict[str, Any] = {}
    for name, vector in vectors.items():
        try:
            support = fv2.eligible_support(scorer.node_type, vector)
            if support == 0:
                sources[name] = {"status": "no_eligible_evidence", "support": 0}
                continue
            rows: dict[str, Any] = {}
            for k in chosen:
                if k > support:
                    rows[str(k)] = {"status": "beyond_support", "support": support}
                    continue
                plus = fv2.fidelity_plus_v2(scorer, vector, target_class, k=k, tie_draws=tie_draws, seed=seed)
                minus = fv2.fidelity_minus_v2(scorer, vector, target_class, k=k, tie_draws=tie_draws, seed=seed)
                random_plus = float(reference._draws("deletion", k).mean())
                random_minus = float(reference._draws("insertion", k).mean())
                rows[str(k)] = {
                    "plus": plus["prob"], "minus": minus["prob"],
                    "plus_random_mean": random_plus, "minus_random_mean": random_minus,
                    "plus_minus_random": plus["prob"] - random_plus,
                    "minus_minus_random": minus["prob"] - random_minus,
                    "tied_at_cut": bool(plus["tied_at_cut"]),
                }
            sources[name] = {"support": support, "by_k": rows}
        except Exception as exc:  # noqa: BLE001 - one source must not sink the others
            sources[name] = {"status": "failed", "error": str(exc)}
    return {"ks": chosen, "random_repeats": repeats, "removable_node_count": removable_count,
            "sources": sources}


def _encoded_after(scorer, deleted) -> Any:
    """The visit's record with `deleted` removed (and its cascade), encoded for the model."""
    graph, membership, _ = delete_from_record(
        scorer.graph, scorer.membership, sorted(int(i) for i in deleted), combo_rules=scorer.combo_rules)
    return scorer._encode(graph, membership)


def protgnn_prototype_check(adapter, scorer, detail: dict[str, Any], target_class: int, *,
                            repeats: int = PROTOTYPE_RANDOM_REPEATS, seed: int) -> dict[str, Any]:
    """What ProtGNN's subgraph is worth as evidence, measured on the model's own quantities.

    For the explained prototype p of class c (`detail`), and the subgraph S its search returned:
    activation of p, p's contribution to the class-c logit (classifier weight x activation), the
    class-c logit and probability, each

    - on the untouched visit (`base`),
    - after deleting the eligible nodes of S (`deleted`),
    - after deleting the same number of random eligible nodes (`random_deleted`, mean and SD over
      `repeats` draws),
    - with only those nodes kept among the eligible ones (`kept_only`), and with the same number of
      random eligible nodes kept (`random_kept`).

    The record also holds the kinds of every node of S and how many of them are removable.
    """
    node_type = scorer.node_type
    removable = fv2.removable_node_mask(node_type).detach().cpu().numpy()
    nodes = [int(n) for n in detail.get("subgraph_nodes") or []]
    kinds: dict[str, int] = {}
    for node in nodes:
        kind = NODE_KINDS[int(node_type[node])]
        kinds[kind] = kinds.get(kind, 0) + 1
    eligible_in = [n for n in nodes if removable[n]]
    candidates = np.flatnonzero(removable)
    out: dict[str, Any] = {
        "subgraph_size": len(nodes), "subgraph_kinds": kinds,
        "subgraph_removable": len(eligible_in), "removable_node_count": int(candidates.size),
        "prototype_index": int(detail["prototype_index"]), "explained_class": int(target_class),
    }
    if not eligible_in:
        out["status"] = "subgraph_has_no_removable_node"
        return out
    prototype = int(detail["prototype_index"])
    weight = float(adapter.prototype_classifier.weight[int(target_class), prototype])

    def stats(deleted) -> dict[str, float]:
        data = _encoded_after(scorer, deleted)
        batch = Batch.from_data_list([data])
        was_training = adapter.training
        adapter.eval()
        try:
            with torch.no_grad():
                _, embedding, _ = adapter._node_and_graph_embeddings(batch)
                activations, distances = adapter._prototype_activations(embedding)
                logits = adapter.prototype_classifier(activations)
        finally:
            if was_training:
                adapter.train()
        probs = torch.softmax(logits, dim=-1)
        activation = float(activations[0, prototype])
        return {"activation": activation, "distance": float(distances[0, prototype]),
                "contribution": weight * activation, "class_logit": float(logits[0, target_class]),
                "probability": float(probs[0, target_class])}

    def summarise(draws: list[dict[str, float]]) -> dict[str, Any]:
        keys = draws[0].keys()
        return {key: {"mean": float(np.mean([d[key] for d in draws])),
                      "sd": float(np.std([d[key] for d in draws], ddof=1)) if len(draws) > 1 else 0.0}
                for key in keys} | {"draws": len(draws)}

    rng = np.random.default_rng(int(seed))
    size = len(eligible_in)
    inside = set(eligible_in)
    out["base"] = stats(())
    out["deleted"] = stats(eligible_in)
    out["kept_only"] = stats([int(c) for c in candidates if int(c) not in inside])
    random_deleted, random_kept = [], []
    for _ in range(repeats):
        chosen = rng.choice(candidates, size=size, replace=False).tolist()
        random_deleted.append(stats(chosen))
        chosen_set = set(int(c) for c in chosen)
        random_kept.append(stats([int(c) for c in candidates if int(c) not in chosen_set]))
    out["random_deleted"] = summarise(random_deleted)
    out["random_kept"] = summarise(random_kept)
    out["status"] = "ok"
    return out
