"""Fidelity by deleting evidence nodes from the graph (intervention contract version 6).

`core/explain/fidelity.py` (v1) "removes" a node by zeroing its `x` row; contract versions 2-4 of
this module also set its token to the pad id and masked its edges. In both the node kept its type,
its visit membership and its share of every pooled mean, so the model still saw it. Here a node is
removed by deleting it from the raw graph record and re-encoding the record with the run's own
preprocessing (`core/explain/node_deletion.py`): the score is the model's own forward on the graph
it would have been given with that evidence deleted (not a rebuild of the graph without the events: see
the two decided departures in INTERVENTION_CONTRACT). Every metric takes a `DeletionScorer` of one visit.

Eligible evidence nodes are complaints, vitals, measurements (lab results) and prior diagnoses.
Patient and visit (scaffolding) and analyte and knowledge nodes (concepts derived from the evidence)
are never selected; the derived concept nodes leave with their evidence (cascade).

Deletion/insertion curves over k, and their AUC, are reported instead of a single top-20% point, as in
the insertion/deletion metrics of RISE: nothing is retrained, the trained
model scores every reduced graph. A random-order deletion through the same operator on the same graph
is the reference every source is compared with.

The module name fidelity_v2 is historical; INTERVENTION_CONTRACT_VERSION is the version.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any, Sequence

import numpy as np
import torch

from core import NODE_KINDS

INTERVENTION_CONTRACT_VERSION = 6
INTERVENTION_CONTRACT = (
    "evidence removal = node deletion in the graph record: the selected evidence nodes and every edge "
    "touching them are deleted from the raw graph record, which is re-encoded with the run's own fitted "
    "preprocessing and scored by the model's own forward (no masks, no wrapper). Eligible evidence "
    "nodes: complaint, vital, measurement (lab result), diagnosis (prior). Patient, visit, analyte and "
    "knowledge nodes are never selected. Combo threshold states follow combo provenance: the "
    "medical:indicates edges of one (state node, rule_id) of a combo rule form a group, and deleting any "
    "anchor of a group deletes the whole group; when the deleted node was the latest index value of a "
    "combo token, the rule is evaluated again on the next-latest value with the graph builder's own rule "
    "check and the threshold table of the graph artifact, and a rule that fires again gets its edges "
    "back; when the record lacks an input of that check the group is deleted without re-evaluation and "
    "the record names the input; without a threshold table a group whose anchors carry different tokens "
    "counts as a combo and is never re-evaluated. Cascade: an analyte goes when no kept measurement has an "
    "instance_of edge to it, a knowledge node goes when no kept non-knowledge node is linked to it "
    "(repeated until nothing changes); then every node left without an edge goes (patient and index "
    "visit excepted), as in source masking. Membership pairs of deleted nodes go and every surviving "
    "analyte or knowledge node takes the visits of its surviving anchors (an analyte: its measurements' "
    "visits, prior labs included; a knowledge node: its anchors' visits), as if the removed evidence had "
    "never been recorded; prior visits the deletion emptied are dropped, a prior visit already empty in "
    "the record is kept, and the remaining visits are renumbered in order (the index visit is always "
    "kept). This is the record with the nodes deleted, not the record the graph builder would produce "
    "without the events, in two decided ways: prior labs of an analyte whose index labs are all deleted "
    "stay, with the analyte and its knowledge nodes (the builder pulls prior history only for analytes "
    "measured at the index visit), and edges are not relinked: a baseline_of edge is not re-anchored to "
    "the next index lab or the previous prior lab, and deleting a middle measurement of a trajectory "
    "leaves the chain broken. Nodes are ranked by |importance|, so a signed explanation ranks a node that votes against "
    "the class by its magnitude. k is capped at the explanation's own support (nodes with nonzero "
    "importance). Ties: when the cut at k falls inside a group of equal |importance|, the metric is the "
    "mean over seeded random orders inside that group (tie_draws, default 10; every distinct choice when "
    "there are at most tie_draws of them). Deletion/insertion curves with up to 10 points (every k when "
    "the support is at most 10) and AUC over k/removable_count; the insertion curve starts from its "
    "measured k = 0 value (every eligible node deleted, with its cascade), the deletion curve from "
    "(0, 0). Random reference: random deletion orders (default 50) through the same operator; a draw "
    "equal to the source counts as half beaten; SD with ddof 1. Not comparable with fidelity v1 "
    "(core/explain/fidelity.py), with contract versions 2-4 (features zeroed and edges masked, the "
    "node itself kept) or with version 5 (combo states kept while any component was left, every "
    "prior visit without a node dropped)."
)
STRUCTURAL_NODE_KINDS = ("patient", "visit", "knowledge", "analyte")
EVIDENCE_NODE_KINDS = tuple(kind for kind in NODE_KINDS if kind not in STRUCTURAL_NODE_KINDS)
STRUCTURAL_NODE_TYPE_IDS = tuple(NODE_KINDS.index(kind) for kind in STRUCTURAL_NODE_KINDS)
NODE_TOP_K_FRACTION = 0.2
CURVE_POINTS = 10
TIE_DRAWS = 10
RANDOM_REPEATS = 50


def removable_node_mask(node_type: torch.Tensor) -> torch.Tensor:
    """True for every node this intervention is allowed to select."""
    structural = torch.zeros_like(node_type, dtype=torch.bool)
    for type_id in STRUCTURAL_NODE_TYPE_IDS:
        structural |= node_type == type_id
    return ~structural


def resolve_k(num_removable: int, fraction: float = NODE_TOP_K_FRACTION) -> int:
    if type(num_removable) is not int or num_removable < 1:
        raise ValueError("no removable nodes to select from")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")
    return min(num_removable, max(1, int(fraction * num_removable)))


def _abs_importance(node_type: torch.Tensor, node_importance) -> np.ndarray:
    """|importance| as float, one finite value per node, or ValueError."""
    importance = np.asarray(node_importance, dtype=float)
    if importance.shape != (int(node_type.numel()),):
        raise ValueError(f"node importance has shape {importance.shape}, expected "
                         f"({int(node_type.numel())},)")
    if not np.isfinite(importance).all():
        raise ValueError("node importance has non-finite values")
    return np.abs(importance)


def _ordered_removable_indices(importance: np.ndarray, removable: torch.Tensor) -> np.ndarray:
    """Removable node indices, most-to-least |importance|, ties by lower index."""
    importance = np.asarray(importance, dtype=float)
    if not np.isfinite(importance).all():
        raise ValueError("node importance has non-finite values")
    removable_indices = np.flatnonzero(removable.detach().cpu().numpy())
    if removable_indices.size == 0:
        raise ValueError("no removable nodes available for top-k selection")
    order = sorted(removable_indices.tolist(), key=lambda i: (-abs(importance[i]), i))
    return np.asarray(order, dtype=np.int64)


def eligible_support(node_type: torch.Tensor, node_importance) -> int:
    """How many eligible evidence nodes the explanation gives nonzero importance."""
    removable = removable_node_mask(node_type).detach().cpu().numpy()
    importance = _abs_importance(node_type, node_importance)
    return int(((importance > 0) & removable).sum())


def tie_summary(node_type: torch.Tensor, importance) -> dict[str, Any]:
    """How much of the ranking over eligible nodes is decided by ties of |importance|."""
    removable = removable_node_mask(node_type).detach().cpu().numpy()
    values = _abs_importance(node_type, importance)[removable]
    if values.size == 0:
        raise ValueError("no eligible evidence nodes")
    _, counts = np.unique(values, return_counts=True)
    return {"distinct_values_eligible": int(counts.size),
            "largest_tie_group_eligible": int(counts.max()),
            "constant_over_eligible": bool(counts.size == 1)}


def _top_sets(importance: np.ndarray, ordered: np.ndarray, k: int, *, tie_draws: int,
              seed: int, keep_important: bool):
    """The top-k node sets to average over and the tie record of this cut.

    Without a tie at the cut there is one set, ordered[:k]. When the k-th node's |importance| is shared
    by nodes on both sides of the cut, the nodes above that value are always in and the rest of the k
    come from the tied group: every distinct choice when there are at most `tie_draws` of them,
    otherwise `tie_draws` seeded random choices (the seed depends on the visit seed, k and the mode
    only, never on call order)."""
    if k == 0:
        return [ordered[:0]], {"tied_at_cut": False, "tie_group_size": 0, "tie_draws": 1}
    values = importance[ordered]
    cut_value = values[k - 1]
    group = np.sort(ordered[values == cut_value])
    above = ordered[values > cut_value]
    needed = k - above.size
    if needed == group.size:
        return [ordered[:k]], {"tied_at_cut": False, "tie_group_size": int(group.size), "tie_draws": 1}
    if comb(int(group.size), int(needed)) <= tie_draws:
        choices = [np.asarray(c, dtype=np.int64) for c in combinations(group.tolist(), int(needed))]
    else:
        rng = np.random.default_rng([int(seed), int(k), int(keep_important)])
        choices = [rng.permutation(group)[:needed] for _ in range(tie_draws)]
    sets = [np.concatenate([above, choice]) for choice in choices]
    return sets, {"tied_at_cut": True, "tie_group_size": int(group.size), "tie_draws": len(sets)}


def _drop(scorer, deleted, target_class, orig_probs, orig_pred) -> tuple[float, int]:
    probs = scorer.probs(deleted)
    pred = int(probs.argmax(dim=-1).item())
    return (float(orig_probs[0, target_class] - probs[0, target_class]),
            int(orig_pred == target_class) - int(pred == target_class))


def _score_at_k(scorer, node_importance, target_class, k: int, *, keep_important: bool,
                tie_draws: int = TIE_DRAWS, seed: int = 0) -> dict[str, Any]:
    node_type = scorer.node_type
    removable = removable_node_mask(node_type)
    importance = _abs_importance(node_type, node_importance)
    ordered = _ordered_removable_indices(importance, removable)
    support = int((importance[ordered] > 0).sum())
    if support == 0:
        raise ValueError("explanation has no nonzero importance on eligible evidence nodes")
    if tie_draws < 1:
        raise ValueError("tie_draws must be positive")
    requested = int(k)
    # Never pad a short explanation (e.g. ProtGNN's 3-6 node subgraph) with nodes the
    # model did not choose: past the support, order is a tie-break, not an explanation.
    k = min(requested, ordered.size, support)
    orig_probs = scorer.probs(())
    orig_pred = int(orig_probs.argmax(dim=-1).item())
    sets, ties = _top_sets(importance, ordered, k, tie_draws=tie_draws, seed=seed,
                           keep_important=keep_important)
    drops, accs = [], []
    for top in sets:
        deleted = np.setdiff1d(ordered, top) if keep_important else top
        prob, acc = _drop(scorer, deleted.tolist(), target_class, orig_probs, orig_pred)
        drops.append(prob)
        accs.append(acc)
    return {
        "acc": float(np.mean(accs)),
        "prob": float(np.mean(drops)),
        "k": int(k),
        "k_requested": requested,
        "support_count": support,
        "target_class": int(target_class),
        "removable_node_count": int(removable.sum().item()),
        **ties,
    }


def fidelity_plus_v2(scorer, node_importance, target_class, k: int | None = None, *,
                     tie_draws: int = TIE_DRAWS, seed: int = 0) -> dict[str, Any]:
    """Delete the top-k most important eligible nodes; higher prob drop = more faithful. k defaults
    to 20% of the eligible nodes, floor 1 (the policy of core/explain/fidelity.py over the eligible
    set instead of every node)."""
    removable_count = int(removable_node_mask(scorer.node_type).sum().item())
    resolved_k = resolve_k(removable_count) if k is None else int(k)
    return _score_at_k(scorer, node_importance, target_class, resolved_k, keep_important=False,
                       tie_draws=tie_draws, seed=seed)


def fidelity_minus_v2(scorer, node_importance, target_class, k: int | None = None, *,
                      tie_draws: int = TIE_DRAWS, seed: int = 0) -> dict[str, Any]:
    """Keep only the top-k most important eligible nodes and delete every other eligible node
    (with the cascade); lower prob drop = more faithful."""
    removable_count = int(removable_node_mask(scorer.node_type).sum().item())
    resolved_k = resolve_k(removable_count) if k is None else int(k)
    return _score_at_k(scorer, node_importance, target_class, resolved_k, keep_important=True,
                       tie_draws=tie_draws, seed=seed)


def curve_ks(removable_count: int, support: int, num_points: int) -> list[int]:
    """Curve points spread over the explanation's own support (dense importance: the whole eligible
    set; a 5-node subgraph: 5 nodes); every k when the support has at most `num_points` nodes."""
    span = max(1, min(removable_count, support))
    if span <= num_points:
        return list(range(1, span + 1))
    fractions = [i / num_points for i in range(1, num_points + 1)]
    return sorted({max(1, int(round(fraction * span))) for fraction in fractions})


def _auc(ks: Sequence[int], probs: Sequence[float], removable_count: int, *,
         anchor: float = 0.0) -> float:
    # Trapezoidal AUC over k/removable_count with an explicit start point at k = 0 whose
    # ordinate is `anchor`. Deletion: 0.0 (nothing removed -> no probability change, by
    # construction). Insertion: the measured drop of the graph with every eligible node
    # deleted, which is not zero in general.
    xs = [0.0] + [k / removable_count for k in ks]
    integrate = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2 renamed trapz
    return float(integrate([float(anchor)] + list(probs), xs))


def _empty_evidence_drop(scorer, target_class) -> float:
    """Drop of the explained-class probability with every eligible node deleted (and its cascade):
    the insertion curves' k = 0 value, shared by the source and every random draw."""
    candidates = np.flatnonzero(removable_node_mask(scorer.node_type).detach().cpu().numpy())
    if candidates.size == 0:
        raise ValueError("no removable nodes available for top-k selection")
    orig_probs = scorer.probs(())
    orig_pred = int(orig_probs.argmax(dim=-1).item())
    return _drop(scorer, candidates.tolist(), target_class, orig_probs, orig_pred)[0]


def _curve(scorer, node_importance, target_class, *, keep_important: bool, num_points: int,
           tie_draws: int, seed: int) -> dict[str, Any]:
    node_type = scorer.node_type
    removable_count = int(removable_node_mask(node_type).sum().item())
    support = eligible_support(node_type, node_importance)
    ks = curve_ks(removable_count, max(support, 1), num_points)
    points = [
        _score_at_k(scorer, node_importance, target_class, k, keep_important=keep_important,
                    tie_draws=tie_draws, seed=seed)
        for k in ks
    ]
    deleted_info = scorer.last_info
    anchor = _empty_evidence_drop(scorer, target_class) if keep_important else 0.0
    probs = [point["prob"] for point in points]
    used = [point["k"] for point in points]
    auc = _auc(used, probs, removable_count, anchor=anchor)
    last_fraction = used[-1] / removable_count
    return {
        "k_values": used,
        "prob_at_k": probs,
        # probability drop at k = 0 (the curve's first ordinate): 0.0 for deletion, the
        # empty-evidence drop for insertion
        "k0_prob": float(anchor),
        "removable_node_count": removable_count,
        "support_count": support,
        "explained_fraction": last_fraction,
        "auc": auc,
        # AUC divided by the width of the curve; still width-dependent, a side column next to
        # the width-matched differences to random
        "mean_change": auc / last_fraction,
        "tied_at_cut": [point["tied_at_cut"] for point in points],
        "tie_group_size": [point["tie_group_size"] for point in points],
        "tie_draws": [point["tie_draws"] for point in points],
        # what the deletion operator removed at the last point (last tie draw)
        "deleted_info": deleted_info,
    }


def deletion_curve(scorer, node_importance, target_class, *, num_points: int = CURVE_POINTS,
                   tie_draws: int = TIE_DRAWS, seed: int = 0) -> dict[str, Any]:
    """Fidelity+ at increasing k (most-important-first deletion); AUC over k/removable_count. Higher
    AUC = deleting important evidence consistently hurts the prediction. Starts from (0, 0)."""
    return _curve(scorer, node_importance, target_class, keep_important=False,
                  num_points=num_points, tie_draws=tie_draws, seed=seed)


def insertion_curve(scorer, node_importance, target_class, *, num_points: int = CURVE_POINTS,
                    tie_draws: int = TIE_DRAWS, seed: int = 0) -> dict[str, Any]:
    """Fidelity- at increasing k (most-important-first retention, every other eligible node
    deleted); AUC over k/removable_count. Lower AUC = a small amount of important evidence
    consistently suffices to preserve the prediction.

    The curve starts at its measured k = 0 point: the probability drop of the graph with every
    eligible node deleted (`k0_prob`), not a fixed 0."""
    return _curve(scorer, node_importance, target_class, keep_important=True,
                  num_points=num_points, tie_draws=tie_draws, seed=seed)


def sparsity_over_eligible(node_type: torch.Tensor, node_importance, mass: float = 0.9) -> dict[str, Any]:
    """Conciseness of one explanation, on the same eligible nodes for every source
    (built-in or post hoc). mass90_sparsity: fraction of eligible nodes that can be
    dropped while keeping 90% of the importance mass (core/explain/fidelity.sparsity).
    support_fraction: share of eligible nodes given any importance at all -- for a
    yes/no subgraph this is simply subgraph size / eligible size."""
    from core.explain.fidelity import sparsity

    removable = removable_node_mask(node_type).detach().cpu().numpy()
    importance = _abs_importance(node_type, node_importance)[removable]
    if importance.size == 0:
        raise ValueError("no eligible evidence nodes")
    support = int((importance > 0).sum())
    return {
        "mass90_sparsity": float(sparsity(importance, mass)),
        "support_count": support,
        "support_fraction": support / importance.size,
        "eligible_node_count": int(importance.size),
    }


class RandomReference:
    """What random deletion does on this very graph: the calibration an explanation has to beat.
    Every number comes from the same deletion operator as the real explanations (only the order
    differs), so any out-of-distribution effect of deleting evidence is present in both."""

    def __init__(self, ks, deletion, insertion, removable_count, seed, insertion_k0=0.0):
        self.ks = [int(k) for k in ks]
        # the insertion curves' measured k = 0 value on this graph (every eligible node
        # deleted); every random insertion draw starts from it, like the source's curve does
        self.insertion_k0 = float(insertion_k0)
        self.deletion = {int(k): np.asarray(v, dtype=float) for k, v in deletion.items()}
        self.insertion = {int(k): np.asarray(v, dtype=float) for k, v in insertion.items()}
        self.removable_count = int(removable_count)
        self.seed = int(seed)
        self.repeats = len(next(iter(self.deletion.values())))

    def _draws(self, kind, k):
        table = self.deletion if kind == "deletion" else self.insertion
        if int(k) not in table:
            raise ValueError(f"random reference was not computed at k={int(k)} "
                             f"(has {self.ks}); build it over every k the source uses")
        return table[int(k)]

    def _auc_draws(self, kind, ks):
        return np.asarray([
            _auc(ks, [self._draws(kind, k)[r] for k in ks], self.removable_count,
                 anchor=self.insertion_k0 if kind == "insertion" else 0.0)
            for r in range(self.repeats)
        ])

    @staticmethod
    def _summary(source, draws, *, higher_is_better):
        mean = float(draws.mean())
        sd = float(draws.std(ddof=1)) if draws.size > 1 else 0.0
        beaten = (draws < source) if higher_is_better else (draws > source)
        # a draw equal to the source is half beaten (short yes/no explanations often tie)
        score = beaten.astype(float) + 0.5 * (draws == source)
        return {"source": float(source), "random_mean": mean, "random_sd": sd,
                "difference": float(source - mean),
                "beats_random_fraction": float(score.mean())}

    def compare(self, *, fidelity_plus, fidelity_minus, deletion_curve, insertion_curve):
        """Each source number against the random draws of the same graph.

        `deletion_mean_change` / `insertion_mean_change` are the AUC keys divided by the source's
        own curve width (`explained_fraction`), the same width on both sides, so their `difference`
        is the width-matched difference to random. `difference` and `beats_random_fraction` of the
        AUC keys do not depend on the insertion anchor: the first trapezoid adds the same
        1/2 * d0 * k1/removable_count to the source and to every draw, because both start from the
        same measured k = 0 value d0 and share their first k.
        """
        deletion_draws = self._auc_draws("deletion", deletion_curve["k_values"])
        insertion_draws = self._auc_draws("insertion", insertion_curve["k_values"])
        deletion_width = deletion_curve["explained_fraction"]
        insertion_width = insertion_curve["explained_fraction"]
        return {
            "repeats": self.repeats,
            "seed": self.seed,
            "fidelity_plus": self._summary(
                fidelity_plus["prob"], self._draws("deletion", fidelity_plus["k"]),
                higher_is_better=True),
            "fidelity_minus": self._summary(
                fidelity_minus["prob"], self._draws("insertion", fidelity_minus["k"]),
                higher_is_better=False),
            "deletion_auc": self._summary(
                deletion_curve["auc"], deletion_draws, higher_is_better=True),
            "insertion_auc": self._summary(
                insertion_curve["auc"], insertion_draws, higher_is_better=False),
            "deletion_mean_change": self._summary(
                deletion_curve["mean_change"], deletion_draws / deletion_width,
                higher_is_better=True),
            "insertion_mean_change": self._summary(
                insertion_curve["mean_change"], insertion_draws / insertion_width,
                higher_is_better=False),
        }


def random_reference(scorer, target_class, ks: Sequence[int], *, repeats: int = RANDOM_REPEATS,
                     seed: int) -> RandomReference:
    """Random-order deletion at each k, `repeats` times, through the same deletion operator."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2 (the SD uses ddof 1)")
    candidates = np.flatnonzero(removable_node_mask(scorer.node_type).detach().cpu().numpy())
    if candidates.size == 0:
        raise ValueError("no removable nodes available for a random reference")
    rng = np.random.default_rng(int(seed))
    orig_probs = scorer.probs(())
    orig_pred = int(orig_probs.argmax(dim=-1).item())
    ks = sorted({min(int(k), int(candidates.size)) for k in ks})
    deletion = {k: [] for k in ks}
    insertion = {k: [] for k in ks}
    # the insertion curves' start (every eligible node deleted) does not depend on the draw's
    # order; it is the very number the source curves use
    insertion_k0 = _empty_evidence_drop(scorer, target_class)
    for _ in range(repeats):
        order = rng.permutation(candidates)
        for k in ks:
            deletion[k].append(_drop(scorer, order[:k].tolist(), target_class, orig_probs, orig_pred)[0])
            insertion[k].append(_drop(scorer, order[k:].tolist(), target_class, orig_probs, orig_pred)[0])
    return RandomReference(ks, deletion, insertion, int(candidates.size), seed,
                           insertion_k0=insertion_k0)


def relation_vs_payload_contribution(wrapper, x, edge_index, edge_attr,
                                     num_relation_columns: int,
                                     target_class: int) -> dict[str, Any]:
    """How much of the edges' contribution to this prediction is the relation TYPE (the one-hot
    block, columns [0, num_relation_columns)) versus the numeric PAYLOAD (delta/interval/recency/
    prior-encounters, every column after it)? Answered by zeroing one block and measuring the
    probability drop rather than by a gradient, so it needs no machinery beyond the
    edge_attr_override hook of the wrapper. Graph-level (both blocks across every edge at once),
    not per-edge: the question is which TYPE of edge evidence this prediction leans on, not
    which single edge.
    """
    if edge_index.numel() == 0:
        return {"relation_type_prob_drop": 0.0, "numeric_payload_prob_drop": 0.0,
               "edge_count": 0}
    if edge_attr.ndim != 2 or edge_attr.size(1) <= num_relation_columns:
        raise ValueError("edge_attr must have at least one payload column "
                        "after num_relation_columns")
    with torch.no_grad():
        orig_probs = torch.softmax(wrapper(x, edge_index), dim=-1)
        relation_zeroed = edge_attr.clone()
        relation_zeroed[:, :num_relation_columns] = 0.0
        relation_zeroed_probs = torch.softmax(
            wrapper(x, edge_index, edge_attr_override=relation_zeroed), dim=-1)
        payload_zeroed = edge_attr.clone()
        payload_zeroed[:, num_relation_columns:] = 0.0
        payload_zeroed_probs = torch.softmax(
            wrapper(x, edge_index, edge_attr_override=payload_zeroed), dim=-1)
    target = int(target_class)
    return {
        # Dropping relation-type-only isolates relation type's contribution;
        # dropping payload-only isolates the payload's. The two need not sum
        # to the full-removal drop (they can overlap or partially substitute).
        "relation_type_prob_drop": float(orig_probs[0, target] - relation_zeroed_probs[0, target]),
        "numeric_payload_prob_drop": float(orig_probs[0, target] - payload_zeroed_probs[0, target]),
        "edge_count": int(edge_index.size(1)),
        "target_class": target,
    }
