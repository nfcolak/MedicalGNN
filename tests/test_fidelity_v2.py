"""Fidelity by node deletion (core/explain/fidelity_v2.py, intervention contract version 5).

Two kinds of scorer: hand-computable toy scorers (an additive logit, a counting model) for the
metric arithmetic, ties, the random reference and the width check, and the real node-deletion scorer
of the four final methods (gsat, graphcare, protgnn, cei_gnn_mp2) on the synthetic graph record of
tests/test_node_deletion.py for the end-to-end path and node-order invariance.
"""
import math
from itertools import combinations

import numpy as np
import pytest
import torch

from core import NODE_KINDS
from core.explain import fidelity_v2 as fv2
from core.explain.node_deletion import DeletionScorer
from tests.graphxai_fixtures import NODE_DIM, NUM_TOKENS, NUM_TRIPLES, EDGE_DIM, method_args, synthetic_graph
from tests.test_node_deletion import (METHODS, P, build_model, encoder, model_forward, preprocessing,
                                      record)

PATIENT_KIND = NODE_KINDS.index("patient")
VISIT_KIND = NODE_KINDS.index("visit")
MEASUREMENT_KIND = NODE_KINDS.index("measurement")


# ------------------------------------------------------------------ toy scorers
class ToyScorer:
    """logit_1 - logit_0 = bias + sum of w over the kept nodes: deleting node i moves the logit by
    exactly -w_i, so every drop is known."""

    def __init__(self, w, bias=-3.0, node_type=None):
        self.w = np.asarray(w, dtype=float)
        self.bias = float(bias)
        self.node_type = (node_type if node_type is not None
                          else torch.full((self.w.size,), MEASUREMENT_KIND, dtype=torch.long))
        self.last_info = {"selected": 0}
        self.calls = []

    def logit(self, deleted):
        kept = np.ones(self.w.size, dtype=bool)
        kept[list(deleted)] = False
        return self.bias + math.fsum(self.w[kept])     # exactly rounded: independent of node order

    def p1(self, deleted=()):
        return 1.0 / (1.0 + math.exp(-self.logit(deleted)))

    def probs(self, deleted=()):
        deleted = sorted({int(i) for i in deleted})
        self.calls.append(tuple(deleted))
        self.last_info = {"selected": len(deleted)}
        return torch.softmax(torch.tensor([[0.0, self.logit(deleted)]], dtype=torch.float64), dim=-1)


class CountingScorer:
    """P(class 1) = (n + 1) / (n + 2) with n the number of kept eligible nodes."""

    def __init__(self, node_types):
        self.node_type = torch.tensor(node_types, dtype=torch.long)
        self.removable = fv2.removable_node_mask(self.node_type).nonzero().view(-1).tolist()
        self.last_info = {}

    def probs(self, deleted=()):
        n = len(set(self.removable) - {int(i) for i in deleted})
        return torch.softmax(torch.tensor([[0.0, math.log(n + 1)]], dtype=torch.float64), dim=-1)


THREE_REMOVABLE = [NODE_KINDS.index(kind) for kind in
                   ("patient", "visit", "complaint", "measurement", "analyte", "vital", "knowledge")]
# removable: nodes 2 (complaint), 3 (measurement), 5 (vital); P(class 1) with n kept = (n+1)/(n+2)
IMPORTANCE_3 = np.array([0, 0, 0.9, 0.5, 0, 0.1, 0.0])
RNG = np.random.default_rng(0)
# 12 evidence nodes of an additive toy: four strong positive, the rest weak
W12 = np.array([2.0, 1.6, 1.2, 0.9] + list(RNG.uniform(0.0, 0.25, 8)))


def toy_metrics(scorer, importance, *, num_points=fv2.CURVE_POINTS, tie_draws=fv2.TIE_DRAWS, seed=0,
                k=None):
    kwargs = dict(tie_draws=tie_draws, seed=seed)
    return {"fidelity_plus": fv2.fidelity_plus_v2(scorer, importance, 1, k, **kwargs),
            "fidelity_minus": fv2.fidelity_minus_v2(scorer, importance, 1, k, **kwargs),
            "deletion_curve": fv2.deletion_curve(scorer, importance, 1, num_points=num_points, **kwargs),
            "insertion_curve": fv2.insertion_curve(scorer, importance, 1, num_points=num_points, **kwargs)}


# ------------------------------------------------------------------ contract
def test_removable_node_mask_excludes_structural_and_concept_kinds():
    node_type = torch.arange(len(NODE_KINDS), dtype=torch.long)  # one of every kind
    removable = fv2.removable_node_mask(node_type)
    for kind in ("patient", "visit", "knowledge", "analyte"):
        assert not bool(removable[NODE_KINDS.index(kind)]), kind
    for kind in ("complaint", "measurement", "vital", "diagnosis"):
        assert bool(removable[NODE_KINDS.index(kind)]), kind
    assert set(fv2.EVIDENCE_NODE_KINDS) == {"complaint", "measurement", "vital", "diagnosis"}


def test_contract_version_five_states_the_operator_ties_reference_and_ranking():
    assert fv2.INTERVENTION_CONTRACT_VERSION == 5
    text = fv2.INTERVENTION_CONTRACT
    for phrase in ("node deletion in the graph record", "re-encoded with the run's own fitted preprocessing",
                   "model's own forward", "Cascade", "instance_of", "prior visits left without a node are dropped",
                   "takes the visits of its surviving anchors", "as if the removed evidence had never been recorded",
                   "not relinked", "chain broken", "ranked by |importance|", "against the class",
                   "Ties", "tie_draws, default 10", "up to 10 points", "every eligible node deleted",
                   "default 50", "half beaten", "ddof 1", "contract versions 2-4"):
        assert phrase in text, phrase
    assert "RISE" in fv2.__doc__ and "ROAR" not in fv2.__doc__
    for gone in ("_evict", "_edge_mask_for_eviction", "PAD_TOKEN_ID"):
        assert not hasattr(fv2, gone), gone
    assert (fv2.CURVE_POINTS, fv2.TIE_DRAWS, fv2.RANDOM_REPEATS) == (10, 10, 50)


# ------------------------------------------------------------------ curves by hand
def test_insertion_curve_starts_from_its_measured_k0_value_by_hand():
    scorer = CountingScorer(THREE_REMOVABLE)
    curve = fv2.insertion_curve(scorer, IMPORTANCE_3, 1, num_points=3)
    # original P = 4/5; keeping k nodes leaves P = (k+1)/(k+2): drops .3, 2/15, .05, 0 at k = 0..3
    assert curve["k_values"] == [1, 2, 3]
    assert curve["prob_at_k"] == pytest.approx([2 / 15, 0.05, 0.0])
    assert curve["k0_prob"] == pytest.approx(0.3)
    # trapezoid over (0, .3), (1/3, 2/15), (2/3, .05), (1, 0) = 1/9
    assert curve["auc"] == pytest.approx(1 / 9)
    assert curve["mean_change"] == pytest.approx(1 / 9)       # the curve spans the whole range
    assert curve["tied_at_cut"] == [False] * 3 and curve["tie_draws"] == [1] * 3


def test_deletion_curve_keeps_its_zero_start_by_hand():
    scorer = CountingScorer(THREE_REMOVABLE)
    curve = fv2.deletion_curve(scorer, IMPORTANCE_3, 1, num_points=3)
    assert curve["prob_at_k"] == pytest.approx([0.05, 2 / 15, 0.3])
    assert curve["k0_prob"] == 0.0
    assert curve["auc"] == pytest.approx(1 / 9)


def test_one_removable_node_gives_an_insertion_auc_of_half_the_k0_drop():
    scorer = CountingScorer([PATIENT_KIND, VISIT_KIND, NODE_KINDS.index("complaint")])
    curve = fv2.insertion_curve(scorer, np.array([0, 0, 1.0]), 1, num_points=3)
    assert curve["k_values"] == [1] and curve["prob_at_k"] == [0.0]
    assert curve["k0_prob"] == pytest.approx(1 / 6)
    assert curve["auc"] == pytest.approx((1 / 6) / 2)


def test_a_graph_without_removable_nodes_raises_for_both_curves_and_the_reference():
    scorer = CountingScorer([PATIENT_KIND, VISIT_KIND, PATIENT_KIND])
    for curve in (fv2.insertion_curve, fv2.deletion_curve):
        with pytest.raises(ValueError, match="no removable nodes available"):
            curve(scorer, np.array([1.0, 1.0, 1.0]), 1)
    with pytest.raises(ValueError, match="no removable nodes available"):
        fv2.random_reference(scorer, 1, [1], repeats=2, seed=0)


def test_an_explanation_with_all_its_weight_on_protected_nodes_is_refused():
    scorer = CountingScorer(THREE_REMOVABLE)
    importance = np.zeros(7)
    importance[PATIENT_KIND] = importance[VISIT_KIND] = 1e6
    with pytest.raises(ValueError, match="no nonzero importance"):
        fv2.fidelity_plus_v2(scorer, importance, 1)


def test_curve_points_default_to_ten_and_cover_every_k_of_a_short_support():
    for span in range(1, 11):
        assert fv2.curve_ks(30, span, 10) == list(range(1, span + 1))
    assert fv2.curve_ks(12, 12, 10) == [1, 2, 4, 5, 6, 7, 8, 10, 11, 12]
    curve = fv2.deletion_curve(ToyScorer(W12), W12, 1)
    assert len(curve["k_values"]) == 10 and curve["k_values"][-1] == 12
    short = W12.copy()
    short[np.argsort(-W12)[4:]] = 0.0
    assert fv2.deletion_curve(ToyScorer(W12), short, 1)["k_values"] == [1, 2, 3, 4]


# ------------------------------------------------------------------ ties
def test_a_cut_inside_a_tie_group_is_the_mean_over_every_choice_by_hand():
    scorer = ToyScorer([0.5, 1.0, 1.5, 2.0, 0.25], bias=-1.0)
    importance = np.array([5.0, 2.0, 2.0, 2.0, 1.0])
    orig = scorer.p1()
    # k = 2: node 0 always goes, one of the three tied nodes 1, 2, 3 goes with it
    plus = fv2.fidelity_plus_v2(scorer, importance, 1, 2)
    expected = np.mean([orig - scorer.p1((0, i)) for i in (1, 2, 3)])
    assert plus["prob"] == pytest.approx(expected, abs=1e-12)
    assert (plus["tied_at_cut"], plus["tie_group_size"], plus["tie_draws"]) == (True, 3, 3)
    minus = fv2.fidelity_minus_v2(scorer, importance, 1, 2)
    kept_expected = np.mean([orig - scorer.p1(tuple(sorted({1, 2, 3, 4} - {i}))) for i in (1, 2, 3)])
    assert minus["prob"] == pytest.approx(kept_expected, abs=1e-12)
    # k = 4: the whole group is inside the cut, no tie
    whole = fv2.fidelity_plus_v2(scorer, importance, 1, 4)
    assert (whole["tied_at_cut"], whole["tie_group_size"], whole["tie_draws"]) == (False, 3, 1)
    assert whole["prob"] == pytest.approx(orig - scorer.p1((0, 1, 2, 3)), abs=1e-12)
    # a binary (yes/no) explanation: every k inside its support is a tie
    binary = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    curve = fv2.deletion_curve(scorer, binary, 1)
    assert curve["k_values"] == [1, 2, 3]
    assert curve["tied_at_cut"] == [True, True, False]
    assert curve["prob_at_k"][1] == pytest.approx(
        np.mean([orig - scorer.p1(pair) for pair in combinations((1, 2, 3), 2)]), abs=1e-12)


def test_many_tied_choices_are_drawn_with_the_visit_seed_independently_of_call_order():
    scorer = ToyScorer(W12)
    flat = np.ones(12)
    first = fv2.fidelity_plus_v2(scorer, flat, 1, 3, seed=7)          # C(12, 3) = 220 > 10
    assert (first["tied_at_cut"], first["tie_group_size"], first["tie_draws"]) == (True, 12, 10)
    fv2.deletion_curve(scorer, flat, 1, seed=7)
    fv2.fidelity_plus_v2(scorer, flat, 1, 3, seed=8)
    assert fv2.fidelity_plus_v2(scorer, flat, 1, 3, seed=7) == first
    assert fv2.fidelity_plus_v2(scorer, flat, 1, 3, seed=7, tie_draws=300)["tie_draws"] == 220
    with pytest.raises(ValueError, match="tie_draws"):
        fv2.fidelity_plus_v2(scorer, flat, 1, 3, tie_draws=0)


def test_node_storage_order_does_not_change_a_constant_or_an_untied_explanation():
    """The same additive graph with its evidence stored in two orders (under contract version 4, node
    order alone moved fidelity+ of this constant explanation from 0.449 to 0.0003)."""
    order = np.argsort(W12)                         # storage B: weakest first
    a, b = ToyScorer(W12), ToyScorer(W12[order])
    flat = np.ones(12)
    # every distinct tied choice enumerated (C(12, 6) = 924): equal up to the order of summation
    ta, tb = toy_metrics(a, flat, tie_draws=1000), toy_metrics(b, flat, tie_draws=1000)
    for key in ta:
        assert ta[key]["prob" if "fidelity" in key else "auc"] == pytest.approx(
            tb[key]["prob" if "fidelity" in key else "auc"], abs=1e-12), key
    assert ta["deletion_curve"]["prob_at_k"] == pytest.approx(tb["deletion_curve"]["prob_at_k"], abs=1e-12)
    # with the default 10 draws only the sampling error remains
    da, db = toy_metrics(a, flat), toy_metrics(b, flat)
    assert abs(da["fidelity_plus"]["prob"] - db["fidelity_plus"]["prob"]) < 0.1
    # an explanation without ties: exactly equal
    ua, ub = toy_metrics(a, W12), toy_metrics(b, W12[order])
    assert ua == ub


def test_tie_summary_describes_the_ties_over_eligible_nodes():
    node_type = torch.tensor([PATIENT_KIND, VISIT_KIND] + [MEASUREMENT_KIND] * 4
                             + [NODE_KINDS.index("analyte")])
    assert fv2.tie_summary(node_type, [9.0, 9.0, 0.5, 0.5, 0.5, 0.5, 3.0]) == {
        "distinct_values_eligible": 1, "largest_tie_group_eligible": 4, "constant_over_eligible": True}
    assert fv2.tie_summary(node_type, [0, 0, 1.0, -1.0, 0.0, 2.0, 0]) == {
        "distinct_values_eligible": 3, "largest_tie_group_eligible": 2, "constant_over_eligible": False}


# ------------------------------------------------------------------ non-finite, sign, width
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_importance_raises_everywhere(bad):
    scorer = CountingScorer(THREE_REMOVABLE)
    importance = IMPORTANCE_3.copy()
    importance[3] = bad
    for call in (lambda: fv2.fidelity_plus_v2(scorer, importance, 1),
                 lambda: fv2.fidelity_minus_v2(scorer, importance, 1),
                 lambda: fv2.deletion_curve(scorer, importance, 1),
                 lambda: fv2.insertion_curve(scorer, importance, 1),
                 lambda: fv2.eligible_support(scorer.node_type, importance),
                 lambda: fv2.sparsity_over_eligible(scorer.node_type, importance),
                 lambda: fv2.tie_summary(scorer.node_type, importance),
                 lambda: fv2._ordered_removable_indices(np.abs(importance),
                                                        fv2.removable_node_mask(scorer.node_type))):
        with pytest.raises(ValueError, match="non-finite"):
            call()
    with pytest.raises(ValueError, match="shape"):
        fv2.fidelity_plus_v2(scorer, IMPORTANCE_3[:-1], 1)


def test_signed_importance_is_ranked_by_its_magnitude():
    w = W12.copy()
    w[5] = -2.5                                     # one node votes strongly against the class
    scorer = ToyScorer(w, bias=-1.0)
    signed = toy_metrics(scorer, w)
    assert signed == toy_metrics(scorer, np.abs(w))
    # node 5 is deleted first, which RAISES the class probability (the disclosed |.| ranking)
    assert fv2.fidelity_plus_v2(scorer, w, 1, 1)["prob"] < 0


def test_the_width_matched_difference_to_random_does_not_penalise_a_short_support():
    """An oracle and the same oracle cut to its top 4 rank the same top 4. The
    raw mean_change prefers the dense one only because its curve is wider; the width-matched
    difference to random (the headline) does not."""
    scorer = ToyScorer(W12)
    cut = W12.copy()
    cut[np.argsort(-W12)[4:]] = 0.0
    results = {}
    for name, importance in (("dense", W12), ("cut", cut)):
        metrics = toy_metrics(scorer, importance)
        ks = sorted(set(metrics["deletion_curve"]["k_values"]) | set(metrics["insertion_curve"]["k_values"])
                    | {metrics["fidelity_plus"]["k"], metrics["fidelity_minus"]["k"]})
        reference = fv2.random_reference(scorer, 1, ks, seed=7)
        results[name] = (metrics, reference.compare(**metrics))
    dense, cut_ = results["dense"], results["cut"]
    assert dense[0]["deletion_curve"]["mean_change"] > cut_[0]["deletion_curve"]["mean_change"]
    assert (cut_[1]["deletion_mean_change"]["difference"]
            >= dense[1]["deletion_mean_change"]["difference"])
    assert (cut_[1]["insertion_mean_change"]["difference"]
            <= dense[1]["insertion_mean_change"]["difference"])
    for key in ("prob", "k"):                     # same top-k: the same fidelity+
        assert dense[0]["fidelity_plus"][key] == cut_[0]["fidelity_plus"][key]


# ------------------------------------------------------------------ random reference
def test_a_draw_equal_to_the_source_counts_half_and_the_sd_uses_ddof_one():
    summary = fv2.RandomReference._summary(2.0, np.array([1.0, 2.0, 3.0]), higher_is_better=True)
    assert summary["beats_random_fraction"] == pytest.approx(0.5)
    assert summary["random_sd"] == pytest.approx(1.0)
    flat_scorer = ToyScorer(np.zeros(6), bias=0.3)        # no node matters: every drop is 0
    importance = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    metrics = toy_metrics(flat_scorer, importance)
    reference = fv2.random_reference(flat_scorer, 1, [1, 2, 3], seed=1)
    assert reference.repeats == 50
    comparison = reference.compare(**metrics)
    for key in ("fidelity_plus", "fidelity_minus", "deletion_auc", "insertion_auc"):
        assert comparison[key]["beats_random_fraction"] == 0.5, key
        assert comparison[key]["random_sd"] == 0.0
    with pytest.raises(ValueError, match="at least 2"):
        fv2.random_reference(flat_scorer, 1, [1], repeats=1, seed=0)


def test_the_random_reference_shares_the_anchor_and_the_auc_difference_does_not_depend_on_it():
    scorer = ToyScorer(W12)
    importance = np.random.default_rng(3).uniform(0.0, 1.0, 12)
    metrics = toy_metrics(scorer, importance)
    insertion = metrics["insertion_curve"]
    reference = fv2.random_reference(scorer, 1, insertion["k_values"] + [metrics["fidelity_plus"]["k"]],
                                     repeats=25, seed=11)
    assert reference.insertion_k0 == insertion["k0_prob"] != 0.0
    comparison = reference.compare(**metrics)
    removable, ks = insertion["removable_node_count"], insertion["k_values"]
    zero_source = fv2._auc(ks, insertion["prob_at_k"], removable)
    zero_draws = np.asarray([fv2._auc(ks, [reference.insertion[k][r] for k in ks], removable)
                             for r in range(reference.repeats)])
    assert comparison["insertion_auc"]["difference"] == pytest.approx(zero_source - zero_draws.mean(),
                                                                      abs=1e-12)
    for kind in ("deletion", "insertion"):
        auc_part, mean_part = comparison[f"{kind}_auc"], comparison[f"{kind}_mean_change"]
        width = metrics[f"{kind}_curve"]["explained_fraction"]
        assert mean_part["difference"] * width == pytest.approx(auc_part["difference"])
        assert mean_part["beats_random_fraction"] == auc_part["beats_random_fraction"]


# ------------------------------------------------------------------ the four methods, end to end
@pytest.fixture(scope="module")
def models(tmp_path_factory):
    graph, _ = record()
    prep = preprocessing([graph])
    return {name: build_model(name, prep, tmp_path_factory.mktemp(name)) for name in METHODS}, prep


def permuted(graph, membership, perm):
    """The same record with node i stored at position perm.index(i) and the edges in reverse order."""
    new_of = {old: new for new, old in enumerate(perm)}
    nodes = [graph["nodes"][old] for old in perm]
    pairs = sorted([o, new_of[i]] for o, i in membership["membership_pairs"])
    return ({**graph, "nodes": nodes, "edges": list(reversed(graph["edges"]))},
            {**membership, "membership_pairs": pairs,
             "global_node_mask": [membership["global_node_mask"][old] for old in perm]})


def real_metrics(scorer, importance, tie_draws):
    return {"fidelity_plus": fv2.fidelity_plus_v2(scorer, importance, 0, tie_draws=tie_draws),
            "fidelity_minus": fv2.fidelity_minus_v2(scorer, importance, 0, tie_draws=tie_draws),
            "deletion_curve": fv2.deletion_curve(scorer, importance, 0, tie_draws=tie_draws),
            "insertion_curve": fv2.insertion_curve(scorer, importance, 0, tie_draws=tie_draws)}


def numbers(metrics):
    out = []
    for block in metrics.values():
        for key in sorted(block):
            value = block[key]
            if key == "deleted_info":
                out.append(value)
            elif isinstance(value, list):
                out.extend(float(v) for v in value)
            else:
                out.append(float(value))
    return out


@pytest.mark.parametrize("name", METHODS)
def test_every_number_comes_from_deleting_the_named_nodes(models, name):
    built, prep = models
    graph, membership = record()
    scorer = DeletionScorer(model_forward(built[name]), graph, membership, encoder(prep), y=0)
    importance = np.arange(1.0, 17.0)                        # distinct: no ties
    orig = scorer.probs(())
    target = int(orig.argmax())
    eligible = sorted((i for i in range(16) if bool(fv2.removable_node_mask(scorer.node_type)[i])),
                      key=lambda i: -importance[i])
    plus = fv2.fidelity_plus_v2(scorer, importance, target)
    k = fv2.resolve_k(len(eligible))
    assert plus["prob"] == float(orig[0, target] - scorer.probs(eligible[:k])[0, target])
    minus = fv2.fidelity_minus_v2(scorer, importance, target)
    assert minus["prob"] == float(orig[0, target] - scorer.probs(eligible[k:])[0, target])
    insertion = fv2.insertion_curve(scorer, importance, target)
    assert insertion["k_values"] == list(range(1, 10))
    assert insertion["k0_prob"] == float(orig[0, target] - scorer.probs(eligible)[0, target])
    deletion = fv2.deletion_curve(scorer, importance, target)
    assert deletion["prob_at_k"][-1] == insertion["k0_prob"]
    assert deletion["deleted_info"]["cascaded"] == {"analyte": 2, "knowledge": 3, "orphan": 0}
    assert deletion["deleted_info"]["visits_dropped"] == 2
    reference = fv2.random_reference(scorer, target, sorted(set(deletion["k_values"]) | {k}), seed=4)
    assert reference.insertion_k0 == insertion["k0_prob"] and reference.repeats == 50
    comparison = reference.compare(fidelity_plus=plus, fidelity_minus=minus, deletion_curve=deletion,
                                   insertion_curve=insertion)
    assert set(comparison) == {"repeats", "seed", "fidelity_plus", "fidelity_minus", "deletion_auc",
                               "insertion_auc", "deletion_mean_change", "insertion_mean_change"}
    assert all(math.isfinite(v) for key, block in comparison.items() if isinstance(block, dict)
               for v in block.values())


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("kind", ["constant", "binary", "distinct"])
def test_node_storage_order_leaves_every_number_unchanged(models, name, kind):
    """Every tied choice is enumerated (tie_draws above C(9, 4) = 126), so the only difference left
    is the model's own floating-point summation order over a reordered graph."""
    built, prep = models
    graph, membership = record()
    importance = {"constant": np.ones(16),
                  "binary": np.isin(np.arange(16), [P["m1"], P["c1"], P["d1"]]).astype(float),
                  "distinct": np.arange(1.0, 17.0)}[kind]
    perm = [P[n] for n in ("d2", "m3", "k3", "v", "mp", "c2", "a1", "p", "m1", "hr", "k1", "d1", "a2",
                           "m2", "c1", "k2")]
    graph_b, membership_b = permuted(graph, membership, perm)
    forward = model_forward(built[name])
    a = DeletionScorer(forward, graph, membership, encoder(prep), y=0)
    b = DeletionScorer(forward, graph_b, membership_b, encoder(prep), y=0)
    assert torch.allclose(a.probs(()), b.probs(()), atol=1e-6)
    ma = real_metrics(a, importance, tie_draws=200)
    mb = real_metrics(b, importance[perm], tie_draws=200)
    na, nb = numbers(ma), numbers(mb)
    assert len(na) == len(nb)
    for x, y in zip(na, nb):
        if isinstance(x, dict):
            assert x == y
        else:
            assert x == pytest.approx(y, abs=1e-6)
    if kind != "distinct":
        assert any(ma["deletion_curve"]["tied_at_cut"])
    tie = fv2.tie_summary(a.node_type, importance)
    assert tie == fv2.tie_summary(b.node_type, importance[perm])
    assert tie["constant_over_eligible"] == (kind == "constant")


# ------------------------------------------------------------------ kept for tests/test_graphxai_phase_b.py
def seven_node_graph():
    """node_type cycles 0..6 (patient, visit, complaint, measurement, analyte, vital, knowledge) over
    7 nodes; membership pairs cover nodes 0-5, node 6 is the sole global node."""
    graph = synthetic_graph(
        sample_id="single", label=0, node_count=7, num_visits=3,
        membership=[[0, 0, 0, 1, 1, 2], [0, 1, 2, 3, 4, 5]],
        global_mask=[False, False, False, False, False, False, True],
    )
    graph.node_type = torch.arange(7, dtype=torch.long) % len(NODE_KINDS)
    return graph


def _wrap(name, graph):
    from core.explain.graphxai_wrapper import ClinicalGraphXAIWrapper, GraphCareGraphXAIWrapper
    from core.registry import build_method
    from core.tensorize import ALL_RELATIONS

    model = build_method(name, num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                         num_classes=2, hidden=8, layers=2, dropout=0.0, token_dim=4,
                         num_triples=NUM_TRIPLES, args=method_args()).eval()
    dims = dict(node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS, num_triples=NUM_TRIPLES,
                num_relations=len(ALL_RELATIONS))
    wrapper = (GraphCareGraphXAIWrapper(model, **dims) if name == "graphcare"
               else ClinicalGraphXAIWrapper(model, method=name, **dims))
    return wrapper, wrapper.set_context(graph)


WRAPPERS = {"gsat": lambda graph: _wrap("gsat", graph), "graphcare": lambda graph: _wrap("graphcare", graph)}
