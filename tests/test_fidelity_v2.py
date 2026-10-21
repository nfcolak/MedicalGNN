"""Synthetic tests for the item-#4 full-evidence-removal fidelity module.

Covers all three wrapper shapes this project built (a plain ClinicalMethodAdapter
via gsat, GraphCare's own wrapper, and a non-adapter black-box ClinicalGNN) to prove the
mechanism generalises, not just one lucky case.
"""
import math

import pytest
import torch

from core import NODE_KINDS
from core.explain import fidelity_v2 as fv2
from core.model import ClinicalGNN
from core.explain.graphxai_wrapper import (
    ClinicalGNNGraphXAIWrapper,
    ClinicalGraphXAIWrapper,
    GraphCareGraphXAIWrapper,
)
from core.tensorize import ALL_RELATIONS, PAYLOAD_WIDTH
from tests.graphxai_fixtures import (
    EDGE_DIM, NODE_DIM, NUM_CLASSES, NUM_TOKENS, NUM_TRIPLES, synthetic_graph,
)
from tests.test_clinical_graphxai_wrapper import build_adapter

NUM_RELATIONS = len(ALL_RELATIONS)
PATIENT_KIND = NODE_KINDS.index("patient")
VISIT_KIND = NODE_KINDS.index("visit")


def seven_node_graph():
    """node_type cycles 0..6 (patient, visit, complaint, measurement, analyte,
    vital, knowledge) over 7 nodes -- one of each kind including both
    structural kinds, so removability logic is actually exercised, not
    vacuously true. Membership pairs cover nodes 0-5 (every non-global node);
    node 6 (knowledge) is the sole global node."""
    graph = synthetic_graph(
        sample_id="single", label=0, node_count=7, num_visits=3,
        membership=[[0, 0, 0, 1, 1, 2], [0, 1, 2, 3, 4, 5]],
        global_mask=[False, False, False, False, False, False, True],
    )
    graph.node_type = torch.arange(7, dtype=torch.long) % len(NODE_KINDS)
    return graph


def wrap_gsat(graph):
    model = build_adapter("gsat")
    model.eval()
    wrapper = ClinicalGraphXAIWrapper(
        model, method="gsat", node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


def wrap_graphcare(graph):
    model = build_adapter("graphcare")
    model.eval()
    wrapper = GraphCareGraphXAIWrapper(
        model, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


def wrap_black_box(graph):
    model = ClinicalGNN(
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=NUM_CLASSES,
        hidden=8, layers=2, dropout=0.0, token_dim=4, conv="edge_conditioned",
        num_triples=NUM_TRIPLES, payload_dim=PAYLOAD_WIDTH,
    )
    model.eval()
    wrapper = ClinicalGNNGraphXAIWrapper(
        model, method="edge_conditioned", node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_tokens=NUM_TOKENS, num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


WRAPPERS = {"gsat": wrap_gsat, "graphcare": wrap_graphcare, "black_box": wrap_black_box}


def test_removable_node_mask_excludes_structural_and_concept_kinds():
    node_type = torch.arange(len(NODE_KINDS), dtype=torch.long)  # one of every kind
    removable = fv2.removable_node_mask(node_type)
    # patient/visit are scaffolding; knowledge and analyte are concept nodes, not facts
    for kind in ("patient", "visit", "knowledge", "analyte"):
        assert not bool(removable[NODE_KINDS.index(kind)]), kind
    for kind in ("complaint", "measurement", "vital", "diagnosis"):
        assert bool(removable[NODE_KINDS.index(kind)]), kind
    assert set(fv2.EVIDENCE_NODE_KINDS) == {"complaint", "measurement", "vital", "diagnosis"}


@pytest.mark.parametrize("name", ["gsat", "graphcare", "black_box"])
def test_fidelity_plus_and_minus_v2_are_finite_and_well_formed(name):
    torch.manual_seed(101)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.rand(graph.x.size(0)).numpy()

    plus = fv2.fidelity_plus_v2(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance, target
    )
    minus = fv2.fidelity_minus_v2(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance, target
    )
    for result in (plus, minus):
        assert math.isfinite(result["prob"])
        assert result["k"] >= 1
        # 7 nodes: patient, visit, knowledge, analyte protected -> complaint,
        # measurement, vital = 3 removable.
        assert result["removable_node_count"] == 3


@pytest.mark.parametrize("name", ["gsat", "graphcare", "black_box"])
def test_structural_nodes_are_never_selected_even_with_maximal_importance(name):
    """Give the patient and visit nodes the highest possible importance score
    and confirm they're still never evicted -- removability is enforced by
    node kind, not merely by being low-ranked in practice."""
    torch.manual_seed(103)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.full((graph.x.size(0),), 1e-3).numpy().copy()
    importance[PATIENT_KIND] = 1e6  # node 0 IS kind 'patient' by construction above
    importance[VISIT_KIND] = 1e6    # node 1 IS kind 'visit'

    # k = all 3 removable nodes: if structural nodes leaked into the ordering,
    # they would be evicted first and the removable_node_count would be wrong.
    plus = fv2.fidelity_plus_v2(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance, target, k=3
    )
    assert plus["removable_node_count"] == 3
    assert plus["k"] == 3


@pytest.mark.parametrize("name", ["gsat", "graphcare", "black_box"])
def test_an_explanation_with_all_its_weight_on_protected_nodes_is_refused(name):
    """Zero importance on every eligible node means the explanation names no
    evidence. It is refused, not silently padded with tie-break nodes."""
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    importance = torch.zeros(graph.x.size(0)).numpy().copy()
    importance[PATIENT_KIND] = importance[VISIT_KIND] = 1e6
    with pytest.raises(ValueError, match="no nonzero importance"):
        fv2.fidelity_plus_v2(wrapper, x, graph.token, graph.edge_index, graph.node_type,
                             importance, 0)


@pytest.mark.parametrize("name", ["gsat", "graphcare", "black_box"])
def test_full_eviction_changes_token_not_only_x(name):
    """The whole point of item #4: prove the intervention actually reaches
    token (and edges), not only x. Construct an importance vector that always
    selects the same single node, zero x for it by hand (the OLD v1-style
    intervention) and confirm the model's prediction under v1-style zeroing
    still differs from v2's full eviction for at least one node in the graph
    -- i.e. token/edge removal is doing something x-zeroing alone doesn't.
    """
    torch.manual_seed(107)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    removable = fv2.removable_node_mask(graph.node_type)
    removable_indices = removable.nonzero(as_tuple=True)[0].tolist()
    assert removable_indices, "fixture must have at least one removable node"

    found_a_difference = False
    for node_index in removable_indices:
        x_v1 = x.clone()
        x_v1[node_index] = 0.0
        with torch.no_grad():
            v1_probs = torch.softmax(wrapper(x_v1, graph.edge_index), dim=-1)

        x_v2, token_v2, evicted = fv2._evict(x, graph.token, torch.tensor([node_index]))
        edge_mask = fv2._edge_mask_for_eviction(graph.edge_index, evicted)
        with torch.no_grad():
            v2_probs = torch.softmax(
                wrapper(x_v2, graph.edge_index, edge_mask=edge_mask, token_override=token_v2),
                dim=-1,
            )
        if not torch.allclose(v1_probs, v2_probs, atol=1e-6):
            found_a_difference = True
            break
    assert found_a_difference, (
        f"{name}: v2's token+edge eviction never differed from v1's x-only "
        "zeroing on any removable node -- the extra channels may not be wired in"
    )


@pytest.mark.parametrize("name", ["gsat", "graphcare", "black_box"])
def test_deletion_and_insertion_curves_are_well_formed(name):
    torch.manual_seed(109)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.rand(graph.x.size(0)).numpy()

    deletion = fv2.deletion_curve(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance,
        target, num_points=4,
    )
    insertion = fv2.insertion_curve(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance,
        target, num_points=4,
    )
    for curve in (deletion, insertion):
        assert curve["k_values"] == sorted(curve["k_values"])
        assert curve["k_values"][-1] <= curve["removable_node_count"]
        assert len(curve["prob_at_k"]) == len(curve["k_values"])
        assert all(math.isfinite(p) for p in curve["prob_at_k"])
        assert math.isfinite(curve["auc"])


def test_fidelity_minus_v2_with_k_equal_to_all_removable_nodes_is_the_identity():
    """keep_important evicts everyone NOT in the top-k; k == removable_count
    means nothing is evicted (the untouched graph), not an error."""
    torch.manual_seed(113)
    graph = seven_node_graph()
    wrapper, x = wrap_gsat(graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.rand(graph.x.size(0)).numpy()
    result = fv2.fidelity_minus_v2(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance, target, k=4
    )
    assert result["prob"] == 0.0
    assert result["acc"] == 0


# ------------------------------------------------------------------ insertion anchor (contract version 4)
import numpy as np   # noqa: E402  (kept next to the tests that use it)


class CountingWrapper:
    """A hand-computable model: P(class 1) = (n + 1) / (n + 2) with n the number of removable
    nodes whose x row is still nonzero, so every drop is a known fraction."""

    def __init__(self, removable_indices):
        self.removable_indices = list(removable_indices)

    def __call__(self, x, edge_index, token_override=None, edge_mask=None):
        n = int((x[self.removable_indices].abs().sum(dim=-1) > 0).sum().item())
        return torch.tensor([[0.0, math.log(n + 1)]])


def counting_graph(node_types):
    node_type = torch.tensor(node_types, dtype=torch.long)
    count = node_type.numel()
    return (torch.ones(count, 1), torch.zeros(count, dtype=torch.long),
            torch.tensor([[0, 1], [1, 2]]), node_type)


THREE_REMOVABLE = [NODE_KINDS.index(kind) for kind in
                   ("patient", "visit", "complaint", "measurement", "analyte", "vital", "knowledge")]
# removable: nodes 2 (complaint), 3 (measurement), 5 (vital); P(class 1) with n kept = (n+1)/(n+2)
IMPORTANCE_3 = np.array([0, 0, 0.9, 0.5, 0, 0.1, 0.0])


def old_auc(ks, probs, removable_count):
    """The version-3 trapezoid: a fixed (0, 0) start for both curves."""
    xs = [0.0] + [k / removable_count for k in ks]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate([0.0] + list(probs), xs))


def old_curve(wrapper, x, token, edge_index, node_type, node_importance, target_class,
              *, keep_important, num_points):
    """The version-3 `_curve`, copied as the reference the new code must match."""
    removable_count = int(fv2.removable_node_mask(node_type).sum().item())
    support = fv2.eligible_support(node_type, node_importance)
    ks = fv2.curve_ks(removable_count, max(support, 1), num_points)
    orig = fv2._predict(wrapper, x, edge_index)
    points = [fv2._score_at_k(wrapper, x, token, edge_index, node_type, node_importance,
                              target_class, k, keep_important=keep_important, orig=orig) for k in ks]
    probs = [point["prob"] for point in points]
    used = [point["k"] for point in points]
    auc = old_auc(used, probs, removable_count)
    last_fraction = used[-1] / removable_count
    return {"k_values": used, "prob_at_k": probs, "removable_node_count": removable_count,
            "support_count": support, "explained_fraction": last_fraction, "auc": auc,
            "mean_change": auc / last_fraction}


def test_contract_version_is_four_and_the_text_states_the_anchor():
    assert fv2.INTERVENTION_CONTRACT_VERSION == 4
    assert "k = 0" in fv2.INTERVENTION_CONTRACT and "version 3" in fv2.INTERVENTION_CONTRACT


def test_insertion_curve_starts_from_its_measured_k0_value_by_hand():
    x, token, edge_index, node_type = counting_graph(THREE_REMOVABLE)
    wrapper = CountingWrapper([2, 3, 5])
    curve = fv2.insertion_curve(wrapper, x, token, edge_index, node_type, IMPORTANCE_3, 1, num_points=3)
    # original P = 4/5; keeping k nodes leaves P = (k+1)/(k+2): drops .3, 2/15, .05, 0 at k = 0..3
    assert curve["k_values"] == [1, 2, 3]
    assert curve["prob_at_k"] == pytest.approx([2 / 15, 0.05, 0.0])
    assert curve["k0_prob"] == pytest.approx(0.3)
    # trapezoid over (0, .3), (1/3, 2/15), (2/3, .05), (1, 0) = 1/9 (the fixed-zero start gave 11/180)
    assert curve["auc"] == pytest.approx(1 / 9)
    assert curve["mean_change"] == pytest.approx(1 / 9)       # the curve spans the whole range
    assert old_auc(curve["k_values"], curve["prob_at_k"], 3) == pytest.approx(11 / 180)


def test_deletion_curve_keeps_its_zero_start_by_hand():
    x, token, edge_index, node_type = counting_graph(THREE_REMOVABLE)
    wrapper = CountingWrapper([2, 3, 5])
    curve = fv2.deletion_curve(wrapper, x, token, edge_index, node_type, IMPORTANCE_3, 1, num_points=3)
    assert curve["prob_at_k"] == pytest.approx([0.05, 2 / 15, 0.3])
    assert curve["k0_prob"] == 0.0
    assert curve["auc"] == pytest.approx(1 / 9)


def test_one_removable_node_gives_an_insertion_auc_of_half_the_k0_drop():
    x, token, edge_index, node_type = counting_graph([PATIENT_KIND, VISIT_KIND, NODE_KINDS.index("complaint")])
    wrapper = CountingWrapper([2])
    curve = fv2.insertion_curve(wrapper, x, token, edge_index, node_type, np.array([0, 0, 1.0]), 1,
                                num_points=3)
    # original P = 2/3, empty evidence P = 1/2: d0 = 1/6, and the only point (k = 1) has no change
    assert curve["k_values"] == [1] and curve["prob_at_k"] == [0.0]
    assert curve["k0_prob"] == pytest.approx(1 / 6)
    assert curve["auc"] == pytest.approx((1 / 6) / 2)
    assert curve["mean_change"] == pytest.approx((1 / 6) / 2)


def test_a_graph_without_removable_nodes_still_raises_for_both_curves():
    x, token, edge_index, node_type = counting_graph([PATIENT_KIND, VISIT_KIND, PATIENT_KIND])
    wrapper = CountingWrapper([])
    for curve in (fv2.insertion_curve, fv2.deletion_curve):
        with pytest.raises(ValueError, match="no removable nodes available"):
            curve(wrapper, x, token, edge_index, node_type, np.array([1.0, 1.0, 1.0]), 1)
    with pytest.raises(ValueError, match="no removable nodes available"):
        fv2.random_reference(wrapper, x, token, edge_index, node_type, 1, [1], repeats=2, seed=0)


@pytest.mark.parametrize("name", ["gsat", "graphcare", "black_box"])
def test_deletion_curve_and_the_single_point_metrics_equal_the_old_code_exactly(name):
    torch.manual_seed(211)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.rand(graph.x.size(0)).numpy()
    args = (wrapper, x, graph.token, graph.edge_index, graph.node_type, importance, target)

    new = fv2.deletion_curve(*args, num_points=4)
    old = old_curve(*args, keep_important=False, num_points=4)
    assert {key: new[key] for key in old} == old                   # bit-identical, not approx
    assert new["k0_prob"] == 0.0

    insertion = fv2.insertion_curve(*args, num_points=4)
    old_insertion = old_curve(*args, keep_important=True, num_points=4)
    for key in ("k_values", "prob_at_k", "removable_node_count", "support_count", "explained_fraction"):
        assert insertion[key] == old_insertion[key]
    removable = int(fv2.removable_node_mask(graph.node_type).sum())
    assert insertion["auc"] - old_insertion["auc"] == pytest.approx(
        0.5 * insertion["k0_prob"] * insertion["k_values"][0] / removable, abs=1e-12)

    # fidelity+/- and sparsity: re-derived by hand from the same removal path
    ordered = fv2._ordered_removable_indices(np.abs(importance), fv2.removable_node_mask(graph.node_type))
    resolved = fv2.resolve_k(removable)
    orig_probs = torch.softmax(wrapper(x, graph.edge_index), dim=-1).detach()
    for keep, result in ((False, fv2.fidelity_plus_v2(*args)), (True, fv2.fidelity_minus_v2(*args))):
        top = ordered[:resolved]
        gone = np.setdiff1d(ordered, top) if keep else top
        if gone.size == 0:
            expected = 0.0
        else:
            x_gone, token_gone, evicted = fv2._evict(x, graph.token, gone)
            mask = fv2._edge_mask_for_eviction(graph.edge_index, evicted)
            probs = fv2._predict(wrapper, x_gone, graph.edge_index, token=token_gone, edge_mask=mask)[1]
            expected = float(orig_probs[0, target] - probs[0, target])
        assert result["prob"] == expected
    from core.explain.fidelity import sparsity
    eligible = np.abs(importance)[fv2.removable_node_mask(graph.node_type).numpy()]
    assert fv2.sparsity_over_eligible(graph.node_type, importance)["mass90_sparsity"] == float(
        sparsity(eligible, 0.9))


@pytest.mark.parametrize("name", ["gsat", "graphcare", "black_box"])
def test_random_reference_shares_the_anchor_and_the_auc_comparison_is_unchanged_by_it(name):
    torch.manual_seed(223)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.rand(graph.x.size(0)).numpy() + 0.05
    args = (wrapper, x, graph.token, graph.edge_index, graph.node_type)
    plus = fv2.fidelity_plus_v2(*args, importance, target)
    minus = fv2.fidelity_minus_v2(*args, importance, target)
    deletion = fv2.deletion_curve(*args, importance, target, num_points=3)
    insertion = fv2.insertion_curve(*args, importance, target, num_points=3)
    reference = fv2.random_reference(*args, target, insertion["k_values"], repeats=25, seed=5)
    comparison = reference.compare(fidelity_plus=plus, fidelity_minus=minus,
                                   deletion_curve=deletion, insertion_curve=insertion)

    assert insertion["k0_prob"] != 0.0                         # the fixture does exercise the anchor
    assert reference.insertion_k0 == insertion["k0_prob"]      # same removal path, same number
    removable = insertion["removable_node_count"]
    ks = insertion["k_values"]
    draws = np.asarray([[reference.insertion[k][r] for k in ks] for r in range(reference.repeats)])
    anchored = np.asarray([fv2._auc(ks, row, removable, anchor=reference.insertion_k0) for row in draws])
    assert np.allclose(reference._auc_draws("insertion", ks), anchored)

    # the version-3 comparison (both sides from a fixed zero) gives the same difference and beaten fraction
    old_source = old_auc(ks, insertion["prob_at_k"], removable)
    old_draws = np.asarray([old_auc(ks, row, removable) for row in draws])
    assert comparison["insertion_auc"]["difference"] == pytest.approx(old_source - old_draws.mean(), abs=1e-12)
    assert comparison["insertion_auc"]["beats_random_fraction"] == float((old_draws > old_source).mean())
    assert comparison["insertion_auc"]["source"] == insertion["auc"]

    # mean_change comparison: the AUC keys divided by the source's own width
    for kind, curve in (("deletion", deletion), ("insertion", insertion)):
        auc_part, mean_part = comparison[f"{kind}_auc"], comparison[f"{kind}_mean_change"]
        width = curve["explained_fraction"]
        assert mean_part["source"] == curve["mean_change"]
        assert mean_part["random_mean"] * width == pytest.approx(auc_part["random_mean"])
        assert mean_part["random_sd"] * width == pytest.approx(auc_part["random_sd"])
        assert mean_part["difference"] * width == pytest.approx(auc_part["difference"])
        assert mean_part["beats_random_fraction"] == auc_part["beats_random_fraction"]


def test_random_reference_anchor_by_hand_on_the_counting_model():
    x, token, edge_index, node_type = counting_graph(THREE_REMOVABLE)
    wrapper = CountingWrapper([2, 3, 5])
    reference = fv2.random_reference(wrapper, x, token, edge_index, node_type, 1, [1, 2, 3],
                                     repeats=4, seed=1)
    assert reference.insertion_k0 == pytest.approx(0.3)
    # every random order gives the same counts here, so each draw's anchored insertion AUC is 1/9
    assert np.allclose(reference._auc_draws("insertion", [1, 2, 3]), 1 / 9)
    assert np.allclose(reference._auc_draws("deletion", [1, 2, 3]), 1 / 9)
