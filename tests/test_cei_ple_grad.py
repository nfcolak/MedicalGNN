"""CEI-GNN mp2 explanations with the PLE term recomputed from the explained features.

His wrapper (cei/studies/cei_v3_graphxai.py) caches the PLE term at the original features and
holds it constant. core/explain/cei_bridge.py drops that cache, so gradients, IG and the removal
interventions reach the value column through the PLE basis. Fixtures: the synthetic mp2 graph of
tests/test_cei_mp2_explain.py (nodes 3 and 4 carry a value, every other node has has_value = 0).
"""
import pytest
import torch

from core.explain import cei_bridge, fidelity_v2, graphxai_standardized as gx, ig_clinical
from core.explain.cei_bridge import PLE_TERM, CEIGraphXAIWrapper
from tests.test_cei_gnn_v3_core import HIDDEN, LAYOUT, NODE_DIM, TOKEN_DIM
from tests.test_cei_mp2_explain import _batch, _mp2_adapter

VALUED, UNVALUED = (3, 4), (0, 1, 2, 5, 6, 7)
VALUE_COLUMN = LAYOUT.index("scaled_value")
HAS_VALUE_COLUMN = LAYOUT.index("has_value")


@pytest.fixture
def mp2(tmp_path):
    adapter = _mp2_adapter(tmp_path)
    wrapper = CEIGraphXAIWrapper(adapter, method="cei_gnn_mp2")
    batch = _batch()
    return adapter, wrapper, batch, wrapper.set_context(batch)


def _fixed(adapter, batch):
    """The wrapper this change replaces: his CEIV3GraphXAIWrapper with the PLE term cached."""
    from cei.studies.cei_v3_graphxai import CEIV3GraphXAIWrapper

    return CEIV3GraphXAIWrapper(adapter, batch).eval()


def _fixed_wrapper(adapter, batch):
    """The same runner-facing wrapper, wired to the fixed-PLE inner (for the fidelity code)."""
    wrapper = CEIGraphXAIWrapper(adapter, method="cei_gnn_mp2")
    wrapper.set_context(batch)
    wrapper.inner = _fixed(adapter, batch)
    return wrapper


def _grad(model, x, edge_index, target):
    point = x.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(model(point, edge_index)[0, target], point)
    return grad


def _target(wrapper, x, edge_index):
    with torch.no_grad():
        return int(wrapper(x, edge_index).argmax(-1))


# ------------------------------------------------------------------ at the original features
def test_logits_at_the_original_features_equal_the_fixed_wrapper_and_the_adapter(mp2):
    adapter, wrapper, batch, x = mp2
    fixed = _fixed(adapter, batch)
    with torch.no_grad():
        got = wrapper(x, batch.edge_index)
        reference = adapter(batch, epoch=0).logits
        frozen = fixed(x, batch.edge_index)
    assert (got - frozen).abs().max().item() <= 1e-6
    assert (got - reference).abs().max().item() <= 1e-6
    assert wrapper.verify(x, batch.edge_index)[0]


def test_the_cached_constant_is_gone_and_the_continuous_term_is_installed(mp2):
    adapter, wrapper, batch, _ = mp2
    network = wrapper.inner.network
    assert not hasattr(network, "_graphxai_fixed_ple")
    assert network._ple_term.__func__ is cei_bridge._smooth_ple_term
    assert hasattr(_fixed(adapter, batch).network, "_graphxai_fixed_ple")    # what we replaced
    assert "_ple_term" not in vars(adapter.network)                          # checkpoint copy untouched
    assert network.ple_active


def test_the_term_follows_the_features_it_is_given(mp2):
    _, wrapper, batch, x = mp2
    network = wrapper.inner.network
    clinical = wrapper.inner.clinical
    with torch.no_grad():
        term = network._ple_term(x, clinical)
        moved = x.clone()
        moved[3, VALUE_COLUMN] = -1.5
        assert not torch.equal(network._ple_term(moved, clinical)[3], term[3])
        assert torch.equal(network._ple_term(moved, clinical)[4], term[4])
        assert not torch.allclose(wrapper(moved, batch.edge_index), wrapper(x, batch.edge_index))


# ------------------------------------------------------------------ gradients
def test_the_value_column_gradient_reaches_the_ple_basis_only_for_valued_nodes(mp2):
    adapter, wrapper, batch, x = mp2
    fixed = _fixed(adapter, batch)
    target = _target(wrapper, x, batch.edge_index)
    # the numeric block also feeds the node encoder directly, so the PLE path is the difference
    ple_path = (_grad(wrapper, x, batch.edge_index, target)
                - _grad(fixed, x, batch.edge_index, target))[:, VALUE_COLUMN]
    assert all(abs(float(ple_path[node])) > 1e-6 for node in VALUED), ple_path
    assert all(float(ple_path[node]) == 0.0 for node in UNVALUED), ple_path
    # the has_value gate is linear now: its gradient is live for valued nodes, zero elsewhere
    gate_path = (_grad(wrapper, x, batch.edge_index, target)
                 - _grad(fixed, x, batch.edge_index, target))[:, HAS_VALUE_COLUMN]
    assert all(abs(float(gate_path[node])) > 1e-6 for node in VALUED), gate_path
    assert all(float(gate_path[node]) == 0.0 for node in UNVALUED), gate_path


def test_grad_and_ig_node_scores_differ_from_the_fixed_ple_ones_on_valued_nodes(mp2):
    adapter, wrapper, batch, x = mp2
    fixed = _fixed(adapter, batch)
    target = _target(wrapper, x, batch.edge_index)
    scores = {}
    for name, model in (("recomputed", wrapper), ("fixed", fixed)):
        grad = _grad(model, x, batch.edge_index, target).abs().sum(dim=1)
        baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT,
                                                     **wrapper.ig_feature_spec())
        attribution, completeness = ig_clinical.integrated_gradients(
            model, x, batch.edge_index, baseline, target, 64)
        scores[name] = (grad, ig_clinical.node_importance(attribution), completeness)
    for kind in (0, 1):       # Grad, IG
        for node in VALUED:
            assert abs(float(scores["recomputed"][kind][node] - scores["fixed"][kind][node])) > 1e-6
    # IG with the fixed term was within tolerance (the PLE input never moved)
    assert scores["fixed"][2]["completeness_error_rel"] <= 0.05


def test_the_graphxai_explainers_run_on_the_recomputed_term(mp2):
    _, wrapper, batch, x = mp2
    result = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=4, epochs=2)
    assert all(result[name]["status"] == "success" for name in gx.ALGORITHMS), result
    grad = result["GradExplainer"]["node_explanation"]["node_importance"]
    assert torch.as_tensor(grad).shape == (8,)


# ------------------------------------------------------------------ IG completeness, baseline
def _completeness(wrapper, x, batch, steps):
    target = _target(wrapper, x, batch.edge_index)
    baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT,
                                                 **wrapper.ig_feature_spec())
    return baseline, target, ig_clinical.integrated_gradients(
        wrapper, x, batch.edge_index, baseline, target, steps)[1]


@pytest.mark.parametrize("steps", [16, 32, 128])
def test_ig_completeness_stays_within_the_default_tolerance(mp2, steps):
    """The has_value gate is interpolated linearly, so the path from the not-recorded baseline
    has no jump (with the hard `> 0` gate the error was 0.09 at any step count)."""
    _, wrapper, batch, x = mp2
    _, _, completeness = _completeness(wrapper, x, batch, steps)
    assert completeness["completeness_error_rel"] <= ig_clinical.DEFAULT_TOLERANCE, completeness


def test_the_logit_is_continuous_at_the_start_of_the_ig_path(mp2):
    _, wrapper, batch, x = mp2
    baseline, target, _ = _completeness(wrapper, x, batch, 4)
    with torch.no_grad():
        jump = float(wrapper(baseline + 1e-7 * (x - baseline), batch.edge_index)[0, target]
                     - wrapper(baseline, batch.edge_index)[0, target])
    assert abs(jump) < 1e-5, jump


def test_the_smooth_gate_equals_the_hard_gate_at_has_value_zero_and_one(mp2):
    """Logits at the original features and at the not-recorded baseline equal the network's own
    (hard-gated) forward; the gate is only interpolated in between."""
    adapter, wrapper, batch, x = mp2
    baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT,
                                                 **wrapper.ig_feature_spec())
    inner = wrapper.inner
    for features in (x, baseline):
        with torch.no_grad():
            hard = adapter.network.forward_continuous(
                features, batch.edge_index, inner.clinical, inner.membership,
                visit_graph=inner.visit_graph)
            assert (wrapper(features, batch.edge_index) - hard).abs().max().item() <= 1e-6
            assert torch.equal(wrapper.inner.network._ple_term(features, inner.clinical),
                               adapter.network._ple_term(features, inner.clinical))


def test_a_non_finite_value_on_a_row_without_a_value_stays_finite(mp2):
    adapter, wrapper, batch, x = mp2
    garbage = x.clone()
    garbage[list(UNVALUED), VALUE_COLUMN] = float("nan")
    network, clinical = wrapper.inner.network, wrapper.inner.clinical
    with torch.no_grad():
        term = network._ple_term(garbage, clinical)
        assert torch.isfinite(term).all()
        assert torch.equal(term, network._ple_term(x, clinical))


def test_the_term_is_linear_in_has_value_between_the_two_ends(mp2):
    _, wrapper, batch, x = mp2
    network, clinical = wrapper.inner.network, wrapper.inner.clinical
    half = x.clone()
    half[:, HAS_VALUE_COLUMN] *= 0.5
    with torch.no_grad():
        assert torch.allclose(network._ple_term(half, clinical),
                              0.5 * network._ple_term(x, clinical), atol=1e-6)


def test_the_not_recorded_baseline_gives_an_unrecorded_node_its_zero_ple_term(mp2):
    adapter, wrapper, batch, x = mp2
    baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT,
                                                 **wrapper.ig_feature_spec())
    assert torch.count_nonzero(baseline[:, VALUE_COLUMN]) == 0
    assert torch.count_nonzero(baseline[:, HAS_VALUE_COLUMN]) == 0
    network, clinical = wrapper.inner.network, wrapper.inner.clinical
    with torch.no_grad():
        term_at_baseline = network._ple_term(baseline, clinical)
        term_at_x = network._ple_term(x, clinical)
        # the same as a node that never had a value in the explained row (ple_projection has no bias)
        assert torch.count_nonzero(term_at_baseline) == 0
        assert torch.equal(term_at_baseline[list(UNVALUED)], term_at_x[list(UNVALUED)])
        assert all(torch.count_nonzero(term_at_x[node]) > 0 for node in VALUED)
        # the explained logits at the baseline are the network's own on that row ...
        own = adapter.network.forward_continuous(baseline, batch.edge_index, clinical,
                                                 wrapper.inner.membership,
                                                 visit_graph=wrapper.inner.visit_graph)
        assert torch.allclose(wrapper(baseline, batch.edge_index), own, atol=1e-6)
        # ... and no longer the fixed-PLE ones, which still carried the original value
        assert not torch.allclose(wrapper(baseline, batch.edge_index),
                                  _fixed(adapter, batch)(baseline, batch.edge_index), atol=1e-6)


# ------------------------------------------------------------------ fidelity on the fixture
def _evicting(model, wrapper, batch, x, node):
    importance = torch.zeros(8)
    importance[node] = 1.0
    return fidelity_v2.fidelity_plus_v2(
        model, x, batch.token, batch.edge_index, batch.node_type, importance,
        _target(wrapper, x, batch.edge_index), k=1)


def test_fidelity_is_unchanged_for_a_node_without_a_value_and_moves_for_a_valued_one(mp2):
    adapter, wrapper, batch, x = mp2
    fixed = _fixed_wrapper(adapter, batch)
    unvalued = (_evicting(wrapper, wrapper, batch, x, 2), _evicting(fixed, wrapper, batch, x, 2))
    assert unvalued[0] == unvalued[1]
    valued = (_evicting(wrapper, wrapper, batch, x, 3), _evicting(fixed, wrapper, batch, x, 3))
    assert valued[0] != valued[1]            # the evicted value now also leaves the PLE term


def test_the_curves_run_on_the_recomputed_term(mp2):
    _, wrapper, batch, x = mp2
    importance = torch.arange(8, dtype=torch.float32)
    target = _target(wrapper, x, batch.edge_index)
    for curve in (fidelity_v2.deletion_curve, fidelity_v2.insertion_curve):
        result = curve(wrapper, x, batch.token, batch.edge_index, batch.node_type, importance,
                       target)
        assert torch.isfinite(torch.tensor(float(result["auc"]))), result


# ------------------------------------------------------------------ provenance
def test_the_ple_term_is_declared_in_the_provenance(mp2):
    _, wrapper, batch, x = mp2
    assert PLE_TERM == ("recomputed from the explained features (differentiable; has_value gate "
                        "interpolated linearly, identical at 0 and 1)")
    wrapper.builtin_node_importance(x, batch.edge_index)
    detail = wrapper.builtin_detail()
    assert detail["ple_term"] == PLE_TERM and detail["ple_inputs"] == PLE_TERM
    assert wrapper.ig_feature_provenance()["ple_term"] == PLE_TERM
    assert wrapper.ig_feature_spec() == {"numeric_width": NODE_DIM, "zeroed_tail": TOKEN_DIM,
                                         "kept_tail": HIDDEN}       # widths only: kwargs of the baseline
    assert "recomputed from the zeroed row" in wrapper.intervention_scope
