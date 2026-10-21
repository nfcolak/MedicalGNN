"""CEI-GNN mp2 explanations: the network's own PLE term outside IG, the smoothed one inside it.

The CEI study wrapper (cei/studies/cei_v3_graphxai.py) caches the PLE term at the original
features and holds it constant. core/explain/cei_bridge.py drops that cache: every forward,
gradient and GNNExplainer step runs the network's own hard-gated term on the explained features,
and only inside `wrapper.ig_path()` is the has_value gate linear (IG completeness). Fixtures: the
synthetic mp2 graph of tests/test_cei_mp2_explain.py (nodes 3 and 4 carry a value, every other
node has has_value = 0).
"""
import copy

import pytest
import torch

from core.explain import cei_bridge, graphxai_standardized as gx, ig_clinical
from core.explain.cei_bridge import PLE_TERM_EXACT, PLE_TERM_IG, CEIGraphXAIWrapper
from tests.test_cei_gnn_v3_core import LAYOUT, NODE_DIM, TOKEN_DIM
from tests.test_cei_mp2_explain import _batch, _mp2_adapter

VALUED, UNVALUED = (3, 4), (0, 1, 2, 5, 6, 7)
VALUE_COLUMN = LAYOUT.index("scaled_value")
HAS_VALUE_COLUMN = LAYOUT.index("has_value")
WIDTH = NODE_DIM + TOKEN_DIM


@pytest.fixture
def mp2(tmp_path):
    adapter = _mp2_adapter(tmp_path)
    wrapper = CEIGraphXAIWrapper(adapter, method="cei_gnn_mp2")
    batch = _batch()
    return adapter, wrapper, batch, wrapper.set_context(batch)


def _network(adapter, batch):
    """The real network (its own hard-gated PLE term) on [x | the batch's type vectors]."""
    network = copy.deepcopy(adapter.network).eval()
    clinical, membership, visit_graph = adapter._read(batch)
    with torch.no_grad():
        types = adapter.continuous_inputs(batch)[:, WIDTH:].detach()

    def forward(x, edge_index):
        return network.forward_continuous(torch.cat((x, types), dim=-1), edge_index, clinical,
                                          membership, visit_graph=visit_graph)
    return forward


def _grad(model, x, edge_index, target):
    point = x.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(model(point, edge_index)[0, target], point)
    return grad


def _target(wrapper, x, edge_index):
    with torch.no_grad():
        return int(wrapper(x, edge_index).argmax(-1))


def _baseline(wrapper, x, batch):
    return ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT, **wrapper.ig_feature_spec())


# ------------------------------------------------------------------ outside ig_path: our network
def test_the_explained_row_is_numeric_plus_token_vectors(mp2):
    adapter, _, batch, x = mp2
    assert x.shape == (8, WIDTH)
    with torch.no_grad():
        assert torch.equal(x, adapter.continuous_inputs(batch)[:, :WIDTH])


def test_logits_at_the_input_equal_the_adapter(mp2):
    adapter, wrapper, batch, x = mp2
    with torch.no_grad():
        assert torch.equal(wrapper(x, batch.edge_index), adapter(batch, epoch=0).logits)
    assert wrapper.verify(x, batch.edge_index) == (True, 0.0)


def test_outside_ig_path_the_network_runs_its_own_ple_term(mp2):
    adapter, wrapper, batch, _ = mp2
    network = wrapper.inner.network
    assert not hasattr(network, "_graphxai_fixed_ple")              # the cached constant is gone
    assert "_ple_term" not in vars(network)                         # no instance override
    assert network._ple_term.__func__ is type(network)._ple_term
    assert wrapper.ple_term == PLE_TERM_EXACT
    assert "_ple_term" not in vars(adapter.network)                 # checkpoint copy untouched
    assert network.ple_active


def test_the_gradient_equals_the_real_networks_on_every_column(mp2):
    adapter, wrapper, batch, x = mp2
    network = _network(adapter, batch)
    for target in range(2):
        ours = _grad(wrapper, x, batch.edge_index, target)
        real = _grad(network, x, batch.edge_index, target)
        assert torch.allclose(ours, real, atol=1e-6, rtol=0), (ours - real).abs().max()
    # the PLE path is live in the value column of the valued nodes
    assert all(abs(float(real[node, VALUE_COLUMN])) > 0 for node in VALUED)


@pytest.mark.parametrize("scale", [0.1, 0.5, 0.9])
def test_a_scaled_row_gives_the_real_networks_logits(mp2, scale):
    """GNNExplainer's feature mask scales x every epoch: the wrapper is our network there too."""
    adapter, wrapper, batch, x = mp2
    network = _network(adapter, batch)
    with torch.no_grad():
        assert torch.allclose(wrapper(x * scale, batch.edge_index),
                              network(x * scale, batch.edge_index), atol=1e-6, rtol=0)
        with wrapper.ig_path():      # the smoothed gate differs in between (has_value * scale)
            assert not torch.allclose(wrapper(x * scale, batch.edge_index),
                                      network(x * scale, batch.edge_index), atol=1e-6)


def test_a_moved_value_moves_the_logits_through_the_ple_term(mp2):
    _, wrapper, batch, x = mp2
    network, clinical = wrapper.inner.network, wrapper.inner.clinical
    features = torch.cat((x, wrapper._type_vectors), dim=-1)
    with torch.no_grad():
        term = network._ple_term(features, clinical)
        moved = features.clone()
        moved[3, VALUE_COLUMN] = -1.5
        assert not torch.equal(network._ple_term(moved, clinical)[3], term[3])
        assert torch.equal(network._ple_term(moved, clinical)[4], term[4])
        shifted = x.clone()
        shifted[3, VALUE_COLUMN] = -1.5
        assert not torch.allclose(wrapper(shifted, batch.edge_index), wrapper(x, batch.edge_index))


# ------------------------------------------------------------------ inside ig_path: the smooth gate
def test_ig_path_installs_the_smooth_term_and_restores_ours(mp2):
    _, wrapper, _, _ = mp2
    network = wrapper.inner.network
    with wrapper.ig_path():
        assert network._ple_term.__func__ is cei_bridge._smooth_ple_term
        assert wrapper.ple_term == PLE_TERM_IG
        with wrapper.ig_path():                                     # nested: stays smooth
            pass
        assert network._ple_term.__func__ is cei_bridge._smooth_ple_term
    assert network._ple_term.__func__ is type(network)._ple_term
    assert wrapper.ple_term == PLE_TERM_EXACT
    with pytest.raises(RuntimeError):
        with wrapper.ig_path():
            raise RuntimeError("leave the block by an exception")
    assert wrapper.ple_term == PLE_TERM_EXACT


def test_inside_ig_path_the_gate_gradient_is_live_only_for_valued_nodes(mp2):
    adapter, wrapper, batch, x = mp2
    target = _target(wrapper, x, batch.edge_index)
    real = _grad(_network(adapter, batch), x, batch.edge_index, target)
    with wrapper.ig_path():
        smooth = _grad(wrapper, x, batch.edge_index, target)
    gate_path = (smooth - real)[:, HAS_VALUE_COLUMN]
    assert all(abs(float(gate_path[node])) > 1e-6 for node in VALUED), gate_path
    assert all(float(gate_path[node]) == 0.0 for node in UNVALUED), gate_path
    assert torch.allclose((smooth - real)[:, VALUE_COLUMN], torch.zeros(8), atol=1e-6)


@pytest.mark.parametrize("steps", [16, 32, 128])
def test_ig_completeness_holds_inside_ig_path(mp2, steps):
    """The has_value gate is interpolated linearly, so the path from the not-recorded baseline
    has no jump."""
    _, wrapper, batch, x = mp2
    target = _target(wrapper, x, batch.edge_index)
    with wrapper.ig_path():
        _, completeness = ig_clinical.integrated_gradients(
            wrapper, x, batch.edge_index, _baseline(wrapper, x, batch), target, steps)
    assert completeness["completeness_error_rel"] <= ig_clinical.DEFAULT_TOLERANCE, completeness


def test_the_escalating_ig_meets_completeness_inside_ig_path(mp2):
    _, wrapper, batch, x = mp2
    target = _target(wrapper, x, batch.edge_index)
    with wrapper.ig_path():
        attribution, info = ig_clinical.integrated_gradients_until_complete(
            wrapper, x, batch.edge_index, _baseline(wrapper, x, batch), target)
    assert info["completeness_met"] and info["steps_used"] == info["tried"][-1]["steps"]
    assert attribution.shape == x.shape


def test_the_logit_is_continuous_at_the_start_of_the_ig_path(mp2):
    _, wrapper, batch, x = mp2
    baseline = _baseline(wrapper, x, batch)
    target = _target(wrapper, x, batch.edge_index)
    with torch.no_grad(), wrapper.ig_path():
        jump = float(wrapper(baseline + 1e-7 * (x - baseline), batch.edge_index)[0, target]
                     - wrapper(baseline, batch.edge_index)[0, target])
    assert abs(jump) < 1e-5, jump


def test_the_smooth_gate_equals_the_hard_gate_at_has_value_zero_and_one(mp2):
    adapter, wrapper, batch, x = mp2
    network = _network(adapter, batch)
    clinical = wrapper.inner.clinical
    for row in (x, _baseline(wrapper, x, batch)):
        features = torch.cat((row, wrapper._type_vectors), dim=-1)
        with torch.no_grad(), wrapper.ig_path():
            assert (wrapper(row, batch.edge_index) - network(row, batch.edge_index)).abs().max() <= 1e-6
            assert torch.equal(wrapper.inner.network._ple_term(features, clinical),
                               adapter.network._ple_term(features, clinical))


def test_a_non_finite_value_on_a_row_without_a_value_stays_finite_on_the_ig_path(mp2):
    _, wrapper, _, x = mp2
    garbage = torch.cat((x, wrapper._type_vectors), dim=-1)
    clean = garbage.clone()
    garbage[list(UNVALUED), VALUE_COLUMN] = float("nan")
    network, clinical = wrapper.inner.network, wrapper.inner.clinical
    with torch.no_grad(), wrapper.ig_path():
        term = network._ple_term(garbage, clinical)
        assert torch.isfinite(term).all()
        assert torch.equal(term, network._ple_term(clean, clinical))


def test_the_smooth_term_is_linear_in_has_value(mp2):
    _, wrapper, _, x = mp2
    network, clinical = wrapper.inner.network, wrapper.inner.clinical
    full = torch.cat((x, wrapper._type_vectors), dim=-1)
    half = full.clone()
    half[:, HAS_VALUE_COLUMN] *= 0.5
    with torch.no_grad(), wrapper.ig_path():
        assert torch.allclose(network._ple_term(half, clinical),
                              0.5 * network._ple_term(full, clinical), atol=1e-6)


# ------------------------------------------------------------------ baseline
def test_the_not_recorded_baseline_zeroes_value_and_token_and_keeps_the_kind(mp2, tmp_path):
    adapter, wrapper, batch, x = mp2
    baseline = _baseline(wrapper, x, batch)
    assert wrapper.ig_feature_spec() == {"token_columns": [NODE_DIM, WIDTH]}
    assert torch.count_nonzero(baseline[:, [VALUE_COLUMN, HAS_VALUE_COLUMN]]) == 0
    assert torch.count_nonzero(baseline[:, NODE_DIM:]) == 0              # the pad embedding ...
    (tmp_path / "plain").mkdir()
    trained_like = _mp2_adapter(tmp_path / "plain", perturb=False)       # ... whose row is zero
    assert torch.count_nonzero(trained_like.network.token_embedding.weight[0]) == 0
    network, clinical = wrapper.inner.network, wrapper.inner.clinical
    with torch.no_grad():
        term = network._ple_term(torch.cat((baseline, wrapper._type_vectors), dim=-1), clinical)
    assert torch.count_nonzero(term) == 0                               # no recorded value


# ------------------------------------------------------------------ provenance per source
def test_each_explainer_records_the_ple_term_it_ran_on(mp2):
    _, wrapper, batch, x = mp2
    result = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=4, epochs=2)
    assert all(result[name]["status"] == "success" for name in gx.ALGORITHMS), result
    assert result["GradExplainer"]["provenance"]["ple_term"] == PLE_TERM_EXACT
    assert result["IntegratedGradExplainer"]["provenance"]["ple_term"] == PLE_TERM_IG
    assert result["GNNExplainer"]["provenance"]["ple_term"] == PLE_TERM_EXACT
    assert wrapper.ple_term == PLE_TERM_EXACT                          # back to our network
    assert len(result["GradExplainer"]["node_explanation"]["node_importance"]) == 8


def test_the_grad_explainer_score_is_the_real_networks_gradient(mp2):
    adapter, wrapper, batch, x = mp2
    result = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=1)
    target = _target(wrapper, x, batch.edge_index)
    real = _grad(_network(adapter, batch), x, batch.edge_index, target).abs().sum(dim=1)
    got = torch.tensor(result["GradExplainer"]["node_explanation"]["node_importance"])
    assert torch.allclose(got, real, atol=1e-6)


def test_ig_and_builtin_provenance_name_their_ple_term(mp2):
    _, wrapper, batch, x = mp2
    provenance = wrapper.ig_feature_provenance()
    assert provenance["ple_term"] == PLE_TERM_IG
    assert provenance["token_columns"] == [NODE_DIM, WIDTH]
    assert provenance["explained_columns"] == {"numeric": [0, NODE_DIM],
                                               "token_vector": [NODE_DIM, WIDTH]}
    wrapper.builtin_node_importance(x, batch.edge_index)
    assert wrapper.builtin_detail()["ple_term"] == PLE_TERM_EXACT
