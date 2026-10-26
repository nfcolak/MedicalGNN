"""The two historical CEI GraphXAI screens (cei/studies/cei_graphxai.py and
cei/studies/cei_v3_graphxai.py) run end to end on the synthetic mp2 fixture of
tests/test_cei_mp2_explain.py: GNNExplainer keeps the vendored coefficients there, every
node explanation carries the v1 fidelity, and the screens' own export validator accepts them.
The final explanation study keeps the mean feature penalty.
"""
import pytest
import torch

from cei.studies import cei_graphxai, cei_v3_graphxai
from core.explain import graphxai_standardized as gx
from core.explain import runner
from core.explain.cei_bridge import CEIGraphXAIWrapper
from tests.test_cei_mp2_explain import _batch, _mp2_adapter

VENDORED_FEATURE = {'entropy': 0.1, 'size': 1.0}


@pytest.fixture
def coefficients(monkeypatch):
    """Every GNNExplainer built during the test: (feature_width, feature coefficients)."""
    seen = []
    original = gx.CompatibleGNNExplainer.__init__

    def spy(self, model, coeff=None, *, feature_width=None):
        original(self, model, coeff, feature_width=feature_width)
        seen.append((feature_width, dict(self.coeff['feature'])))

    monkeypatch.setattr(gx.CompatibleGNNExplainer, '__init__', spy)
    return seen


def test_the_mp2_screen_runs_with_the_vendored_coefficients_and_passes_its_export(tmp_path,
                                                                                  coefficients):
    result = cei_graphxai.explain_graph(_mp2_adapter(tmp_path), _batch(), steps=2, epochs=2)
    assert set(result) == set(gx.ALGORITHMS)
    for name, item in result.items():
        cei_graphxai._validate_algorithm_record(name, item)
        assert {'fidelity_plus', 'fidelity_minus'} <= set(item['node_explanation'])
    assert coefficients == [(None, VENDORED_FEATURE)]
    provenance = result['GNNExplainer']['provenance']
    assert provenance['feature_penalty'] == 'sum'
    assert provenance['coefficients'] == gx.GNNEXPLAINER_COEFF_NOTES['sum']


def test_the_v3_screen_runs_with_the_vendored_coefficients_and_the_v1_fidelity(tmp_path,
                                                                               coefficients):
    result, logits, diff = cei_v3_graphxai.explain_graph(_mp2_adapter(tmp_path), _batch())
    assert diff == 0.0 and torch.isfinite(torch.as_tensor(logits)).all()
    assert {name for name, item in result.items() if item['status'] == 'success'} == {
        *gx.ALGORITHMS, 'Native', 'Random'}
    for item in result.values():
        assert {'fidelity_plus', 'fidelity_minus'} <= set(item['node_explanation'])
    assert coefficients == [(None, VENDORED_FEATURE)]
    assert result['GNNExplainer']['provenance']['feature_penalty'] == 'sum'


def test_the_final_study_keeps_the_mean_feature_penalty(tmp_path, coefficients):
    adapter = _mp2_adapter(tmp_path)
    wrapper = CEIGraphXAIWrapper(adapter, method='cei_gnn_mp2')
    batch = _batch()
    x = wrapper.set_context(batch)
    with torch.no_grad():
        prediction = int(wrapper(x, batch.edge_index).argmax(-1))
    out = runner._collect_importances(wrapper, batch, x, method='cei_gnn_mp2', steps=2, epochs=2,
                                      ig=None, prediction=prediction, seed=0)
    width = x.size(1)
    assert coefficients == [(width, {'entropy': 0.1, 'size': 1.0 / width})]
    provenance = out['graphxai']['GNNExplainer']['provenance']
    assert provenance['feature_penalty'] == 'mean'
    assert provenance['coefficients'] == gx.GNNEXPLAINER_COEFF_NOTE


def test_an_unknown_feature_penalty_is_refused(tmp_path):
    adapter = _mp2_adapter(tmp_path)
    wrapper = CEIGraphXAIWrapper(adapter, method='cei_gnn_mp2')
    batch = _batch()
    x = wrapper.set_context(batch)
    with pytest.raises(ValueError, match='feature_penalty'):
        gx.explain_algorithms(wrapper, x, batch.edge_index,
                              batch=torch.zeros(x.size(0), dtype=torch.long), steps=2, epochs=2,
                              feature_penalty='max')
