"""CEI-GNN v3 under the clinical_graph_v2 explanation runner (the bridge), the shared GraphXAI
and IG code it runs through, plus the dev-fold and CLI-default changes the bridge needed.

The model under test is the real EvidenceAdapterV3 on the synthetic graph of
tests/test_cei_gnn_v3_core.py, not a stand-in.
"""
import copy
import inspect
import json
import sys

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing

from core.explain import graphxai_standardized as gx
from core.explain import ig_clinical, runner
from core.explain.cei_bridge import CEIGraphXAIWrapper
from tests.test_cei_gnn_v3_core import (
    CLASSES, EDGE_DIM, HIDDEN, LAYOUT, NODE_DIM, NUM_TOKENS, RELATIONS, TOKEN_DIM, TRIPLES,
    _adapter, _graph,
)
from tests.test_explain_runner import train_one

WIDTH = NODE_DIM + TOKEN_DIM


@pytest.fixture
def bridge(tmp_path):
    torch.manual_seed(5)
    adapter = _adapter(tmp_path).eval()
    wrapper = CEIGraphXAIWrapper(adapter, method="cei_gnn_v3")
    batch = Batch.from_data_list([_graph()])
    return adapter, wrapper, batch, wrapper.set_context(batch)


def _full(adapter, batch, x):
    """[x | the batch's node-type vectors]: the network's own continuous row."""
    with torch.no_grad():
        return torch.cat((x, adapter.continuous_inputs(batch)[:, WIDTH:]), dim=-1)


# ------------------------------------------------------------------ wrapper protocol
def test_wrapper_reproduces_the_adapter_on_numeric_plus_token_vectors(bridge):
    adapter, wrapper, batch, x = bridge
    ok, diff = wrapper.verify(x, batch.edge_index)
    assert ok and diff < 1e-6
    assert x.shape == (8, WIDTH)                                # numeric | token vectors
    with torch.no_grad():
        token_vectors = adapter.network.token_embedding(batch.token)
    assert torch.equal(x[:, NODE_DIM:], token_vectors)
    assert torch.equal(x[:, :NODE_DIM], batch.x)


def test_the_protocol_attributes(bridge):
    adapter, wrapper, batch, _ = bridge
    assert wrapper.builtin_class_specific is True
    assert wrapper.inert_edge_mask() is None
    assert wrapper.method == "cei_gnn_v3"
    assert wrapper.ig_feature_spec() == {"token_columns": [NODE_DIM, WIDTH]}
    with torch.no_grad():
        assert torch.equal(wrapper.deletion_forward(_graph()), adapter(batch, epoch=0).logits)


def test_forward_has_no_token_override_and_refuses_unknown_inputs(bridge):
    _, wrapper, batch, x = bridge
    assert "token_override" not in inspect.signature(wrapper.forward).parameters
    with pytest.raises(TypeError, match="token_override"):
        wrapper(x, batch.edge_index, token_override=batch.token)
    with pytest.raises(NotImplementedError, match="edge-attribute override"):
        wrapper(x, batch.edge_index, edge_attr_override=torch.zeros(6, EDGE_DIM))
    with pytest.raises(ValueError, match="token vector"):
        wrapper(_full(wrapper.adapter, batch, x), batch.edge_index)      # the old composite row


def test_forward_declares_no_edge_mask_so_the_explainer_has_one_delivery_path(bridge):
    _, wrapper, _, _ = bridge
    assert "edge_mask" not in inspect.signature(wrapper.forward).parameters
    assert gx._model_declares_edge_mask(wrapper) is False


@pytest.mark.parametrize("value", [0.5, 0.0])
def test_a_removal_mask_matches_the_networks_own_public_hook_exactly_once(bridge, value):
    """Independent reference: the network's public set_edge_mask. If our PyG-hook delivery
    applied the mask twice, 0.5 would match a 0.25 reference instead."""
    adapter, wrapper, batch, x = bridge
    edges = batch.edge_index
    mask = torch.full((edges.size(1),), value)
    got = wrapper(x, edges, edge_mask=mask).detach()

    reference = copy.deepcopy(adapter)
    reference.network.set_edge_mask(mask)
    expected = reference.forward_continuous(_full(adapter, batch, x), edges, batch).detach()
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    assert not torch.allclose(got, wrapper(x, edges).detach(), atol=1e-6)    # the mask does something


def test_the_builtin_is_the_exact_additive_accounting(bridge):
    adapter, wrapper, batch, x = bridge
    importance = wrapper.builtin_node_importance(x, batch.edge_index)
    detail = wrapper.builtin_detail()
    prediction = int(wrapper(x, batch.edge_index).argmax(-1))
    assert isinstance(importance, list) and len(importance) == 8 and min(importance) >= 0
    assert detail["explained_class"] == prediction
    assert detail["accounting_logit_abs_diff"] < 1e-4          # contributions sum to the logit
    for key in ("structural_abs_share", "negative_eligible_count", "negative_eligible_share"):
        assert key in detail
    other = (prediction + 1) % CLASSES
    redirected = wrapper.builtin_node_importance(x, batch.edge_index, target_class=other)
    assert wrapper.builtin_detail()["explained_class"] == other
    assert importance != redirected
    assert wrapper.builtin_detail(target_class=prediction)["explained_class"] == prediction


def test_gnnexplainer_reaches_cei_and_leaves_no_mask_state_behind(bridge):
    _, wrapper, batch, x = bridge
    result = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=3)
    assert all(result[a]["status"] == "success" for a in gx.ALGORITHMS), result
    provenance = result["GNNExplainer"]["provenance"]
    assert provenance["edge_gradient_verified"] is True and provenance["edge_gradient_l1"] > 0
    assert provenance["disclosure"] == gx.GNNEXPLAINER_DISCLOSURE["cei_gnn_v3"]
    assert provenance["soft_mask"] == gx.SOFT_MASK_NOTE
    assert not wrapper.inner.network.edge_aggregator.explain
    final = ("gsat", "protgnn", "graphcare", "cei_gnn_mp2")
    assert all(gx.GNNEXPLAINER_DISCLOSURE.get(method) for method in final)
    assert "invisible" in gx.GNNEXPLAINER_DISCLOSURE["protgnn"]           # weighted mean
    assert "inert_edges" in gx.GNNEXPLAINER_DISCLOSURE["graphcare"]
    assert "pair" in gx.GNNEXPLAINER_DISCLOSURE["cei_gnn_mp2"]
    explicit = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                     batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=2,
                                     method="cei_gnn_mp2")
    assert explicit["GNNExplainer"]["provenance"]["disclosure"] == \
        gx.GNNEXPLAINER_DISCLOSURE["cei_gnn_mp2"]


def test_ig_baseline_keeps_the_kind_and_zeroes_value_and_token_vectors(bridge):
    _, wrapper, batch, x = bridge
    spec = wrapper.ig_feature_spec()
    baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT, **spec)
    # the kind columns are kept (tests/test_graphxai_phase_b.py, real layout names); this
    # fixture's layout has none, so the whole numeric block is "not recorded"
    assert torch.count_nonzero(baseline[:, :NODE_DIM]) == 0
    assert torch.count_nonzero(baseline[:, NODE_DIM:]) == 0              # the pad embedding
    with_kinds = ig_clinical.not_recorded_baseline(
        x, batch.node_type, ["complaint"] + LAYOUT[1:], **spec)
    assert torch.equal(with_kinds[:, 0], x[:, 0]) and torch.count_nonzero(with_kinds[:, 1:]) == 0
    with pytest.raises(ValueError, match="columns"):
        ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT)   # token block not declared
    with pytest.raises(ValueError, match="columns"):
        ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT, token_columns=[NODE_DIM, 3])


def test_build_model_rebuilds_cei_from_its_recorded_options(bridge):
    adapter, _, batch, _ = bridge
    model, is_adapter = runner.build_model(_binding(adapter), adapter.state_dict())
    assert is_adapter is True
    with torch.no_grad():
        assert torch.equal(model.eval()(batch, epoch=0).logits, adapter(batch, epoch=0).logits)


def test_build_model_refuses_a_cei_whose_recorded_configuration_does_not_match(bridge):
    adapter, *_ = bridge
    tampered = adapter.run_config()
    tampered["parameter_count"] += 1
    with pytest.raises(ValueError, match="configuration differs"):
        runner.build_model(_binding(adapter, method_config=tampered), adapter.state_dict())


def _binding(adapter, **override):
    return {"method": "cei_gnn_v3", "conv": None, "node_dim": NODE_DIM, "edge_dim": EDGE_DIM,
            "num_classes": CLASSES, "vocabulary_size": NUM_TOKENS, "num_relations": RELATIONS,
            "num_meta_relations": TRIPLES, "hidden": HIDDEN, "layers": 1,
            "edge_direction": "forward", "seed": 1234,
            "method_config": adapter.run_config()} | override


# ------------------------------------------------------------------ shared explainer code
class _EdgeSum(torch.nn.Module):
    """Toy graph model with an explicit edge_mask keyword: logit c = sum over read edges of
    mask * w_c . (x[src] + x[dst]); edges in `inert` are never read (like GraphCare's edges to a
    global node)."""

    def __init__(self, width, inert=()):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.weight = torch.nn.Parameter(torch.randn(2, width, generator=generator))
        self.head = torch.nn.Identity()        # the vendored base reads a second module
        self.inert = set(inert)

    def forward(self, x, edge_index, batch=None, edge_mask=None):
        mask = torch.ones(edge_index.size(1)) if edge_mask is None else edge_mask
        read = torch.tensor([e not in self.inert for e in range(edge_index.size(1))],
                            dtype=x.dtype)
        messages = (x[edge_index[0]] + x[edge_index[1]]) @ self.weight.t()
        return self.head(((mask * read)[:, None] * messages).sum(0, keepdim=True) + 0.1 * x.sum())


def _toy_graph():
    x = torch.tensor([[1.0, 0.5, -0.2], [0.3, -1.0, 0.8], [0.7, 0.2, 0.1], [-0.4, 0.9, 0.6]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    return x, edge_index, torch.zeros(4, dtype=torch.long)


def test_explain_algorithms_leaves_the_global_rng_untouched(bridge):
    _, wrapper, batch, x = bridge
    for seed in (3, None):
        before = torch.random.get_rng_state()
        gx.explain_algorithms(wrapper, x, batch.edge_index, batch=torch.zeros(8, dtype=torch.long),
                              steps=2, epochs=2, seed=seed)
        assert torch.equal(torch.random.get_rng_state(), before)


def test_a_seeded_gnnexplainer_is_reproducible(bridge):
    _, wrapper, batch, x = bridge
    runs = [gx.explain_algorithms(wrapper, x, batch.edge_index,
                                  batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=3,
                                  seed=7)["GNNExplainer"]["node_explanation"]["node_importance"]
            for _ in range(2)]
    assert runs[0] == runs[1]


def test_the_feature_mask_penalty_is_the_mean_over_columns(bridge):
    _, wrapper, batch, x = bridge
    explainer = gx.CompatibleGNNExplainer(wrapper, feature_width=x.size(1))
    assert explainer.coeff["feature"]["size"] == pytest.approx(1.0 / x.size(1))
    assert explainer.coeff["feature"]["entropy"] == 0.1
    assert explainer.coeff["edge"] == {"entropy": 1.0, "size": 0.005}       # vendored defaults
    mask = torch.randn(x.size(1)).sigmoid()
    penalty = explainer.coeff["feature"]["size"] * torch.sum(mask)          # the vendored term
    assert torch.allclose(penalty, mask.mean())
    result = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=2)
    assert result["GNNExplainer"]["provenance"]["feature_penalty"] == "mean"


def test_inert_edges_are_skipped_by_the_node_reduction():
    x, edge_index, batch = _toy_graph()
    model = _EdgeSum(3, inert=(0,))
    inert = torch.tensor([True, False, False])
    out = gx.explain_algorithms(model, x, edge_index, batch=batch, steps=2, epochs=5,
                                node_reduction="mean", seed=1, inert_edge_mask=inert)
    gnn = out["GNNExplainer"]
    assert gnn["status"] == "success", gnn
    importance = gnn["node_explanation"]["node_importance"]
    assert importance[0] == 0.0                        # node 0 touches only the inert edge
    assert gnn["provenance"]["inert_edges"]["count"] == 1
    assert gnn["provenance"]["inert_edges"]["indices"] == [0]
    plain = gx.explain_algorithms(model, x, edge_index, batch=batch, steps=2, epochs=5,
                                  node_reduction="mean", seed=1)["GNNExplainer"]
    assert plain["node_explanation"]["node_importance"][0] > 0.0
    assert plain["provenance"]["inert_edges"]["count"] == 0
    # the same seeded mask: nodes away from the inert edge keep their importance
    assert importance[2:] == plain["node_explanation"]["node_importance"][2:]
    assert importance[1] != plain["node_explanation"]["node_importance"][1]
    with pytest.raises(ValueError, match="one entry per edge"):
        gx.explain_algorithms(model, x, edge_index, batch=batch, steps=2, epochs=2,
                              inert_edge_mask=torch.tensor([True]))


def test_gradient_provenance_names_the_reduction_and_the_columns(bridge):
    _, wrapper, batch, x = bridge
    result = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=2)
    for name in ("GradExplainer", "IntegratedGradExplainer"):
        provenance = result[name]["provenance"]
        assert provenance["node_reduction"] == "sum_abs_over_columns"
        assert provenance["explained_columns"] == {"numeric": [0, NODE_DIM],
                                                   "token_vector": [NODE_DIM, WIDTH]}
    assert result["IntegratedGradExplainer"]["provenance"]["baseline"] == "all_zero"
    toy = gx.explain_algorithms(_EdgeSum(3), *_toy_graph()[:2], batch=_toy_graph()[2], steps=2,
                                epochs=2)
    assert toy["GradExplainer"]["provenance"]["explained_columns"] == {"numeric": [0, 3]}


def test_new_node_explanations_carry_no_v1_fidelity(bridge):
    _, wrapper, batch, x = bridge
    result = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=2)
    for name in gx.ALGORITHMS:
        explanation = result[name]["node_explanation"]
        assert "fidelity_plus" not in explanation and "fidelity_minus" not in explanation
    legacy = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=2,
                                   v1_fidelity=True)
    assert "fidelity_plus" in legacy["GradExplainer"]["node_explanation"]


def test_gnnexplainer_runs_with_the_explained_features_device(monkeypatch):
    """The vendored module moves x to a module-level device ('cuda' when available); a model on
    the CPU must still be explained."""
    module = sys.modules[gx.GNNExplainer.__module__]
    monkeypatch.setattr(module, "device", "meta")
    x, edge_index, batch = _toy_graph()
    out = gx.explain_algorithms(_EdgeSum(3), x, edge_index, batch=batch, steps=2, epochs=2)
    assert out["GNNExplainer"]["status"] == "success", out["GNNExplainer"]
    assert module.device == "meta"                    # restored after the call


class _Propagate(MessagePassing):
    """A real subclass: instantiating PyG's base class itself rewrites its class-level
    propagate and breaks every later MessagePassing model in the session."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


class _BothPaths(_EdgeSum):
    def __init__(self):
        super().__init__(3)
        self.conv = _Propagate()


def test_a_model_with_both_mask_paths_is_refused():
    x, edge_index, batch = _toy_graph()
    out = gx.explain_algorithms(_BothPaths(), x, edge_index, batch=batch, steps=2, epochs=2)
    assert out["GNNExplainer"]["status"] == "failed"
    assert "applied twice" in out["GNNExplainer"]["error"]
    assert out["GradExplainer"]["status"] == "success"


class _Curved(torch.nn.Module):
    """logit 0 = sum(x ** 8): the trapezoid rule needs more than 8 steps on it."""

    def forward(self, x, edge_index, **kwargs):
        return torch.stack([(x ** 8).sum(), x.sum()]).view(1, 2)


def test_ig_steps_escalate_until_completeness():
    x = torch.tensor([[1.5, -1.2], [0.8, 2.0]])
    baseline, edges = torch.zeros_like(x), torch.empty((2, 0), dtype=torch.long)
    _, at_8 = ig_clinical.integrated_gradients(_Curved(), x, edges, baseline, 0, 8)
    assert at_8["completeness_error_rel"] > ig_clinical.DEFAULT_TOLERANCE
    attribution, info = ig_clinical.integrated_gradients_until_complete(
        _Curved(), x, edges, baseline, 0)
    assert info["completeness_met"] is True and info["steps_used"] > 8
    assert info["tried"][0] == {"steps": 8,
                                "completeness_error_rel": at_8["completeness_error_rel"]}
    assert [t["steps"] for t in info["tried"]] == [s for s in ig_clinical.DEFAULT_STEP_CANDIDATES
                                                   if s <= info["steps_used"]]
    assert info["tried"][-1]["completeness_error_rel"] == info["completeness_error_rel"]
    direct, _ = ig_clinical.integrated_gradients(_Curved(), x, edges, baseline, 0,
                                                 info["steps_used"])
    assert torch.equal(attribution, direct)
    started, later = ig_clinical.integrated_gradients_until_complete(
        _Curved(), x, edges, baseline, 0, start_steps=info["steps_used"])
    assert later["tried"] == [{"steps": info["steps_used"],
                               "completeness_error_rel": info["completeness_error_rel"]}]


def test_ig_escalation_reports_an_unmet_tolerance_and_refuses_an_unknown_start():
    x = torch.tensor([[1.5, -1.2]])
    baseline, edges = torch.zeros_like(x), torch.empty((2, 0), dtype=torch.long)
    _, info = ig_clinical.integrated_gradients_until_complete(
        _Curved(), x, edges, baseline, 0, candidates=(8, 16), tolerance=1e-12)
    assert info["completeness_met"] is False and info["steps_used"] == 16
    assert len(info["tried"]) == 2
    with pytest.raises(ValueError, match="start_steps"):
        ig_clinical.integrated_gradients_until_complete(_Curved(), x, edges, baseline, 0,
                                                        start_steps=10)


def test_node_importance_sums_absolute_attribution_over_every_column():
    attribution = torch.tensor([[1.0, -2.0, 0.5, -0.5], [0.0, 0.0, -3.0, 0.0]])
    assert torch.equal(ig_clinical.node_importance(attribution), torch.tensor([4.0, 3.0]))
    assert ig_clinical.NODE_REDUCTION == gx.GRADIENT_NODE_REDUCTION


def test_real_patient_references_keep_the_token_block_of_the_explained_row():
    import numpy as np
    import types

    graph = types.SimpleNamespace(x=torch.tensor([[1.0, 1.0]]), token=torch.tensor([5]),
                                  node_type=torch.tensor([5]))
    bank = ig_clinical.ReferenceBank.from_graphs([graph])
    x = torch.tensor([[7.0, 7.0, 0.3, 0.4], [8.0, 8.0, 0.5, 0.6]])
    fallback = torch.zeros_like(x)
    drawn = bank.draw(torch.tensor([5, 9]), torch.tensor([5, 5]), fallback,
                      np.random.default_rng(0), fill=x)
    assert drawn[0].tolist() == pytest.approx([1.0, 1.0, 0.3, 0.4])   # real row, own token vector
    assert drawn[1].tolist() == [0.0, 0.0, 0.0, 0.0]          # unknown key: fallback
    with pytest.raises(ValueError, match="fill"):
        bank.draw(torch.tensor([5]), torch.tensor([5]), fallback[:1], np.random.default_rng(0))


# ------------------------------------------------------------------ fold + CLI defaults
DEV_RUN = {"selection_fold": "dev", "dev_limit": 3}    # a run that selects on its dev fold


def test_a_run_that_keeps_validation_closed_is_explained_on_its_dev_fold(tmp_path):
    run_dir = train_one(tmp_path, "run_dev_only", method="gsat", conv=None, final_eval="none",
                        **DEV_RUN)
    with pytest.raises(ValueError, match="selected_validation"):
        runner.run_explanations(run_dir, tmp_path / "ex_val", fold="validation", max_subjects=1,
                                steps=2, epochs=3)
    out = tmp_path / "ex_dev"
    manifest = runner.run_explanations(run_dir, out, fold="dev", max_subjects=2, steps=2, epochs=3,
                                       ig_step_candidates=(4,), random_repeats=0)
    assert manifest["replay_verified"] and manifest["fold"] == "dev" and manifest["failed_count"] == 0
    cohort = json.loads((out / "cohort.json").read_text())
    assert cohort["fold"] == "dev"
    assert cohort["fold_sample_ids_sha256"] == manifest["fold_sample_ids_sha256"]


def test_the_dev_fold_replay_really_checks_the_dev_hash(tmp_path):
    run_dir = train_one(tmp_path, "run_dev_tamper", method="gsat", conv=None,
                        final_eval="validation", **DEV_RUN)
    binding_path = run_dir / "binding.json"
    binding = json.loads(binding_path.read_text())
    binding["selected_dev"]["prediction_sha256"] = "0" * 64
    binding_path.write_text(json.dumps(binding))
    # Without stored dev predictions the tolerance fallback has nothing to compare with.
    (run_dir / "dev.npz").unlink(missing_ok=True)
    with pytest.raises(ValueError, match="REPLAY CHECK FAILED"):
        runner.run_explanations(run_dir, tmp_path / "ex_tamper", fold="dev", max_subjects=1,
                                steps=2, epochs=3)


def test_cli_explains_500_patients_by_default_and_zero_means_everyone(monkeypatch, tmp_path):
    seen = {}

    def fake(run_dir, output, **kwargs):
        seen.update(kwargs)
        return {"method": "m", "conv": None, "explained_count": 0, "failed_count": 0}

    monkeypatch.setattr(runner, "run_explanations", fake)
    base = ["runner", "--run-dir", str(tmp_path), "--output", str(tmp_path / "o")]
    monkeypatch.setattr(sys, "argv", base)
    runner._cli()
    assert seen["sample_patients"] == runner.DEFAULT_EXPLAINED_PATIENTS == 500
    monkeypatch.setattr(sys, "argv", base + ["--sample-patients", "0", "--fold", "dev"])
    runner._cli()
    assert seen["sample_patients"] is None and seen["fold"] == "dev"
