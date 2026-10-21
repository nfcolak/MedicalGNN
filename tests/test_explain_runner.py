"""End-to-end tests for the item-#1 explanation runner (explain.py).

Unlike every other test file touched this session, this one does NOT stop at
synthetic in-memory tensors: it actually invokes train.run() on tiny synthetic
data, through the real CLI-equivalent pipeline, producing a real binding.json/
preprocessing.json/best.pt/validation.npz on disk -- then points the runner at
that real output directory. This is the only way to prove the reload/replay
logic (which exists specifically to catch a mismatched reconstruction) is
actually correct, not just plausible.

The default fold is the test fold (the report fold, ADR-023); validation and dev
are decision folds. The synthetic runs default to the final protocol's shape
(selection on validation, test scored once).
"""
import copy
import inspect
import json
import sys
import types

import pytest

from core import train
from core.explain import runner as explain
from tests.graphxai_fixtures import artifact, run_args


def train_one(tmp_path, out_name, **overrides):
    root, targets, canonical = artifact(tmp_path, count=24, test_rows=4)
    out = tmp_path / out_name
    args = run_args(root, targets, canonical, out, **overrides)
    train.run(args)
    return out


def test_black_box_runner_replays_and_explains_the_test_fold(tmp_path):
    run_dir = train_one(tmp_path, "run_black_box", method="clinical_gnn", conv="edge_conditioned")
    output = tmp_path / "explanations_black_box"
    manifest = explain.run_explanations(
        run_dir, output, max_subjects=2, steps=2, epochs=3
    )
    assert manifest["method"] == "clinical_gnn"
    assert manifest["conv"] == "edge_conditioned"
    assert manifest["replay_verified"] is True
    assert manifest["explained_count"] >= 1
    assert manifest["failed_count"] == 0, manifest["failures"]

    for filename in manifest["record_files"]:
        record = json.loads((output / filename).read_text())
        assert record["schema"] == "medgnn.clinical_graph_v2_explanation_record"
        assert record["method"] == "clinical_gnn"
        assert record["conv"] == "edge_conditioned"
        for algo in ("GradExplainer", "IntegratedGradExplainer", "GNNExplainer"):
            assert record["graphxai"][algo]["status"] == "success", record["graphxai"][algo]
        # The black-box baseline has no .explain(): builtin readout is correctly absent,
        # not fabricated.
        assert record["builtin_available"] is False
        assert record["builtin_node_importance"] is None


def test_gsat_runner_replays_and_explains(tmp_path):
    run_dir = train_one(tmp_path, "run_gsat", method="gsat", conv=None)
    output = tmp_path / "explanations_gsat"
    manifest = explain.run_explanations(
        run_dir, output, max_subjects=2, steps=2, epochs=3
    )
    assert manifest["method"] == "gsat"
    assert manifest["replay_verified"] is True
    assert manifest["explained_count"] >= 1
    assert manifest["failed_count"] == 0, manifest["failures"]

    for filename in manifest["record_files"]:
        record = json.loads((output / filename).read_text())
        assert record["method"] == "gsat"
        for algo in ("GradExplainer", "IntegratedGradExplainer", "GNNExplainer"):
            assert record["graphxai"][algo]["status"] == "success", record["graphxai"][algo]
        # GSAT has .explain(): its own deterministic attention should be present.
        assert record["builtin_available"] is True
        assert len(record["builtin_node_importance"]) > 0


def test_graphcare_recency_decay_is_recovered_not_defaulted(tmp_path):
    """The one confirmed case where a non-shape hyperparameter is baked
    directly into the forward pass (not just the training loss). If this
    runner silently defaulted recency_decay instead of reading it back from
    method_config.effective_settings, the replay check below would (correctly)
    fail -- this test is the proof it doesn't."""
    run_dir = train_one(tmp_path, "run_graphcare", method="graphcare", conv=None)
    output = tmp_path / "explanations_graphcare"
    manifest = explain.run_explanations(
        run_dir, output, max_subjects=2, steps=2, epochs=3
    )
    assert manifest["method"] == "graphcare"
    assert manifest["replay_verified"] is True
    assert manifest["failed_count"] == 0, manifest["failures"]


def test_replay_check_refuses_a_tampered_prediction_hash(tmp_path):
    """If binding.json's recorded hash doesn't match what reloading + replaying
    best.pt actually produces, and the stored predictions of the fold do not match
    either, the runner must refuse outright, before writing any output -- never
    silently explain a model it can't verify."""
    import numpy as np

    run_dir = train_one(tmp_path, "run_tamper", method="gsat", conv=None)
    binding_path = run_dir / "binding.json"
    binding = json.loads(binding_path.read_text())
    tampered = copy.deepcopy(binding)
    tampered["selected_test"]["prediction_sha256"] = "0" * 64
    binding_path.write_text(json.dumps(tampered))
    stored = dict(np.load(run_dir / "test.npz"))
    stored["proba"] = stored["proba"][:, ::-1].copy()   # a different model's answers
    np.savez(run_dir / "test.npz", **stored)

    output = tmp_path / "explanations_tampered"
    with pytest.raises(ValueError, match="REPLAY CHECK FAILED"):
        explain.run_explanations(run_dir, output, max_subjects=2, steps=2, epochs=3)
    assert not output.exists() or not any(output.iterdir())


def test_replay_falls_back_to_the_stored_predictions_when_the_hash_cannot_reproduce(tmp_path):
    """On another machine the byte hash differs in the last float bits; the replay then
    passes only if every row matches the stored <fold>.npz within the tolerance with
    100% argmax agreement, and the manifest says which rule passed."""
    run_dir = train_one(tmp_path, "run_fallback", method="gsat", conv=None)
    binding_path = run_dir / "binding.json"
    binding = json.loads(binding_path.read_text())
    binding["selected_test"]["prediction_sha256"] = "0" * 64
    binding_path.write_text(json.dumps(binding))

    manifest = explain.run_explanations(run_dir, tmp_path / "out_fallback", max_subjects=1,
                                        steps=2, epochs=3)
    assert manifest["replay"]["mode"] == "stored_predictions_tolerance"
    assert manifest["replay"]["max_abs_diff"] <= explain.REPLAY_TOLERANCE
    assert manifest["replay"]["argmax_agreement"] == 1.0


def test_replay_reports_the_exact_hash_mode_when_it_reproduces(tmp_path):
    run_dir = train_one(tmp_path, "run_exact", method="gsat", conv=None)
    manifest = explain.run_explanations(run_dir, tmp_path / "out_exact", max_subjects=1,
                                        steps=2, epochs=3)
    assert manifest["replay"]["mode"] == "exact_hash"


def test_refuses_to_write_into_an_occupied_output_directory(tmp_path):
    run_dir = train_one(tmp_path, "run_occupied", method="gsat", conv=None)
    output = tmp_path / "explanations_occupied"
    output.mkdir()
    (output / "leftover.txt").write_text("from a previous run")
    with pytest.raises(FileExistsError):
        explain.run_explanations(run_dir, output, max_subjects=1, steps=2, epochs=3)


def test_replay_check_uses_the_trained_batch_size_not_a_hardcoded_default(tmp_path):
    """Regression for a real bug this session found running the real pipeline
    end to end: replay_check() used to default to a hardcoded batch_size
    (64), independent of what training actually used. On a small fold that
    difference was invisible (everything fit in one batch either way) -- it
    only surfaced on a larger real run where batch count actually differed
    between the two sizes, and GSAT's InstanceNorm layers turned out to be
    sensitive to how graphs are grouped into a batch, not just to each
    graph's own content. Force a tiny training batch_size (3) against the
    existing 4-graph validation (decision) fold, which already spans
    multiple batches at size 3 but not at the old hardcoded 64 -- proving
    the fix reads binding['batch_size'] rather than reintroducing the
    mismatch.
    """
    run_dir = train_one(tmp_path, "run_batchsize", method="gsat", conv=None, batch_size=3)
    binding = json.loads((run_dir / "binding.json").read_text())
    assert binding["batch_size"] == 3
    output = tmp_path / "explanations_batchsize"
    manifest = explain.run_explanations(run_dir, output, fold="validation",
                                        max_subjects=4, steps=2, epochs=3)
    assert manifest["replay_verified"] is True
    assert manifest["failed_count"] == 0, manifest["failures"]


def test_default_fold_is_the_test_fold_and_is_labelled_report(tmp_path):
    assert inspect.signature(explain.run_explanations).parameters["fold"].default == "test"
    assert inspect.signature(explain.load_run_bundle).parameters["fold"].default == "test"
    assert inspect.signature(explain.rebuild_split).parameters["fold"].default == "test"
    run_dir = train_one(tmp_path, "run_default_fold", method="gsat", conv=None)
    output = tmp_path / "explanations_default_fold"
    manifest = explain.run_explanations(run_dir, output, max_subjects=2, steps=2, epochs=3)
    binding = json.loads((run_dir / "binding.json").read_text())
    assert manifest["fold"] == "test" and manifest["fold_role"] == "report"
    assert manifest["fold_sample_ids_sha256"] == binding["split_sample_ids_sha256"]["test"]
    assert manifest["replay_verified"] is True and manifest["failed_count"] == 0
    cohort = json.loads((output / "cohort.json").read_text())
    assert cohort["fold"] == "test" and cohort["fold_sample_count"] == 4
    written = json.loads((output / "binding.json").read_text())
    assert written["cohort"]["fold"] == "test" and written["cohort"]["fold_role"] == "report"


def test_validation_fold_is_labelled_decision(tmp_path):
    run_dir = train_one(tmp_path, "run_decision", method="gsat", conv=None)
    output = tmp_path / "explanations_decision"
    manifest = explain.run_explanations(run_dir, output, fold="validation",
                                        max_subjects=2, steps=2, epochs=3)
    assert manifest["fold"] == "validation" and manifest["fold_role"] == "decision"
    assert manifest["replay_verified"] is True
    assert json.loads((output / "binding.json").read_text())["cohort"]["fold_role"] == "decision"


def test_dev_fold_run_replays_and_is_labelled_decision(tmp_path):
    run_dir = train_one(tmp_path, "run_dev", method="gsat", conv=None, selection_fold="dev",
                        dev_limit=3, final_eval="validation")
    output = tmp_path / "explanations_dev"
    manifest = explain.run_explanations(run_dir, output, fold="dev",
                                        max_subjects=2, steps=2, epochs=3)
    assert manifest["fold"] == "dev" and manifest["fold_role"] == "decision"
    assert manifest["replay_verified"] is True


def test_fold_role_values():
    assert explain.FOLD_ROLE == {"test": "report", "validation": "decision", "dev": "decision"}
    assert set(explain.FOLD_ROLE) == set(explain.FOLD_BLOCK)
    assert explain.FOLD_BLOCK["test"] == "selected_test"


def test_test_rows_are_decoded_only_for_the_test_fold(tmp_path):
    run_dir = train_one(tmp_path, "run_load_test", method="gsat", conv=None)
    binding = json.loads((run_dir / "binding.json").read_text())
    assert "test" in explain.rebuild_split(binding, "test")
    assert "test" not in explain.rebuild_split(binding, "validation")
    assert "validation" in explain.rebuild_validation_split(binding)   # the earlier name keeps its default
    assert "test" not in explain.rebuild_validation_split(binding)


def test_a_run_without_a_test_block_is_refused_for_the_test_fold(tmp_path):
    run_dir = train_one(tmp_path, "run_no_test", method="gsat", conv=None, selection_fold="dev",
                        dev_limit=3, final_eval="validation")
    with pytest.raises(ValueError, match="selected_test"):
        explain.load_run_bundle(run_dir, "test")
    with pytest.raises(ValueError, match="fold must be one of"):
        explain.load_run_bundle(run_dir, "holdout")


def test_gchm_runs_are_refused(tmp_path):
    run_dir = train_one(tmp_path, "run_gchm_refused", method="gsat", conv=None)
    binding_path = run_dir / "binding.json"
    binding = json.loads(binding_path.read_text())
    binding.update(method="clinical_gnn", conv="gchm")
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="GCHM-PNA is not part of the explanation study"):
        explain.load_run_bundle(run_dir)
    with pytest.raises(ValueError, match="GCHM-PNA is not part of the explanation study"):
        explain.build_model(binding, {})
    assert "gchm" not in explain.BLACK_BOX_CONVS


def test_cei_mp2_is_registered_and_dispatched_through_the_bridge(monkeypatch):
    """The runner imports the bridge lazily, so a stub proves the dispatch (the bridge's own
    behaviour is covered in tests/test_cei_bridge.py)."""
    calls = {}
    bridge = types.ModuleType("core.explain.cei_bridge")

    def build_cei_model(binding, state_dict):
        calls["build"] = (binding["method"], state_dict)
        return "adapter"

    class CEIGraphXAIWrapper:
        def __init__(self, model, *, method):
            calls["wrap"] = (model, method)

    bridge.build_cei_model, bridge.CEIGraphXAIWrapper = build_cei_model, CEIGraphXAIWrapper
    monkeypatch.setitem(sys.modules, "core.explain.cei_bridge", bridge)

    assert "cei_gnn_mp2" in explain.GRAPHXAI_METHODS
    binding = {"method": "cei_gnn_mp2", "conv": None, "node_dim": 1, "edge_dim": 1,
               "num_classes": 2, "vocabulary_size": 3, "num_relations": 1,
               "num_meta_relations": 1, "hidden": 4, "layers": 1}
    model, is_adapter = explain.build_model(binding, {"w": 1})
    assert (model, is_adapter) == ("adapter", True)
    assert calls["build"] == ("cei_gnn_mp2", {"w": 1})
    explain.build_wrapper(binding, model)
    assert calls["wrap"] == ("adapter", "cei_gnn_mp2")
    assert not hasattr(explain, "_build_cei")
