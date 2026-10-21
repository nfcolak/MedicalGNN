"""End-to-end tests for the explanation runner (core/explain/runner.py).

Unlike most explanation tests, this file does NOT stop at synthetic in-memory tensors: it invokes
train.run() on tiny synthetic data, through the real CLI-equivalent pipeline, producing a real
binding.json/preprocessing.json/best.pt/<fold>.npz on disk -- then points the runner at that real
output directory. This is the only way to prove the reload, split and replay guards (which exist
to catch a mismatched reconstruction) refuse what they must and accept what they must.

The default fold is the test fold (the report fold); validation and dev are decision folds. The
synthetic runs default to the final protocol's shape (selection on validation, test scored once).
The last section explains one visit of a synthetic graph record with each of the four final methods
(gsat, graphcare, protgnn, cei_gnn_mp2) through `explain_subject`.
"""
import copy
import inspect
import json
import sys
import types

import numpy as np
import pytest
import torch

from core import train
from core.explain import fidelity_v2 as fv2
from core.explain import runner as explain
from core.explain.node_deletion import ENCODED_FIELDS, DeletionScorer, encoding_differences
from core.schema import sha256
from tests.graphxai_fixtures import artifact, run_args


def train_one(tmp_path, out_name, **overrides):
    root, targets, canonical = artifact(tmp_path, count=24, test_rows=4)
    out = tmp_path / out_name
    args = run_args(root, targets, canonical, out, **overrides)
    train.run(args)
    return out


def subject_setup(run_dir, fold="test"):
    """(binding, splits, wrapper, records) for explaining visits of one fold directly."""
    bundle = explain.load_run_bundle(run_dir, fold)
    splits = explain.rebuild_split(bundle.binding, fold)
    model, _ = explain.build_model(bundle.binding, bundle.state_dict)
    wrapper = explain.build_wrapper(bundle.binding, model)
    records = splits.visit_records([d.sample_id for d in splits[fold]])
    return bundle.binding, splits, wrapper, records


def subject_kwargs(binding, records, **extra):
    values = dict(method=binding["method"], conv=binding.get("conv"), steps=2, epochs=3,
                  num_relation_columns=binding["num_relations"], records=records,
                  random_repeats=4, tie_draws=3, true_class_explanations=False)
    values.update(extra)
    return values


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
        # The black-box baseline has no built-in readout: absent with its reason, not fabricated.
        assert record["builtin_available"] is False
        assert record["builtin_status"] == "unavailable" and record["builtin_unavailable_reason"]
        assert record["builtin_node_importance"] is None
        assert "builtin" not in record["fidelity_v2"]


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
        assert record["builtin_status"] in ("available", "no_eligible_evidence")
        assert len(record["builtin_node_importance"]) > 0


def test_graphcare_recency_decay_is_recovered_not_defaulted(tmp_path):
    """The one confirmed case where a non-shape hyperparameter is baked
    directly into the forward pass (not just the training loss). GraphCare
    reports it as `recency_decay` but reads it as `graphcare_decay_rate` /
    `decay_rate`, so the run is trained with a NON-default decay: a runner that
    fell back to the native default would rebuild a different model."""
    run_dir = train_one(tmp_path, "run_graphcare", method="graphcare", conv=None,
                        graphcare_decay_rate=0.0116)
    bundle = explain.load_run_bundle(run_dir)
    assert bundle.binding["method_config"]["effective_settings"]["recency_decay"] == 0.0116
    model, _ = explain.build_model(bundle.binding, bundle.state_dict)
    assert model.recency_decay == 0.0116
    output = tmp_path / "explanations_graphcare"
    manifest = explain.run_explanations(
        run_dir, output, max_subjects=2, steps=2, epochs=3
    )
    assert manifest["method"] == "graphcare"
    assert manifest["replay_verified"] is True
    assert manifest["failed_count"] == 0, manifest["failures"]
    written = json.loads((output / "binding.json").read_text())
    assert written["model_run"]["effective_settings"]["recency_decay"] == 0.0116


def test_build_model_refuses_settings_that_do_not_round_trip(tmp_path):
    """A recorded setting the rebuilt model does not report back (a name the adapter
    does not read), a recorded schedule value or dropout it would not rebuild: every one refuses,
    never falls back to a default silently. The untouched binding rebuilds the whole run_config."""
    run_dir = train_one(tmp_path, "run_graphcare_tampered", method="graphcare", conv=None,
                        dropout=0.2)
    bundle = explain.load_run_bundle(run_dir)
    model, _ = explain.build_model(bundle.binding, bundle.state_dict)
    assert json.loads(json.dumps(model.run_config())) == bundle.binding["method_config"]
    binding = copy.deepcopy(bundle.binding)
    binding["method_config"]["effective_settings"]["unread_setting"] = 0.123
    with pytest.raises(ValueError, match=r"run_config differs .*effective_settings.unread_setting"):
        explain.build_model(binding, bundle.state_dict)
    binding = copy.deepcopy(bundle.binding)
    binding["method_config"]["native_schedule"]["patience"] += 1
    with pytest.raises(ValueError, match=r"native_schedule.patience"):
        explain.build_model(binding, bundle.state_dict)
    binding = copy.deepcopy(bundle.binding)
    binding["dropout"] = 0.0
    with pytest.raises(ValueError, match=r"architecture.dropout"):
        explain.build_model(binding, bundle.state_dict)


def test_protgnn_rebuild_keeps_select_after_projection(tmp_path):
    run_dir = train_one(tmp_path, "run_protgnn_select", method="protgnn", conv=None,
                        select_after_projection=True, warm_epochs=1, proj_epochs=1, epochs=3)
    bundle = explain.load_run_bundle(run_dir)
    assert bundle.binding.get("selection_start_epoch_index") is not None
    model, _ = explain.build_model(bundle.binding, bundle.state_dict)
    assert model.select_after_projection is True
    assert model.selection_start_epoch() == bundle.binding["selection_start_epoch_index"]


# ------------------------------------------------------------------ split guards
def test_rebuild_split_returns_the_fitted_preprocessing_and_the_record_encoder(tmp_path):
    from core.tensorize import preprocessing_state

    run_dir = train_one(tmp_path, "run_split", method="gsat", conv=None)
    bundle = explain.load_run_bundle(run_dir)
    splits = explain.rebuild_split(bundle.binding, "test")
    assert isinstance(splits, explain.RebuiltSplits) and "test" in splits
    state = json.dumps(preprocessing_state(splits.preprocessing), indent=2, sort_keys=True) + "\n"
    assert state == (run_dir / "preprocessing.json").read_text()
    records = splits.visit_records([d.sample_id for d in splits["test"]])
    for data in splits["test"]:
        graph, membership = records.records[data.sample_id]
        # every encoded field (x, edges, edge metadata, tokens, types, membership, visit count, global
        # mask), not a subset; the builder-record variant is below
        assert encoding_differences(splits.encode(graph, membership), data) == []


def _swap_two_test_labels(targets_path):
    lines = targets_path.read_text().splitlines()
    first, second = [i for i, line in enumerate(lines) if ",test," in line][:2]
    a, b = lines[first].split(","), lines[second].split(",")
    a[1], b[1] = b[1], a[1]
    assert a[1] != b[1]
    lines[first], lines[second] = ",".join(a), ",".join(b)
    targets_path.write_text("\n".join(lines) + "\n")


def test_a_relabelled_targets_file_is_refused(tmp_path):
    run_dir = train_one(tmp_path, "run_relabel", method="gsat", conv=None)
    binding = json.loads((run_dir / "binding.json").read_text())
    from pathlib import Path

    _swap_two_test_labels(Path(binding["targets_path"]))
    with pytest.raises(ValueError, match="targets file"):
        explain.rebuild_split(binding, "test")
    with pytest.raises(ValueError, match="targets file"):
        explain.run_explanations(run_dir, tmp_path / "out_relabel", max_subjects=1, steps=2, epochs=3)
    assert not (tmp_path / "out_relabel").exists()


def test_an_edited_graph_line_is_refused(tmp_path):
    from pathlib import Path

    run_dir = train_one(tmp_path, "run_graph_edit", method="gsat", conv=None)
    binding = json.loads((run_dir / "binding.json").read_text())
    graphs = Path(binding["artifact"]) / "graphs.jsonl"
    lines = graphs.read_text().splitlines()
    graph = json.loads(lines[-1])
    next(n for n in graph["nodes"] if n["kind"] == "vital")["value"] += 1.0
    lines[-1] = json.dumps(graph)
    graphs.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="graphs.jsonl"):
        explain.rebuild_split(binding, "test")


def test_a_changed_membership_sidecar_is_refused(tmp_path):
    from pathlib import Path

    from core.contracts import VISIT_MEMBERSHIP_FILENAME

    run_dir = train_one(tmp_path, "run_membership_edit", method="gsat", conv=None)
    binding = json.loads((run_dir / "binding.json").read_text())
    sidecar = Path(binding["artifact"]) / VISIT_MEMBERSHIP_FILENAME
    sidecar.write_text(sidecar.read_text() + "\n")
    with pytest.raises(ValueError, match=VISIT_MEMBERSHIP_FILENAME):
        explain.rebuild_split(binding, "test")


def test_a_preprocessing_that_does_not_refit_to_the_recorded_hash_is_refused(tmp_path):
    run_dir = train_one(tmp_path, "run_prep", method="gsat", conv=None)
    binding = json.loads((run_dir / "binding.json").read_text())
    drifted = dict(binding, token_min_count=binding["token_min_count"] + 1)
    with pytest.raises(ValueError, match="refitted preprocessing"):
        explain.rebuild_split(drifted, "test")


def test_kept_classes_and_missing_recorded_hashes_are_refused(tmp_path):
    run_dir = train_one(tmp_path, "run_kept", method="gsat", conv=None)
    binding = json.loads((run_dir / "binding.json").read_text())
    reordered = dict(binding, kept_label_indices=list(reversed(binding["kept_label_indices"])))
    with pytest.raises(ValueError, match="kept classes"):
        explain.rebuild_split(reordered, "test")
    for key in ("targets_sha256", "artifact_graphs_sha256", "artifact_visit_membership_sha256"):
        missing = {k: v for k, v in binding.items() if k != key}
        with pytest.raises(ValueError, match=f"records no {key}"):
            explain.rebuild_split(missing, "test")


def test_every_rebuilt_fold_is_checked_against_its_recorded_hash(tmp_path):
    """A split argument recorded differently from what training used (here the sample seed, which
    draws the bounded train fold) leaves the explained fold and the fixture's preprocessing unchanged,
    but the train fold the IG reference bank is drawn from changes: refused by its recorded hash."""
    run_dir = train_one(tmp_path, "run_fold_hashes", method="gsat", conv=None)
    binding = json.loads((run_dir / "binding.json").read_text())
    assert {"train", "validation", "test"} <= set(binding["split_sample_ids_sha256"])
    explain.rebuild_split(binding, "test")
    shifted = dict(binding, sample_seed=(binding.get("sample_seed") or binding["seed"]) + 99)
    with pytest.raises(ValueError, match=r"rebuilt train split's sample-ID hash .*\['train'\]"):
        explain.rebuild_split(shifted, "test")
    # a fold training did not record (an older binding) is not invented; the explained one must be
    partial = copy.deepcopy(binding)
    del partial["split_sample_ids_sha256"]["train"]
    explain.rebuild_split(partial, "test")
    del partial["split_sample_ids_sha256"]["test"]
    with pytest.raises(ValueError, match=r"split_sample_ids_sha256\['test'\]"):
        explain.rebuild_split(partial, "test")


def test_a_rewired_run_has_no_deletion_records(tmp_path):
    run_dir = train_one(tmp_path, "run_rewire_records", method="gsat", conv=None)
    bundle = explain.load_run_bundle(run_dir)
    splits = explain.rebuild_split(bundle.binding, "test")
    splits._rewired = ("comorbid_with",)
    with pytest.raises(ValueError, match="rewired"):
        splits.visit_records([splits["test"][0].sample_id])


# ------------------------------------------------------------------ replay guards
def test_replay_check_refuses_a_tampered_prediction_hash(tmp_path):
    """If binding.json's recorded hash doesn't match what reloading + replaying
    best.pt actually produces, and the stored predictions of the fold do not match
    either, the runner must refuse outright, before writing any output -- never
    silently explain a model it can't verify."""
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


def _off_by_float_noise(monkeypatch):
    """The replayed probabilities differ from training's in the last bits (another machine)."""
    real = explain.fold_probabilities
    monkeypatch.setattr(explain, "fold_probabilities",
                        lambda *a, **k: real(*a, **k) + np.float32(1e-7))


def test_replay_falls_back_to_the_stored_predictions_when_the_hash_cannot_reproduce(tmp_path, monkeypatch):
    """On another machine the byte hash differs in the last float bits; the replay then
    passes only if the stored <fold>.npz is the training run's own (hash and labels) and every row
    matches it within the tolerance with 100% argmax agreement; the manifest says which rule passed."""
    run_dir = train_one(tmp_path, "run_fallback", method="gsat", conv=None)
    _off_by_float_noise(monkeypatch)
    manifest = explain.run_explanations(run_dir, tmp_path / "out_fallback", max_subjects=1,
                                        steps=2, epochs=3)
    assert manifest["replay"]["mode"] == "stored_predictions_tolerance"
    assert manifest["replay"]["max_abs_diff"] <= explain.REPLAY_TOLERANCE
    assert manifest["replay"]["argmax_agreement"] == 1.0
    assert manifest["replay"]["stored_npz_hash_verified"] is True
    assert manifest["replay"]["stored_labels_verified"] is True


def test_replay_refuses_a_stored_npz_that_is_not_the_training_runs(tmp_path, monkeypatch):
    """A stored npz holding another model's probabilities -- even ones the replay would match --
    is refused: its hash is not the recorded prediction hash."""
    run_dir = train_one(tmp_path, "run_foreign_npz", method="gsat", conv=None)
    bundle = explain.load_run_bundle(run_dir)
    splits = explain.rebuild_split(bundle.binding, "test")
    model, is_adapter = explain.build_model(bundle.binding, bundle.state_dict)
    stored = dict(np.load(run_dir / "test.npz"))
    replayed = explain.fold_probabilities(model, splits, "test", bundle.binding["batch_size"], is_adapter)
    monkeypatch.setattr(explain, "fold_probabilities", lambda *a, **k: replayed)
    stored["proba"] = (replayed + np.float32(1e-7)).astype(np.float32)   # matches the replay, not training
    np.savez(run_dir / "test.npz", **stored)
    binding = dict(bundle.binding, selected_test=dict(bundle.binding["selected_test"],
                                                      prediction_sha256="1" * 64))
    with pytest.raises(ValueError, match="not the training run's"):
        explain.replay_check(model, splits, binding, is_method_adapter=is_adapter, fold="test",
                             stored_npz=run_dir / "test.npz")


def test_replay_refuses_a_stored_npz_with_other_labels(tmp_path, monkeypatch):
    run_dir = train_one(tmp_path, "run_relabelled_npz", method="gsat", conv=None)
    stored = dict(np.load(run_dir / "test.npz"))
    stored["y"] = 1 - stored["y"]
    np.savez_compressed(run_dir / "test.npz", **stored)
    _off_by_float_noise(monkeypatch)
    with pytest.raises(ValueError, match="labels differ"):
        explain.run_explanations(run_dir, tmp_path / "out_relabelled_npz", max_subjects=1,
                                 steps=2, epochs=3)


def test_replay_reports_the_exact_hash_mode_when_it_reproduces(tmp_path):
    run_dir = train_one(tmp_path, "run_exact", method="gsat", conv=None)
    manifest = explain.run_explanations(run_dir, tmp_path / "out_exact", max_subjects=1,
                                        steps=2, epochs=3)
    assert manifest["replay"]["mode"] == "exact_hash"
    assert manifest["replay"]["stored_npz_hash_verified"] is None


def test_refuses_to_write_into_an_occupied_output_directory(tmp_path):
    run_dir = train_one(tmp_path, "run_occupied", method="gsat", conv=None)
    output = tmp_path / "explanations_occupied"
    output.mkdir()
    (output / "leftover.txt").write_text("from a previous run")
    with pytest.raises(FileExistsError):
        explain.run_explanations(run_dir, output, max_subjects=1, steps=2, epochs=3)


def test_replay_check_uses_the_trained_batch_size_not_a_hardcoded_default(tmp_path):
    """The byte hash depends on the float summation order, i.e. on how graphs are grouped into
    batches; replay_check must read binding['batch_size']. Force a tiny training batch_size (3)
    against the 4-graph validation (decision) fold, which spans multiple batches at size 3 but not
    at a hardcoded 64."""
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


def test_a_missing_run_file_or_a_changed_preprocessing_json_is_refused(tmp_path):
    run_dir = train_one(tmp_path, "run_files", method="gsat", conv=None)
    prep = run_dir / "preprocessing.json"
    prep.write_text(prep.read_text() + " ")
    with pytest.raises(ValueError, match="preprocessing.json hash"):
        explain.load_run_bundle(run_dir)
    prep.unlink()
    with pytest.raises(FileNotFoundError, match="preprocessing.json missing"):
        explain.load_run_bundle(run_dir)


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


def test_fold_probabilities_is_the_replay_loop(tmp_path):
    from torch_geometric.loader import DataLoader

    run_dir = train_one(tmp_path, "run_fold_probabilities", method="graphcare", conv=None)
    bundle = explain.load_run_bundle(run_dir, "test")
    splits = explain.rebuild_split(bundle.binding, "test")
    model, is_adapter = explain.build_model(bundle.binding, bundle.state_dict)
    model.eval()
    with torch.no_grad():
        old = torch.softmax(torch.cat([
            model(b, epoch=0).logits for b in DataLoader(splits["test"], batch_size=3)]), dim=1).numpy()
    new = explain.fold_probabilities(model, splits, "test", 3, is_adapter)
    assert np.array_equal(new, old)
    replay = explain.replay_check(model, splits, bundle.binding, is_method_adapter=is_adapter, fold="test",
                                  batch_size=3, stored_npz=run_dir / "test.npz")
    assert replay["mode"] in ("exact_hash", "stored_predictions_tolerance")


def _tie_fixture(tmp_path, monkeypatch, stored, replayed, labels=None):
    """replay_check on crafted stored and replayed probabilities (the forward is replaced); the
    stored npz is the 'training run's' (its hash is the recorded one, its labels the fold's)."""
    stored = np.asarray(stored, dtype=np.float64)
    ids = [f"row-{i}" for i in range(len(stored))]
    y = stored.argmax(1)
    np.savez(tmp_path / "test.npz", sample_ids=np.array(ids), proba=stored,
             y=y if labels is None else np.asarray(labels))
    splits = {"test": [types.SimpleNamespace(sample_id=i, y=torch.tensor([int(c)]))
                       for i, c in zip(ids, y)]}
    monkeypatch.setattr(explain, "fold_probabilities",
                        lambda *a, **k: np.asarray(replayed, dtype=np.float64))
    binding = {"batch_size": 4, "selected_test": {"prediction_sha256": explain.proba_digest(stored)}}
    return lambda: explain.replay_check(None, splits, binding, is_method_adapter=True, fold="test",
                                        stored_npz=tmp_path / "test.npz")


def test_a_tie_flip_with_a_stored_gap_between_one_and_two_tolerances_is_accepted(tmp_path, monkeypatch):
    tol = explain.REPLAY_TOLERANCE
    # row 0: stored top-two gap 1.5 tolerances, both probabilities moved by 0.9 tolerances and crossed
    stored = [[0.40000, 0.40000 + 1.5 * tol, 0.19985], [0.2, 0.3, 0.5]]
    replayed = [[0.40000 + 0.9 * tol, 0.40000 + 1.5 * tol - 0.9 * tol, 0.19985], [0.2, 0.3, 0.5]]
    replay = _tie_fixture(tmp_path, monkeypatch, stored, replayed)()
    assert replay["mode"] == "stored_predictions_tolerance"
    assert replay["argmax_flips_total"] == 1 and replay["argmax_flips_tie"] == 1
    assert replay["argmax_agreement"] == 0.5 and replay["rows"] == 2
    assert replay["max_abs_diff"] <= tol and replay["tolerance"] == tol


def test_no_flip_gives_zero_flip_counts_and_full_agreement(tmp_path, monkeypatch):
    stored = [[0.2, 0.8], [0.6, 0.4]]
    replay = _tie_fixture(tmp_path, monkeypatch, stored, [[0.2 + 5e-5, 0.8 - 5e-5], [0.6, 0.4]])()
    assert replay["argmax_flips_total"] == 0 and replay["argmax_flips_tie"] == 0
    assert replay["argmax_agreement"] == 1.0


def test_a_non_tie_flip_is_refused(tmp_path, monkeypatch):
    # stored gap 0.3, the replay reversed it: not a tie (and far over the tolerance)
    replay = _tie_fixture(tmp_path, monkeypatch, [[0.55, 0.25, 0.2], [0.2, 0.3, 0.5]],
                          [[0.25, 0.55, 0.2], [0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match=r"REPLAY CHECK FAILED.*1 of 2 rows, 0 of them on ties"):
        replay()


def test_a_flip_on_a_tie_row_still_needs_every_row_within_the_tolerance(tmp_path, monkeypatch):
    tol = explain.REPLAY_TOLERANCE
    stored = [[0.4, 0.4 + tol, 0.2 - tol], [0.2, 0.3, 0.5]]
    replayed = [[0.4 + tol, 0.4, 0.2 - tol], [0.2, 0.3 + 5 * tol, 0.5 - 5 * tol]]   # row 1 moved by 5 tolerances
    with pytest.raises(ValueError, match="REPLAY CHECK FAILED"):
        _tie_fixture(tmp_path, monkeypatch, stored, replayed)()


def test_stored_ids_that_differ_from_the_fold_are_refused(tmp_path, monkeypatch):
    stored = [[0.2, 0.8], [0.6, 0.4]]
    replay = _tie_fixture(tmp_path, monkeypatch, stored, [[0.2 + 5e-5, 0.8 - 5e-5], [0.6, 0.4]])
    saved = dict(np.load(tmp_path / "test.npz"))
    saved["sample_ids"] = saved["sample_ids"][::-1].copy()
    np.savez(tmp_path / "test.npz", **saved)
    with pytest.raises(ValueError, match="stored sample ids or shape differ"):
        replay()


def test_a_stored_label_that_differs_from_the_fold_is_refused(tmp_path, monkeypatch):
    stored = [[0.2, 0.8], [0.6, 0.4]]
    replay = _tie_fixture(tmp_path, monkeypatch, stored, [[0.2 + 5e-5, 0.8 - 5e-5], [0.6, 0.4]],
                          labels=[0, 0])
    with pytest.raises(ValueError, match="labels differ"):
        replay()


def test_manifest_records_environment_and_timing_and_records_their_seconds(tmp_path):
    run_dir = train_one(tmp_path, "run_provenance", method="gsat", conv=None)
    output = tmp_path / "out_provenance"
    manifest = explain.run_explanations(run_dir, output, max_subjects=2, steps=2, epochs=3)
    env = manifest["environment"]
    assert env["torch"] == torch.__version__ and env["torch_num_threads"] == torch.get_num_threads()
    assert env["python"] and env["platform"] and env["machine"]
    assert 0 < manifest["seconds_before_first_explanation"] < manifest["seconds_total"]
    written = json.loads((output / "explanation_manifest.json").read_text())
    assert written["environment"] == env
    for filename in manifest["record_files"]:
        record = json.loads((output / filename).read_text())
        assert record["seconds"] > 0
        check = record["single_graph_check"]
        assert set(check) == {"max_abs_diff", "argmax_agrees"}
        assert isinstance(check["argmax_agrees"], bool) and check["max_abs_diff"] >= 0.0


def test_single_graph_check_is_null_without_a_stored_row_and_never_refuses(tmp_path):
    run_dir = train_one(tmp_path, "run_single_graph", method="gsat", conv=None)
    binding, splits, wrapper, records = subject_setup(run_dir)
    data = splits["test"][0]
    kwargs = subject_kwargs(binding, records, random_repeats=0)
    assert explain.explain_subject(wrapper, data, **kwargs)["single_graph_check"] is None
    far = np.array([1.0, 0.0])      # a stored row far from any single-graph forward
    record = explain.explain_subject(wrapper, data, stored_row=far, **kwargs)
    assert record["single_graph_check"]["max_abs_diff"] > explain.REPLAY_TOLERANCE
    assert isinstance(record["single_graph_check"]["argmax_agrees"], bool)
    assert record["seconds"] > 0


def test_explain_subject_needs_the_visit_records(tmp_path):
    run_dir = train_one(tmp_path, "run_needs_records", method="gsat", conv=None)
    binding, splits, wrapper, records = subject_setup(run_dir)
    kwargs = subject_kwargs(binding, None)
    with pytest.raises(ValueError, match="visit records"):
        explain.explain_subject(wrapper, splits["test"][0], **kwargs)
    other = explain.VisitRecords({}, records.encode)
    with pytest.raises(ValueError, match="no graph record"):
        explain.explain_subject(wrapper, splits["test"][0], **subject_kwargs(binding, other))


def test_a_record_whose_encoding_is_not_the_split_tensor_is_refused(tmp_path):
    run_dir = train_one(tmp_path, "run_encoding_check", method="gsat", conv=None)
    binding, splits, wrapper, records = subject_setup(run_dir)
    data = splits["test"][0]
    graph, membership = records.records[data.sample_id]
    changed = copy.deepcopy(graph)
    next(n for n in changed["nodes"] if n["kind"] == "vital")["value"] += 5.0
    tampered = explain.VisitRecords({data.sample_id: (changed, membership)}, records.encode)
    with pytest.raises(ValueError, match="differs from the split tensor"):
        explain.explain_subject(wrapper, data, **subject_kwargs(binding, tampered))


def test_prototype_projection_status_reads_the_selected_epoch(tmp_path):
    binding = {"method": "protgnn",
               "method_config": {"effective_settings": {"proj_epochs": 20, "proj_interval": 25}}}
    cases = ((8, False, None, None), (20, False, None, None), (21, True, 21, 1), (27, True, 21, 7),
             (46, True, 46, 1), (50, True, 46, 5))
    for selected, projected, last, after in cases:
        (tmp_path / "result.json").write_text(json.dumps({"selected_epoch": selected}))
        status = explain.prototype_projection_status(tmp_path, binding)
        assert status["prototypes_projected"] is projected
        assert status["first_projection_epoch"] == 21
        assert status["last_projection_epoch"] == last
        assert status["epochs_trained_after_last_projection"] == after
        assert status["selected_epoch_source"] == "result.json"
    assert explain.prototype_projection_status(tmp_path, {"method": "gsat"}) is None
    (tmp_path / "result.json").unlink()
    # ProtGNN without result.json: the selected epoch of the explained fold's binding block, else
    # recorded as not available
    assert explain.prototype_projection_status(tmp_path, binding) == {
        "status": "selected_epoch_not_recorded"}
    assert explain.prototype_projection_status(tmp_path, binding, "test") == {
        "status": "selected_epoch_not_recorded"}
    with_block = dict(binding, selected_test={"epoch": 22})
    status = explain.prototype_projection_status(tmp_path, with_block, "test")
    assert status["selected_epoch"] == 22 and status["epochs_trained_after_last_projection"] == 2
    assert status["selected_epoch_source"] == "binding.json selected_test"


# ------------------------------------------------------------------ records, isolation, schema
def test_a_visit_that_raises_is_listed_in_the_failures(tmp_path, monkeypatch):
    run_dir = train_one(tmp_path, "run_failing_visit", method="gsat", conv=None)
    real = explain.explain_subject
    failing = {}

    def flaky(wrapper, data, **kwargs):
        if not failing:
            failing["id"] = str(data.sample_id)
            raise RuntimeError("synthetic visit failure")
        return real(wrapper, data, **kwargs)

    monkeypatch.setattr(explain, "explain_subject", flaky)
    output = tmp_path / "out_failing_visit"
    manifest = explain.run_explanations(run_dir, output, max_subjects=3, steps=2, epochs=3,
                                        random_repeats=2)
    assert manifest["failed_count"] == 1 and manifest["explained_count"] == 2
    assert manifest["failures"] == [{"sample_id": failing["id"], "error": "synthetic visit failure"}]
    assert len(manifest["record_files"]) == 2
    assert f"subject_{failing['id']}.json" not in manifest["record_files"]


def test_a_random_reference_failure_does_not_lose_the_metrics(tmp_path, monkeypatch):
    run_dir = train_one(tmp_path, "run_reference_failure", method="gsat", conv=None)
    binding, splits, wrapper, records = subject_setup(run_dir)

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic reference failure")

    monkeypatch.setattr(fv2, "random_reference", broken)
    record = explain.explain_subject(wrapper, splits["test"][0], **subject_kwargs(binding, records))
    assert record["random_reference"]["status"] == "failed"
    assert "synthetic reference failure" in record["random_reference"]["error"]
    scored = [m for m in record["fidelity_v2"].values() if "fidelity_plus" in m]
    assert scored and all("vs_random" not in m for m in scored)


def test_one_source_failure_does_not_lose_the_others(tmp_path, monkeypatch):
    run_dir = train_one(tmp_path, "run_source_failure", method="gsat", conv=None)
    binding, splits, wrapper, records = subject_setup(run_dir)
    real, calls = explain._source_metrics, []

    def first_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("synthetic source failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(explain, "_source_metrics", first_fails)
    record = explain.explain_subject(wrapper, splits["test"][0], **subject_kwargs(binding, records))
    statuses = [m.get("status") for m in record["fidelity_v2"].values()]
    assert statuses.count("failed") == 1 and len(statuses) > 1
    failed = next(m for m in record["fidelity_v2"].values() if m.get("status") == "failed")
    assert failed["error"] == "synthetic source failure"


def test_a_source_without_eligible_evidence_is_counted_not_failed(tmp_path, monkeypatch):
    """A built-in (or any source) with no nonzero importance on an eligible evidence node, e.g. a
    ProtGNN subgraph that lies wholly on structural nodes, is `no_eligible_evidence`, not a failure."""
    run_dir = train_one(tmp_path, "run_no_eligible", method="gsat", conv=None)
    binding, splits, wrapper, records = subject_setup(run_dir)
    data = splits["test"][0]
    structural = fv2.removable_node_mask(data.node_type).logical_not().float().tolist()
    monkeypatch.setattr(wrapper, "builtin_node_importance", lambda *a, **k: structural)
    record = explain.explain_subject(wrapper, data, **subject_kwargs(binding, records))
    assert record["builtin_status"] == "no_eligible_evidence"
    assert record["builtin_tie_summary"] is None
    assert record["fidelity_v2"]["builtin"] == {"status": "no_eligible_evidence", "support": 0}
    assert record["fidelity_v2"]["GradExplainer"].get("fidelity_plus") is not None


def test_records_carry_schema_3_and_every_source_is_scored_by_record_deletion(tmp_path):
    run_dir = train_one(tmp_path, "run_schema", method="gsat", conv=None)
    binding, splits, wrapper, records = subject_setup(run_dir)
    data = splits["test"][0]
    record = explain.explain_subject(wrapper, data, **subject_kwargs(binding, records))
    assert record["schema_version"] == explain.RECORD_SCHEMA_VERSION == 3
    assert {"num_visits", "builtin_class_specific", "builtin_status", "builtin_tie_summary",
            "deletion_check", "fidelity_v2", "fidelity_v2_intervention"} <= set(record)
    assert record["num_visits"] == int(data.num_visits.view(-1)[0])
    assert record["builtin_class_specific"] is False
    assert record["deletion_check"]["encoding_equals_split_tensor"] is True
    assert record["deletion_check"]["max_abs_prob_diff"] <= 1e-6
    intervention = record["fidelity_v2_intervention"]
    assert intervention["version"] == fv2.INTERVENTION_CONTRACT_VERSION == 6
    assert intervention["empty_visits"] == explain.EMPTY_VISIT_TEXT
    assert "emptied are dropped" in intervention["empty_visits"]
    assert "empty in the record are kept" in intervention["empty_visits"]
    assert intervention["chains"].startswith("not relinked") and "baseline_of" in intervention["chains"]
    assert "prior labs of an analyte" in intervention["departures_from_a_rebuild"]
    assert "baseline_of edge is not re-anchored" in intervention["departures_from_a_rebuild"]
    assert "(state node, rule_id)" in intervention["combos"]
    assert intervention["cascade"]
    # the fixture artifact has no threshold layer: the deletion's fallback combo rule, recorded
    assert intervention["combo_source"]["source"] == "distinct_anchor_tokens"
    assert record["deletion_check"]["combo_source"] == intervention["combo_source"]
    deletions = record["deletion_check"]["deletions"]
    assert deletions["sets_scored"] > 0 and deletions["visits_originally_empty"] == 0
    assert deletions["not_reevaluated"] == [] and deletions["sets_with_combo_groups_deleted"] == 0
    # every number is the record-level deletion scorer's (same seed, same tie draws)
    graph, membership = records.records[data.sample_id]
    scorer = DeletionScorer(wrapper.deletion_forward, graph, membership, records.encode, y=int(data.y))
    seed = explain._stable_seed(1234, str(data.sample_id))
    for name, metrics in record["fidelity_v2"].items():
        if "fidelity_plus" not in metrics:
            continue
        importance = (record["builtin_node_importance"] if name == "builtin"
                      else record["graphxai"][name]["node_explanation"]["node_importance"])
        plus = fv2.fidelity_plus_v2(scorer, importance, record["prediction_class_id"], tie_draws=3, seed=seed)
        assert metrics["fidelity_plus"]["prob"] == pytest.approx(plus["prob"], abs=1e-6), name
        assert metrics["tie_summary"] == fv2.tie_summary(data.node_type, importance)
        assert {"tied_at_cut", "tie_group_size", "tie_draws"} <= set(metrics["fidelity_plus"])
        assert metrics["vs_random"]["fidelity_plus"]["beats_random_fraction"] <= 1.0
    assert record["random_reference"]["status"] == "computed" and record["random_reference"]["repeats"] == 4


def test_ig_escalates_per_visit_and_records_its_path(tmp_path):
    run_dir = train_one(tmp_path, "run_ig_path", method="gsat", conv=None)
    output = tmp_path / "out_ig_path"
    manifest = explain.run_explanations(run_dir, output, max_subjects=2, steps=2, epochs=3,
                                        ig_step_candidates=(2, 4, 8), random_repeats=2)
    selection = manifest["clinical_ig"]["step_selection"]
    assert selection["pilot_fold"] == "validation" and selection["pilot_graphs"] >= 1
    assert manifest["settings"]["ig"]["step_candidates"] == [2, 4, 8]
    for filename in manifest["record_files"]:
        record = json.loads((output / filename).read_text())
        # the zero-baseline IG stays in the records but is not scored
        assert "IntegratedGradExplainerZeroBaseline" in record["graphxai"]
        assert "IntegratedGradExplainerZeroBaseline" not in record["fidelity_v2"]
        assert record["fidelity_v2_skipped_sources"] == ["IntegratedGradExplainerZeroBaseline"]
        assert "fidelity_plus" in record["fidelity_v2"]["IntegratedGradExplainer"]
        ig = record["graphxai"]["IntegratedGradExplainer"]["provenance"]
        assert ig["start_steps"] == manifest["clinical_ig"]["steps"]
        assert ig["steps_used"] >= ig["start_steps"] and ig["steps_used"] in (2, 4, 8)
        assert [t["steps"] for t in ig["tried"]][-1] == ig["steps_used"]
        assert "steps" not in ig        # start count and count used have their own keys
        assert ig["completeness_met"] == (ig["completeness_error_rel"] <= ig["tolerance"])
        assert ig["node_reduction"] == "sum_abs_over_columns" and "ple_term" in ig


def test_both_ig_payloads_name_the_step_count_they_ran_with_steps_used(monkeypatch):
    """An escalated visit (start 8, ran with 32): the main and the real-patient payloads both carry
    the count IG ran with as `steps_used`, and no key `steps` that could mean the start count."""
    nodes, columns = 4, 3
    calls = {}

    def until_complete(wrapper, x, edge_index, baseline, target, *, start_steps, candidates, tolerance):
        calls["start"] = start_steps
        return torch.ones(nodes, columns), {"steps_used": 32, "completeness_met": True,
                                            "completeness_error_rel": 0.01, "tolerance": tolerance,
                                            "tried": [{"steps": 8}, {"steps": 16}, {"steps": 32}]}

    def expected_gradients(wrapper, x, edge_index, token, node_type, bank, baseline, target, steps, **_):
        calls["refs_steps"] = steps
        return torch.ones(nodes, columns), {"draws": 1, "mean_relative_error": 0.0}

    monkeypatch.setattr(explain.ig_clinical, "not_recorded_baseline", lambda x, *a, **k: torch.zeros_like(x))
    monkeypatch.setattr(explain.ig_clinical, "integrated_gradients_until_complete", until_complete)
    monkeypatch.setattr(explain.ig_clinical, "expected_gradients", expected_gradients)
    batch = types.SimpleNamespace(edge_index=torch.zeros(2, 0, dtype=torch.long),
                                  node_type=torch.zeros(nodes, dtype=torch.long),
                                  token=torch.zeros(nodes, dtype=torch.long))
    ig = {"layout": [], "steps": 8, "candidates": (8, 16, 32), "tolerance": 0.05, "bank": object(),
          "reference_draws": 1, "seed": 3, "step_selection": {}, "explain_predicted": False}
    payloads = explain._clinical_ig_payloads(types.SimpleNamespace(), torch.ones(nodes, columns), batch,
                                             torch.zeros(nodes, dtype=torch.long), 0, ig)
    main, refs = payloads["main"]["provenance"], payloads["refs"]["provenance"]
    assert calls == {"start": 8, "refs_steps": 32}
    assert main["start_steps"] == 8 and main["steps_used"] == 32 and "steps" not in main
    assert refs["steps_used"] == main["steps_used"] and "steps" not in refs


def test_ig_pilot_takes_one_visit_per_patient():
    visits = [types.SimpleNamespace(subject=s, sample_id=f"{s}-{i}")
              for i, s in enumerate(["p1", "p1", "p2", "p3", "p3", "p4"])]
    fold, rows = explain._ig_pilot_rows({"validation": visits}, 30, seed=5)
    assert fold == "validation" and len(rows) == 4
    assert [r.sample_id for r in rows if r.subject == "p1"] == ["p1-0"]
    fold, rows = explain._ig_pilot_rows({"validation": visits}, 2, seed=5)
    assert len(rows) == 2 and len({r.subject for r in rows}) == 2
    assert inspect.signature(explain.run_explanations).parameters["ig_pilot_patients"].default == 30


def test_defaults_follow_the_fidelity_contract():
    params = inspect.signature(explain.run_explanations).parameters
    assert params["fidelity_v2_curve_points"].default == fv2.CURVE_POINTS == 10
    assert params["random_repeats"].default == fv2.RANDOM_REPEATS == 50
    assert params["tie_draws"].default == fv2.TIE_DRAWS == 10
    params = inspect.signature(explain.explain_subject).parameters
    assert params["fidelity_v2_curve_points"].default == 10 and params["random_repeats"].default == 50


def test_the_graphcare_inert_edges_and_the_method_reach_the_explainers(tmp_path, monkeypatch):
    from core.explain import graphxai_standardized as gx

    run_dir = train_one(tmp_path, "run_inert", method="graphcare", conv=None)
    binding, splits, wrapper, records = subject_setup(run_dir)
    seen = {}
    real = gx.explain_algorithms

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(gx, "explain_algorithms", spy)
    data = splits["test"][0]
    explain.explain_subject(wrapper, data, **subject_kwargs(binding, records, random_repeats=0))
    assert seen["method"] == "graphcare"
    wrapper.set_context(explain._single_graph_batch(data))
    assert torch.equal(seen["inert_edge_mask"], wrapper.inert_edge_mask())


def test_manifest_and_binding_carry_settings_disclosure_code_and_the_model_run(tmp_path):
    from comparison.explain.run import current_source_hashes
    from comparison.explain.summarize import code_identity

    run_dir = train_one(tmp_path, "run_manifest", method="gsat", conv=None)
    output = tmp_path / "out_manifest"
    manifest = explain.run_explanations(run_dir, output, max_subjects=1, steps=2, epochs=3,
                                        random_repeats=2)
    training = json.loads((run_dir / "binding.json").read_text())
    settings = manifest["settings"]
    assert {"curve_points", "random_repeats", "tie_draws", "gnnexplainer_epochs",
            "gnnexplainer_seed_rule", "gnnexplainer_feature_penalty", "ig"} <= set(settings)
    assert settings["gnnexplainer_feature_penalty"] == "mean" and settings["random_repeats"] == 2
    assert {"tolerance", "step_candidates", "escalation"} <= set(settings["ig"])
    assert set(manifest["disclosure"]) >= set(explain.FINAL_METHODS)
    for block in manifest["disclosure"].values():
        assert {"intervention", "builtin", "explainers", "caveats"} <= set(block)
    assert any("official ProtGNN 10/5/10/10" in c for c in manifest["disclosure"]["protgnn"]["caveats"])
    assert any("induced" in c for c in manifest["disclosure"]["protgnn"]["caveats"])
    assert any("absence" in c for c in manifest["disclosure"]["cei_gnn_mp2"]["caveats"])
    assert manifest["builtin_explanation"]["method"] == "gsat"
    assert manifest["builtin_explanation"]["class_specific"] is False
    code = manifest["code"]
    assert code["graphxai_upstream_commit"] == "a11e65f"
    assert len(code["vendored_graphxai_tree_sha256"]) == 64
    assert code["git_head"] and code["git_dirty"] in (True, False, "not_a_checkout")
    assert code["source_hashes"] == current_source_hashes()
    assert {"protgnn/adapter.py", "gsat/adapter.py", "graphcare/adapter.py",
            "cei/studies/cei_v3_graphxai.py", "core/explain/runner.py",
            "comparison/explain/summarize.py"} <= set(code["source_hashes"])
    assert not any(part.startswith("._") for key in code["source_hashes"] for part in key.split("/"))
    assert code_identity(code) is not None
    written = json.loads((output / "binding.json").read_text())
    model_run = written["model_run"]
    assert model_run["protocol"] == training["protocol"]
    assert model_run["top_k_labels"] == training["top_k_labels"] == 2
    assert model_run["num_classes"] == training["num_classes"]
    assert model_run["selected_epoch"] == training["selected_test"]["epoch"]
    assert model_run["protocol_overrides"] == training["protocol_overrides"]
    bundle = explain.load_run_bundle(run_dir)
    model, _ = explain.build_model(bundle.binding, bundle.state_dict)
    rebuilt = json.loads(json.dumps(model.run_config()))
    assert model_run["method_config_sha256"] == explain._canonical_sha256(rebuilt)
    assert model_run["effective_settings"] == rebuilt["effective_settings"]
    assert model_run["run_dir"] == f"{tmp_path.name}/run_manifest"
    assert manifest["run_dir"] == model_run["run_dir"]
    assert written["source_code"] == code["source_hashes"]
    assert written["code"]["graphxai_upstream_commit"] == "a11e65f"
    assert written["explainer_config"]["intervention_contract_version"] == 6
    assert manifest["intervention_contract_version"] == 6
    assert manifest["combo_source"] == written["explainer_config"]["combo_source"]
    assert manifest["combo_source"]["source"] == "distinct_anchor_tokens"
    for block in manifest["disclosure"].values():
        text = block["intervention"]
        assert "version 6" in text and "(state node, rule_id)" in text
        assert explain.EMPTY_VISIT_TEXT in text and explain.DEPARTURES_TEXT in text
    assert written["records"]["manifest_sha256"] == sha256(output / "explanation_manifest.json")


def test_outputs_hold_no_absolute_path(tmp_path):
    from core.paths import REPO_ROOT

    run_dir = train_one(tmp_path, "run_paths", method="gsat", conv=None)
    output = tmp_path / "out_paths"
    explain.run_explanations(run_dir, output, max_subjects=2, steps=2, epochs=3, random_repeats=2)
    forbidden = {str(tmp_path), str(tmp_path.resolve()), str(REPO_ROOT)}
    for path in output.iterdir():
        text = path.read_text()
        assert not any(value in text for value in forbidden), path.name


def test_the_relative_path_helper_and_the_path_scrubber():
    assert explain._relative_run_path("/runs/top10/gsat_seed1234") == "top10/gsat_seed1234"
    scrubbed = explain._without_local_paths({"v3_state": "/a/b/K4.json", "k": 4, "items": ["/x/y.json", "z"]})
    assert scrubbed == {"v3_state": "K4.json", "k": 4, "items": ["y.json", "z"]}


def test_source_hashes_skip_copy_leftovers_and_caches(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "core" / "__pycache__").mkdir(parents=True)
    (root / "core" / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(explain, "REPO_ROOT", root)
    monkeypatch.setattr(explain, "CODE_ROOTS", ("core",))
    before = explain.explanation_source_hashes()
    (root / "core" / "._a.py").write_bytes(b"\x00\x05\x16\x07")
    (root / "core" / "__pycache__" / "a.py").write_text("y = 2\n")
    assert explain.explanation_source_hashes() == before == {"core/a.py": sha256(root / "core" / "a.py")}


def test_the_vendored_tree_hash_skips_copy_leftovers(tmp_path, monkeypatch):
    tree = tmp_path / "graphxai"
    (tree / "explainers").mkdir(parents=True)
    (tree / "explainers" / "grad.py").write_text("g = 1\n")
    monkeypatch.setattr(explain, "VENDORED_GRAPHXAI", tree)
    before = explain.vendored_graphxai_tree_sha256()
    (tree / "explainers" / "._grad.py").write_bytes(b"\x00")
    assert explain.vendored_graphxai_tree_sha256() == before
    (tree / "explainers" / "grad.py").write_text("g = 2\n")
    assert explain.vendored_graphxai_tree_sha256() != before


def test_git_state_outside_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(explain, "REPO_ROOT", tmp_path)
    assert explain._git_state() == {"git_head": "not_a_checkout", "git_dirty": "not_a_checkout"}


# ------------------------------------------------------------------ builder records and the threshold table
def builder_artifact(base, count=24, test_rows=4):
    """An artifact whose graph records come from the graph builder itself
    (`build_graph_with_visit_membership` on the fake event store of tests/test_node_deletion.py):
    prior visits (one of them empty in the record), analytes with prior labs and a baseline edge, prior
    diagnoses, knowledge nodes, threshold states and combo states (SIRS through one and through two
    combos, qSOFA, DKA on the later of two glucose labs), with the threshold table copied next to the
    graphs and named, with its sha256 and level source, in the manifest as the graph build does."""
    import csv
    from datetime import datetime

    from core.contracts import VISIT_MEMBERSHIP_CONTRACT_VERSION, VISIT_MEMBERSHIP_FILENAME
    from core.schema import THRESHOLD_COLUMNS, read_thresholds
    from data.s4_graph.graph import build_graph_with_visit_membership
    from tests import test_node_deletion as nd

    root = base / "artifact"
    root.mkdir(parents=True)
    table = root / "thresholds.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=THRESHOLD_COLUMNS)
        writer.writeheader()
        writer.writerows(nd.RULES)
    rules = read_thresholds(table)
    splits = ["train"] * (count - 4 - test_rows) + ["validation"] * 4 + ["test"] * test_rows
    graphs, rows = [], []
    for i, split in enumerate(splits):
        spec = nd.PATIENTS[("P1", "P2", "P3")[i % 3]]
        sample = {"sample_id": f"bv-{i:02d}", "subject_id": "s1", "stay_id": "I",
                  "cutoff": datetime(2020, 7, 1, 2), "split": split}
        graph, membership = build_graph_with_visit_membership(
            nd.FakeStore(spec), sample, nd.KNOWLEDGE_TRIPLES, nd.COMPLAINT_VOCABULARY,
            nd.FakeDiagnoses(spec["diagnoses"]), thresholds=rules)
        graphs.append(graph)
        rows.append(membership)
    graphs_path, membership_path = root / "graphs.jsonl", root / VISIT_MEMBERSHIP_FILENAME
    graphs_path.write_text("".join(json.dumps(g, default=str) + "\n" for g in graphs))
    membership_path.write_text("".join(json.dumps(m) + "\n" for m in rows))
    labels = [f"synthetic_label_{i}" for i in range(train.NUM_CLASSES)]
    canonical = base / "canonical.json"
    canonical.write_text(json.dumps({"classes": labels}))
    manifest = {"status": "completed", "schema_version": "clinical_graph_v2",
                "logic_contract_version": "clinical_graph_logic_v2", "temporal_clean": False,
                "graphs_sha256": sha256(graphs_path), "inherited_cohort_sha256": "synthetic",
                "visit_membership_contract_version": VISIT_MEMBERSHIP_CONTRACT_VERSION,
                "visit_membership_file": VISIT_MEMBERSHIP_FILENAME, "visit_membership_rows": count,
                "visit_membership_sha256": sha256(membership_path), "level_source": "csv",
                "thresholds": {"file": "thresholds.csv", "sha256": sha256(table), "rows": len(rules)},
                "limitations": ["Synthetic fixture; not a research result."]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    targets = base / "targets.csv"
    targets.write_text("sample_id,target,split,subject_id\n" + "".join(
        f"bv-{i:02d},{i % 2},{split},person-{i:02d}\n" for i, split in enumerate(splits)))
    (base / "binding_manifest.json").write_text(json.dumps(
        {"status": "completed", "labels": labels, "cohort_sha256": "synthetic",
         "label_policy_version": "primary_dx_v3", "artifact_files": {"targets.csv": sha256(targets)}}))
    return root, targets, canonical


@pytest.fixture(scope="module")
def builder_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("builder_run")
    root, targets, canonical = builder_artifact(base)
    out = base / "run_builder"
    train.run(run_args(root, targets, canonical, out, method="gsat", conv=None))
    return out, root


def test_the_record_encoder_is_the_split_tensor_on_builder_records(builder_run):
    """Multi-visit builder records of a rebuilt run: the record encoder gives the split tensor on
    every encoded field, for the test and the validation fold, and each visit's deletion scorer (built
    through VisitRecords.scorer, which checks the same) scores the untouched record exactly as the
    adapter scores the split tensor."""
    from torch_geometric.data import Batch

    run_dir, _ = builder_run
    bundle = explain.load_run_bundle(run_dir)
    model, _ = explain.build_model(bundle.binding, bundle.state_dict)
    wrapper = explain.build_wrapper(bundle.binding, model)
    assert len(ENCODED_FIELDS) == 11
    visit_counts = set()
    for fold in ("test", "validation"):
        splits = explain.rebuild_split(bundle.binding, fold)
        records = splits.visit_records([d.sample_id for d in splits[fold]])
        for data in splits[fold]:
            graph, membership = records.records[data.sample_id]
            assert encoding_differences(splits.encode(graph, membership), data) == [], data.sample_id
            scorer = records.scorer(wrapper.deletion_forward, data)
            with torch.no_grad():
                adapter = torch.softmax(model.eval()(Batch.from_data_list([data]), epoch=0).logits, -1)
            assert torch.equal(scorer.probs(()), adapter)
            visit_counts.add(int(data.num_visits.view(-1)[0]))
    assert max(visit_counts) > 1 and min(visit_counts) == 1     # prior visits and a single visit


def test_the_artifact_threshold_table_reaches_every_deletion(builder_run, tmp_path):
    run_dir, root = builder_run
    binding = json.loads((run_dir / "binding.json").read_text())
    splits = explain.rebuild_split(binding, "test")
    source = splits.combo_source
    assert source["source"] == "threshold_table" and source["sha256_checked"] is True
    assert source["path"] == "thresholds.csv" and source["sha256"] == sha256(root / "thresholds.csv")
    assert source["level_source"] == binding["level_source"] == "csv"
    assert source["combo_rules"] == 4 and source["combo_rows"] == 8
    records = splits.visit_records([d.sample_id for d in splits["test"]])
    assert records.combo_rules is splits.combo_rules

    output = tmp_path / "out_builder"
    manifest = explain.run_explanations(run_dir, output, steps=2, epochs=3, random_repeats=2,
                                        tie_draws=2, ig_step_candidates=(2, 4), clinical_ig=False)
    assert manifest["failed_count"] == 0, manifest["failures"]
    assert manifest["combo_source"] == source
    written = json.loads((output / "binding.json").read_text())
    assert written["explainer_config"]["combo_source"] == source
    totals = {"groups": 0, "refired": 0, "visits_dropped": 0, "originally_empty": 0}
    for filename in manifest["record_files"]:
        record = json.loads((output / filename).read_text())
        assert record["deletion_check"]["combo_source"] == source
        assert record["fidelity_v2_intervention"]["combo_source"] == source
        deletions = record["deletion_check"]["deletions"]
        assert deletions["not_reevaluated"] == []          # the builder records hold every input
        totals["groups"] += deletions["sets_with_combo_groups_deleted"]
        totals["refired"] += deletions["sets_with_rules_refired"]
        totals["visits_dropped"] += deletions["sets_with_visits_dropped"]
        totals["originally_empty"] += deletions["visits_originally_empty"]
    # combo groups deleted, a combo fired again on the next-latest value, emptied visits dropped and
    # a visit empty in the record counted
    assert all(value > 0 for value in totals.values()), totals


def test_a_threshold_table_that_is_not_the_artifacts_is_refused(builder_run, tmp_path):
    run_dir, root = builder_run
    binding = json.loads((run_dir / "binding.json").read_text())
    table = root / "thresholds.csv"
    original = table.read_bytes()
    try:
        table.write_bytes(original.replace(b",90,", b",91,"))
        assert table.read_bytes() != original
        with pytest.raises(ValueError, match="threshold table thresholds.csv has sha256"):
            explain.rebuild_split(binding, "test")
        with pytest.raises(ValueError, match="threshold table"):
            explain.run_explanations(run_dir, tmp_path / "out_table", max_subjects=1, steps=2, epochs=3)
        assert not (tmp_path / "out_table").exists()
    finally:
        table.write_bytes(original)
    with pytest.raises(ValueError, match="level_source"):
        explain.rebuild_split(dict(binding, level_source="row-range"), "test")


def test_artifact_combo_rules_without_a_threshold_layer_or_a_recorded_hash(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps({"status": "completed"}))
    assert explain.artifact_combo_rules({"artifact": str(artifact)}) is None
    assert explain.recorded_combo_source(None)["source"] == "distinct_anchor_tokens"
    from tests import test_node_deletion as nd
    import csv
    from core.schema import THRESHOLD_COLUMNS

    with (artifact / "rules.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=THRESHOLD_COLUMNS)
        writer.writeheader()
        writer.writerows(nd.RULES)
    (artifact / "manifest.json").write_text(json.dumps({"thresholds": {"file": "rules.csv"}}))
    rules = explain.artifact_combo_rules({"artifact": str(artifact)})
    described = explain.recorded_combo_source(rules)
    assert described["sha256_checked"] is False and described["sha256"] == sha256(artifact / "rules.csv")
    assert described["path"] == "rules.csv" and described["level_source"] == "csv"


def test_builtin_explanation_of_cei_and_sources_without_eligible_support(four_setups):
    """The CEI manifest block, and a visit whose every source has zero importance on the eligible
    nodes: no random reference is drawn and every source is no_eligible_evidence."""
    from core.tensorize import encode_graph

    models, prep, graph, membership = four_setups
    wrapper = _four_wrapper("cei_gnn_mp2", models["cei_gnn_mp2"], prep)
    block = explain.builtin_explanation({"method": "cei_gnn_mp2"}, wrapper)
    assert block["method"] == "cei_gnn_mp2" and block["class_specific"] is True
    assert block["settings"] == {"ranking": "|signed contribution|"}

    encode = lambda g, m: encode_graph(g, prep, m)
    data = encode(graph, membership)
    wrapper.set_context(explain._single_graph_batch(data))
    scorer = DeletionScorer(wrapper.deletion_forward, graph, membership, encode)
    eligible = fv2.removable_node_mask(scorer.node_type)
    only_protected = np.where(eligible, 0.0, 1.0)
    results, reference = explain._evaluate_sources(
        scorer, {"a": np.zeros(len(eligible)), "b": only_protected}, 0,
        curve_points=4, random_repeats=3, tie_draws=2, seed=1)
    assert reference == {"status": "no_eligible_evidence", "repeats": 3, "seed": 1}
    assert results == {"a": {"status": "no_eligible_evidence", "support": 0},
                       "b": {"status": "no_eligible_evidence", "support": 0}}


# ------------------------------------------------------------------ the four final methods
FOUR_METHODS = ("gsat", "graphcare", "protgnn", "cei_gnn_mp2")


@pytest.fixture(scope="module")
def four_setups(tmp_path_factory):
    from tests import test_node_deletion as nd

    graph, membership = nd.record("four-1")
    prep = nd.preprocessing([graph])
    models = {name: nd.build_model(name, prep, tmp_path_factory.mktemp(name)) for name in FOUR_METHODS}
    return models, prep, graph, membership


def _four_wrapper(name, model, prep):
    from core.method_base import relation_count
    from core.tensorize import ALL_RELATIONS, PAYLOAD_WIDTH, node_feature_layout, triple_count

    binding = {"method": name, "conv": None, "node_dim": len(node_feature_layout(prep)),
               "edge_dim": len(ALL_RELATIONS) + PAYLOAD_WIDTH,
               "vocabulary_size": len(prep["vocabulary"]), "num_meta_relations": triple_count(prep),
               "num_relations": relation_count(None)}
    return explain.build_wrapper(binding, model)


@pytest.mark.parametrize("name", FOUR_METHODS)
def test_each_final_method_is_explained_through_explain_subject(four_setups, name):
    from core.tensorize import ALL_RELATIONS, encode_graph, node_feature_layout

    models, prep, graph, membership = four_setups
    model = models[name]
    encode = lambda g, m: encode_graph(g, prep, m)
    data = encode(graph, membership)
    data.subject = "person-four"
    wrapper = _four_wrapper(name, model, prep)
    x = wrapper.set_context(explain._single_graph_batch(data))
    with torch.no_grad():
        prediction = int(wrapper(x, data.edge_index).argmax(-1))
    true_class = (prediction + 1) % 3                 # a misclassified visit: the true class is explained too
    data.y = torch.tensor([true_class])
    records = explain.VisitRecords({data.sample_id: (graph, membership)}, encode)
    ig = {"layout": node_feature_layout(prep), "steps": 8, "tolerance": 0.05, "candidates": (8, 16),
          "feature_spec": wrapper.ig_feature_spec(), "feature_provenance": wrapper.ig_feature_provenance(),
          "step_selection": {}, "bank": None, "reference_draws": 0, "keep_zero_baseline": True, "seed": 1}
    record = explain.explain_subject(
        wrapper, data, method=name, conv=None, steps=2, epochs=3,
        num_relation_columns=len(ALL_RELATIONS), records=records, ig=ig, random_repeats=3, tie_draws=3)

    assert record["schema_version"] == 3 and record["prediction_class_id"] == prediction
    assert record["wrapper_verification"]["max_abs_logit_diff"] <= 1e-5
    assert record["deletion_check"]["max_abs_prob_diff"] <= 1e-6
    assert record["builtin_class_specific"] is (name in ("protgnn", "cei_gnn_mp2"))
    assert record["builtin_status"] in ("available", "unavailable", "no_eligible_evidence")
    for source, payload in record["graphxai"].items():
        assert payload["status"] in ("success", "failed", "unsupported"), (source, payload)
        if payload["status"] == "success" and source not in explain.FIDELITY_SKIPPED_SOURCES:
            entry = record["fidelity_v2"][source]
            assert "fidelity_plus" in entry or entry.get("status") in ("no_eligible_evidence", "failed")
    ig_provenance = record["graphxai"]["IntegratedGradExplainer"]["provenance"]
    assert {"steps_used", "completeness_met", "ple_term", "node_reduction"} <= set(ig_provenance)
    if name == "cei_gnn_mp2":
        from core.explain.cei_bridge import PLE_TERM_IG

        assert ig_provenance["ple_term"] == PLE_TERM_IG
    # the deletion scorer of the record scores the built-in as the runner did
    if record["builtin_status"] == "available":
        assert "fidelity_plus" in record["fidelity_v2"]["builtin"]
    # the true class: probability change only, and the class-specific built-in's own detail
    block = record["true_class_explanations"]
    assert block["target_class"] == true_class
    assert "fidelity_v2" not in block
    for metrics in block["true_class_probability_change"].values():
        if "fidelity_plus" in metrics:
            assert "acc" not in metrics["fidelity_plus"] and "acc" not in metrics["fidelity_minus"]
            assert metrics["fidelity_plus"]["target_class"] == true_class
    if name in ("protgnn", "cei_gnn_mp2") and block["builtin_detail"] is not None:
        assert block["builtin_detail"]["explained_class"] == true_class
