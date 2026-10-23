"""Resumable explanation runs (`run_explanations(..., checkpoint_dir=...)`, CLI `--checkpoint-dir`).

Built on the synthetic run the other runner tests use (`train.run()` on tiny data): one GSAT run is
explained without a checkpoint (the reference), with a checkpoint, interrupted after k visits and resumed.
Records are compared field by field except the per-visit `seconds` timing.
"""
import copy
import json
import shutil

import pytest

from core import train
from core.explain import runner as explain
from tests.graphxai_fixtures import artifact, run_args

SUBJECTS = 3
KWARGS = dict(max_subjects=SUBJECTS, steps=2, epochs=3, random_repeats=2)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("explain_checkpoint")
    root, targets, canonical = artifact(base, count=24, test_rows=4)
    out = base / "run_gsat"
    train.run(run_args(root, targets, canonical, out, method="gsat", conv=None))
    return out


@pytest.fixture(scope="module")
def reference(run_dir, tmp_path_factory):
    """The run without a checkpoint: (output dir, manifest)."""
    output = tmp_path_factory.mktemp("reference") / "out"
    return output, explain.run_explanations(run_dir, output, **KWARGS)


@pytest.fixture(scope="module")
def full_checkpoint(run_dir, tmp_path_factory):
    """A checkpoint directory holding every record of a finished run, and that run's output."""
    base = tmp_path_factory.mktemp("full_checkpoint")
    manifest = explain.run_explanations(run_dir, base / "out", checkpoint_dir=base / "ckpt", **KWARGS)
    return base / "ckpt", base / "out", manifest


def without_seconds(value):
    if isinstance(value, dict):
        return {k: without_seconds(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [without_seconds(v) for v in value]
    return value


def read_records(output, manifest):
    return [without_seconds(json.loads((output / name).read_text())) for name in manifest["record_files"]]


def spy_on_visits(monkeypatch, *, stop_after=None):
    """Count the visits `explain_subject` is called for; with `stop_after`, interrupt the run (a
    KeyboardInterrupt, which the runner's per-visit `except Exception` does not swallow) once that many
    visits are explained."""
    real = explain.explain_subject
    seen = []

    def spy(wrapper, data, **kwargs):
        if stop_after is not None and len(seen) >= stop_after:
            raise KeyboardInterrupt
        record = real(wrapper, data, **kwargs)
        seen.append(str(data.sample_id))
        return record

    monkeypatch.setattr(explain, "explain_subject", spy)
    return seen


# ---------------------------------------------------------------- (a) interrupted and resumed run
@pytest.mark.parametrize("stop_after", [0, 1, 2])
def test_a_run_interrupted_after_k_visits_and_resumed_equals_the_uninterrupted_run(
        run_dir, reference, tmp_path, monkeypatch, stop_after):
    ref_output, ref_manifest = reference
    ckpt = tmp_path / "ckpt"
    seen = spy_on_visits(monkeypatch, stop_after=stop_after)
    with pytest.raises(KeyboardInterrupt):
        explain.run_explanations(run_dir, tmp_path / "out_killed", checkpoint_dir=ckpt, **KWARGS)
    assert not (tmp_path / "out_killed").exists()
    assert (ckpt / explain.CHECKPOINT_HEADER).is_file()      # written before the first visit
    assert len(list(ckpt.glob("subject_*.json"))) == stop_after == len(seen)
    assert not list(ckpt.glob(".*.tmp"))

    seen.clear()
    monkeypatch.undo()
    resumed_seen = spy_on_visits(monkeypatch)
    manifest = explain.run_explanations(run_dir, tmp_path / "out_resumed", checkpoint_dir=ckpt, **KWARGS)
    assert len(resumed_seen) == SUBJECTS - stop_after        # only the visits without a record
    assert manifest["checkpoint"] == {"used": True, "records_loaded": stop_after, "segments": 2}
    assert manifest["record_files"] == ref_manifest["record_files"]    # the original subject order
    assert read_records(tmp_path / "out_resumed", manifest) == read_records(ref_output, ref_manifest)
    assert manifest["explained_count"] == ref_manifest["explained_count"] == SUBJECTS
    assert manifest["failures"] == []
    resumed_cohort = json.loads((tmp_path / "out_resumed" / "cohort.json").read_text())
    assert resumed_cohort == json.loads((ref_output / "cohort.json").read_text())


def test_the_resumed_output_has_the_reference_layout_and_manifest_keys(
        run_dir, reference, full_checkpoint):
    ref_output, ref_manifest = reference
    _, output, manifest = full_checkpoint
    assert sorted(p.name for p in output.iterdir()) == sorted(p.name for p in ref_output.iterdir())
    assert set(manifest) == set(ref_manifest)
    assert read_records(output, manifest) == read_records(ref_output, ref_manifest)
    assert manifest["checkpoint"] == {"used": True, "records_loaded": 0, "segments": 1}


def test_a_complete_checkpoint_explains_nothing_and_counts_its_segments(
        run_dir, reference, full_checkpoint, tmp_path, monkeypatch):
    ckpt, _, _ = full_checkpoint
    work = tmp_path / "ckpt"
    shutil.copytree(ckpt, work)
    seen = spy_on_visits(monkeypatch)
    manifest = explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)
    assert seen == []
    assert manifest["checkpoint"] == {"used": True, "records_loaded": SUBJECTS, "segments": 2}
    assert json.loads((work / explain.CHECKPOINT_HEADER).read_text())["segments"] == 2
    assert read_records(tmp_path / "out", manifest) == read_records(*reference)


def test_checkpoint_records_are_the_output_records_byte_for_byte(full_checkpoint):
    ckpt, output, manifest = full_checkpoint
    assert sorted(p.name for p in ckpt.glob("subject_*.json")) == sorted(manifest["record_files"])
    for name in manifest["record_files"]:
        assert (ckpt / name).read_bytes() == (output / name).read_bytes()


# ---------------------------------------------------------------- (b) a checkpoint of another run
HEADER_FIELDS = ["run_dir", "binding_preprocessing_sha256", "fold", "fold_sample_ids_sha256",
                 "fold_prediction_sha256", "model_checkpoint_sha256",
                 "explained_sample_ids_sha256", "source_hashes", "vendored_graphxai_tree_sha256",
                 "graphxai_upstream_commit", "settings", "ig_reference_draws", "keep_zero_baseline_ig",
                 "torch", "torch_geometric", "ig_start_steps", "ig_step_selection"]


def test_the_header_names_every_field_the_contract_requires(full_checkpoint):
    header = json.loads((full_checkpoint[0] / explain.CHECKPOINT_HEADER).read_text())
    assert set(header) == set(HEADER_FIELDS) | {"segments"}
    assert header["segments"] == 1 and header["fold"] == "test"
    assert header["ig_start_steps"] is not None and header["ig_step_selection"]


@pytest.mark.parametrize("field", HEADER_FIELDS)
def test_a_header_that_differs_in_any_field_refuses_the_run_and_names_the_field(
        run_dir, full_checkpoint, tmp_path, monkeypatch, field):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    header_path = work / explain.CHECKPOINT_HEADER
    header = json.loads(header_path.read_text())
    header[field] = {"changed": True} if isinstance(header[field], (dict, list)) else "changed"
    header_path.write_text(json.dumps(header))
    seen = spy_on_visits(monkeypatch)
    with pytest.raises(ValueError, match=f"'{field}'"):
        explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)
    assert seen == [] and not (tmp_path / "out").exists()
    assert json.loads(header_path.read_text()) == header      # a refused run leaves the header alone


@pytest.mark.parametrize("change,field", [({"ig_reference_draws": 1}, "ig_reference_draws"),
                                          ({"keep_zero_baseline_ig": False}, "keep_zero_baseline_ig")])
def test_a_restart_with_other_ig_reference_settings_refuses_before_writing(
        run_dir, full_checkpoint, tmp_path, monkeypatch, change, field):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    seen = spy_on_visits(monkeypatch)
    with pytest.raises(ValueError, match=f"'{field}'"):
        explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **{**KWARGS, **change})
    assert seen == [] and not (tmp_path / "out").exists()


def test_a_restart_under_other_software_versions_refuses(
        run_dir, full_checkpoint, tmp_path, monkeypatch):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    monkeypatch.setattr(explain.torch_geometric, "__version__", "0.0.0-other")
    seen = spy_on_visits(monkeypatch)
    with pytest.raises(ValueError, match="'torch_geometric'"):
        explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)
    assert seen == [] and not (tmp_path / "out").exists()


def test_the_header_pins_the_weights_and_stored_predictions(run_dir, full_checkpoint):
    header = json.loads((full_checkpoint[0] / explain.CHECKPOINT_HEADER).read_text())
    assert header["model_checkpoint_sha256"] == explain.sha256(run_dir / "best.pt")
    assert len(header["fold_prediction_sha256"]) == 64


def perturbed_ig_preparation(monkeypatch, **step_selection_changes):
    """The run's own IG preparation, with its step selection changed afterwards (as another process's
    pilot would give it)."""
    real = explain._prepare_clinical_ig

    def prepare(*args, **kwargs):
        config = real(*args, **kwargs)
        selection = dict(config["step_selection"])
        errors = selection["worst_relative_error_by_steps"]
        selection.update(step_selection_changes)
        if "worst_relative_error_by_steps" not in step_selection_changes:
            selection["worst_relative_error_by_steps"] = {k: v * (1 + 2.9e-7) for k, v in errors.items()}
        return dict(config, step_selection=selection)

    monkeypatch.setattr(explain, "_prepare_clinical_ig", prepare)


def test_the_header_holds_the_ig_decision_and_no_measured_error(full_checkpoint):
    header = json.loads((full_checkpoint[0] / explain.CHECKPOINT_HEADER).read_text())
    assert set(header["ig_step_selection"]) == set(explain.IG_DECISION_KEYS)
    assert "worst_relative_error_by_steps" not in json.dumps(header)


def test_pilot_errors_that_differ_in_the_seventh_digit_still_resume(
        run_dir, reference, full_checkpoint, tmp_path, monkeypatch):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    perturbed_ig_preparation(monkeypatch)
    seen = spy_on_visits(monkeypatch)
    manifest = explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)
    assert seen == []
    assert manifest["checkpoint"] == {"used": True, "records_loaded": SUBJECTS, "segments": 2}
    assert read_records(tmp_path / "out", manifest) == read_records(*reference)


def test_another_chosen_step_count_refuses_the_restart(run_dir, full_checkpoint, tmp_path, monkeypatch):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    header = json.loads((work / explain.CHECKPOINT_HEADER).read_text())
    perturbed_ig_preparation(monkeypatch, steps=header["ig_step_selection"]["steps"] + 1)
    with pytest.raises(ValueError, match="'ig_step_selection'"):
        explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)
    assert not (tmp_path / "out").exists()


def test_a_header_field_the_run_does_not_have_is_refused(run_dir, full_checkpoint, tmp_path):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    header_path = work / explain.CHECKPOINT_HEADER
    header = json.loads(header_path.read_text())
    header["another_field"] = 1
    header_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="another_field"):
        explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)


def test_other_settings_or_another_cohort_refuse_a_real_restart(run_dir, full_checkpoint, tmp_path):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    with pytest.raises(ValueError, match="'settings'"):
        explain.run_explanations(run_dir, tmp_path / "out_a", checkpoint_dir=work,
                                 **dict(KWARGS, epochs=4))
    with pytest.raises(ValueError, match="'explained_sample_ids_sha256'"):
        explain.run_explanations(run_dir, tmp_path / "out_b", checkpoint_dir=work,
                                 **dict(KWARGS, max_subjects=SUBJECTS - 1))
    assert not (tmp_path / "out_a").exists() and not (tmp_path / "out_b").exists()


def test_records_without_a_header_are_refused(run_dir, full_checkpoint, tmp_path):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    (work / explain.CHECKPOINT_HEADER).unlink()
    with pytest.raises(ValueError, match="no checkpoint_header.json"):
        explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)


def test_a_checkpoint_directory_inside_the_output_directory_is_refused(run_dir, tmp_path):
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="outside the output directory"):
        explain.run_explanations(run_dir, output, checkpoint_dir=output / "ckpt", **KWARGS)
    assert not output.exists()


# ---------------------------------------------------------------- (c) partial and foreign files
def test_a_truncated_record_file_is_explained_again(
        run_dir, reference, full_checkpoint, tmp_path, monkeypatch, capsys):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    damaged = sorted(work.glob("subject_*.json"))[1]
    damaged.write_text(damaged.read_text()[:40])              # cut mid-record
    seen = spy_on_visits(monkeypatch)
    manifest = explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)
    assert seen == [damaged.name[len("subject_"):-len(".json")]]
    assert manifest["checkpoint"]["records_loaded"] == SUBJECTS - 1
    assert damaged.name in capsys.readouterr().err            # reported, not silently dropped
    assert read_records(tmp_path / "out", manifest) == read_records(*reference)
    assert json.loads(damaged.read_text())["sample_id"] == seen[0]    # the checkpoint is healed


@pytest.mark.parametrize("damage", ["empty", "not_an_object", "other_sample_id"])
def test_an_unreadable_or_mislabelled_record_file_is_not_used(
        run_dir, reference, full_checkpoint, tmp_path, monkeypatch, damage):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    target = sorted(work.glob("subject_*.json"))[0]
    record = json.loads(target.read_text())
    target.write_text({"empty": "", "not_an_object": "[1, 2]",
                       "other_sample_id": json.dumps(dict(record, sample_id="somebody-else"))}[damage])
    seen = spy_on_visits(monkeypatch)
    manifest = explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)
    assert seen == [target.name[len("subject_"):-len(".json")]]
    assert read_records(tmp_path / "out", manifest) == read_records(*reference)


def test_a_record_of_a_visit_that_is_not_explained_is_ignored(
        run_dir, reference, full_checkpoint, tmp_path, monkeypatch):
    work = tmp_path / "ckpt"
    shutil.copytree(full_checkpoint[0], work)
    first = sorted(work.glob("subject_*.json"))[0]
    stray = copy.deepcopy(json.loads(first.read_text()))
    stray["sample_id"] = "not-in-the-cohort"
    (work / "subject_not-in-the-cohort.json").write_text(json.dumps(stray))
    seen = spy_on_visits(monkeypatch)
    manifest = explain.run_explanations(run_dir, tmp_path / "out", checkpoint_dir=work, **KWARGS)
    assert seen == []
    assert manifest["checkpoint"]["records_loaded"] == SUBJECTS
    assert "subject_not-in-the-cohort.json" not in manifest["record_files"]
    assert not (tmp_path / "out" / "subject_not-in-the-cohort.json").exists()


def test_a_failed_visit_is_not_checkpointed_and_is_tried_again(run_dir, tmp_path, monkeypatch):
    real = explain.explain_subject
    failing = []

    def flaky(wrapper, data, **kwargs):
        if not failing:
            failing.append(str(data.sample_id))
            raise RuntimeError("synthetic visit failure")
        return real(wrapper, data, **kwargs)

    ckpt = tmp_path / "ckpt"
    monkeypatch.setattr(explain, "explain_subject", flaky)
    first = explain.run_explanations(run_dir, tmp_path / "out_first", checkpoint_dir=ckpt, **KWARGS)
    assert first["failed_count"] == 1
    assert not (ckpt / f"subject_{failing[0]}.json").exists()
    monkeypatch.setattr(explain, "explain_subject", real)
    second = explain.run_explanations(run_dir, tmp_path / "out_second", checkpoint_dir=ckpt, **KWARGS)
    assert second["failed_count"] == 0 and second["explained_count"] == SUBJECTS
    assert second["checkpoint"]["records_loaded"] == SUBJECTS - 1


# ---------------------------------------------------------------- (d) without the flag
def test_without_the_flag_nothing_is_written_beside_the_output_and_the_manifest_says_so(
        run_dir, reference, tmp_path):
    output, manifest = reference
    assert manifest["checkpoint"] == {"used": False, "records_loaded": 0, "segments": 1}
    names = sorted(p.name for p in output.iterdir())
    assert names == sorted(["binding.json", "cohort.json", "explanation_manifest.json",
                            *manifest["record_files"]])
    assert json.loads((output / "explanation_manifest.json").read_text())["checkpoint"] == \
        manifest["checkpoint"]
    assert not any(p.name.startswith("checkpoint") for p in output.iterdir())


def test_a_run_without_the_flag_and_a_run_with_it_write_the_same_records(
        run_dir, reference, full_checkpoint):
    ref_output, ref_manifest = reference
    _, output, manifest = full_checkpoint
    assert without_seconds(manifest["settings"]) == without_seconds(ref_manifest["settings"])
    assert manifest["record_files"] == ref_manifest["record_files"]
    assert manifest["clinical_ig"] == ref_manifest["clinical_ig"]
    assert read_records(output, manifest) == read_records(ref_output, ref_manifest)
    assert (output / "cohort.json").read_bytes() == (ref_output / "cohort.json").read_bytes()


def test_two_plain_runs_with_other_earlier_random_draws_give_identical_records(
        run_dir, reference, tmp_path):
    import torch

    ref_output, ref_manifest = reference
    torch.manual_seed(987654)
    torch.rand(1000)                                          # other earlier draws on the ambient generator
    manifest = explain.run_explanations(run_dir, tmp_path / "out", **KWARGS)
    assert manifest["record_files"] == ref_manifest["record_files"]
    assert read_records(tmp_path / "out", manifest) == read_records(ref_output, ref_manifest)


def test_the_cli_takes_the_checkpoint_directory(monkeypatch, tmp_path):
    captured = {}

    def fake(run, output, **kwargs):
        captured.update(kwargs)
        return {"method": "gsat", "conv": None, "explained_count": 0, "failed_count": 0}

    monkeypatch.setattr(explain, "run_explanations", fake)
    monkeypatch.setattr("sys.argv", ["runner", "--run-dir", str(tmp_path), "--output",
                                     str(tmp_path / "o"), "--checkpoint-dir", str(tmp_path / "c")])
    explain._cli()
    assert captured["checkpoint_dir"] == tmp_path / "c"
    monkeypatch.setattr("sys.argv", ["runner", "--run-dir", str(tmp_path), "--output",
                                     str(tmp_path / "o")])
    explain._cli()
    assert captured["checkpoint_dir"] is None
