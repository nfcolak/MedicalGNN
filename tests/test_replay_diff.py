"""Tests of the replay difference tool (comparison/explain/replay_diff.py) on tiny synthetic trained runs.

Like tests/test_explain_runner.py, the runs are really trained (train.run on the synthetic artifact), so the
reload/replay path under test is the real one. No real data, no real run.
"""
import copy
import json
import shutil

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from comparison.explain import replay_diff
from core import train
from core.explain import runner
from tests.graphxai_fixtures import artifact, run_args


def train_run(tmp_path_factory, name, **overrides):
    base = tmp_path_factory.mktemp(name)
    root, targets, canonical = artifact(base, count=24, test_rows=4)
    out = base / "run"
    train.run(run_args(root, targets, canonical, out, **overrides))
    return out


@pytest.fixture(scope="module")
def graphcare_run(tmp_path_factory):
    return train_run(tmp_path_factory, "graphcare", method="graphcare", conv=None)


@pytest.fixture(scope="module")
def black_box_run(tmp_path_factory):
    return train_run(tmp_path_factory, "black_box", method="clinical_gnn", conv="edge_conditioned")


def run_tool(run_dir, tmp_path, *extra, name="report.json"):
    out = tmp_path / name
    assert replay_diff.main(["--run-dir", str(run_dir), "--out", str(out), *extra]) == 0
    return json.loads(out.read_text())


def test_fold_probabilities_equals_the_old_replay_loop(graphcare_run, black_box_run):
    for run_dir in (graphcare_run, black_box_run):
        bundle = runner.load_run_bundle(run_dir, "test")
        splits = runner.rebuild_split(bundle.binding, "test")
        model, is_adapter = runner.build_model(bundle.binding, bundle.state_dict)
        batch_size = bundle.binding["batch_size"]
        # the loop replay_check ran before fold_probabilities was extracted
        logits = []
        model.eval()
        with torch.no_grad():
            for batch in DataLoader(splits["test"], batch_size=batch_size):
                logits.append((model(batch, epoch=0).logits if is_adapter else model(batch)).cpu())
        old = torch.softmax(torch.cat(logits), dim=1).numpy()
        new = runner.fold_probabilities(model, splits, "test", batch_size, is_adapter)
        assert new.dtype == old.dtype and np.array_equal(new, old)
        assert runner.proba_digest(new) == bundle.binding["selected_test"]["prediction_sha256"]


def test_agreement_is_exact_on_this_machine(graphcare_run, tmp_path):
    report = run_tool(graphcare_run, tmp_path, "--threads", "1")
    assert report["rows"] == 4 and report["fold"] == "test" and report["method"] == "graphcare"
    assert report["max_abs_diff"] == 0.0 and report["mean_abs_diff"] == 0.0
    assert report["argmax_agreement"] == 1.0 and report["rows_over_1e-4"] == 0
    assert report["exact_hash_match"] is True
    assert sum(report["abs_diff_histogram"].values()) == 4
    assert list(report["abs_diff_histogram"]) == [
        "lt_1e-08", "1e-08_1e-07", "1e-07_1e-06", "1e-06_1e-05", "1e-05_1e-04", "1e-04_1e-03",
        "1e-03_1e-02", "1e-02_1e-01", "ge_1e-01"]
    assert report["environment"]["torch_num_threads"] == 1
    for key in ("torch", "python", "platform", "machine"):
        assert report["environment"][key]
    assert "sample-" not in json.dumps(report) and "/" not in json.dumps(report["method"])


def test_a_perturbed_stored_npz_is_reported_with_the_right_counts(graphcare_run, tmp_path):
    run_dir = tmp_path / "perturbed_run"
    shutil.copytree(graphcare_run, run_dir)
    stored = dict(np.load(run_dir / "test.npz"))
    stored["proba"] = stored["proba"].astype(np.float64)
    original = stored["proba"].copy()
    stored["proba"][0, 0] += 2e-4          # two rows moved by 2e-4 and 3e-6
    stored["proba"][2, 1] += 3e-6
    np.savez(run_dir / "test.npz", **stored)
    report = run_tool(run_dir, tmp_path)
    assert report["rows"] == 4
    assert report["rows_over_1e-4"] == 1
    assert report["max_abs_diff"] == pytest.approx(2e-4, abs=1e-7)
    assert report["mean_abs_diff"] == pytest.approx((2e-4 + 3e-6) / stored["proba"].size, rel=1e-3)
    assert report["exact_hash_match"] is True    # the hash is of the replay, which did not change
    assert report["abs_diff_histogram"]["1e-04_1e-03"] == 1
    assert report["abs_diff_histogram"]["1e-06_1e-05"] == 1
    assert report["abs_diff_histogram"]["lt_1e-08"] == 2
    expected = float((original.argmax(1) == stored["proba"].argmax(1)).mean())
    assert report["argmax_agreement"] == expected


def test_dump_has_no_ids_and_is_identical_across_two_calls(graphcare_run, tmp_path):
    first, second = tmp_path / "dump_a.json", tmp_path / "dump_b.json"
    for dump in (first, second):
        run_tool(graphcare_run, tmp_path, "--dump", str(dump), "--max-batches", "2", "--threads", "1",
                 name=f"report_{dump.stem}.json")
    text = first.read_text()
    assert "sample-" not in text and "person-" not in text
    a, b = json.loads(text), json.loads(second.read_text())
    assert a["batches"] == b["batches"] and len(a["batches"]) == 1   # 4 test rows fit one batch of 4
    batch = a["batches"][0]
    assert {"x", "token", "edge_index", "edge_attr", "edge_relation", "edge_triple",
            "visit_membership_index", "num_visits", "global_node_mask"} <= set(batch["inputs"])
    names = {c["module"] for c in batch["calls"]}
    assert "<root>" in names and any(c["parent"] is not None for c in batch["calls"])
    bundle = runner.load_run_bundle(graphcare_run, "test")
    model, _ = runner.build_model(bundle.binding, bundle.state_dict)
    # every module that is called, leaves and containers (a ModuleList is never called itself)
    called = {n for n, m in model.named_modules()
              if n and not isinstance(m, (torch.nn.ModuleList, torch.nn.ModuleDict))}
    assert called <= names
    assert set(batch["last"]) == {
        "last_alpha", "last_beta", "last_visit_weight", "last_node_context",
        "last_visit_state", "last_graph_state", "last_direct_state", "last_global_state"}
    fingerprint = batch["last"]["last_graph_state"]
    assert set(fingerprint) == {"shape", "dtype", "sum", "abs_sum", "min", "max", "projection"}


def test_max_batches_limits_the_dump(graphcare_run, tmp_path):
    run_tool(graphcare_run, tmp_path, "--dump", str(tmp_path / "d.json"), "--max-batches", "1")
    assert len(json.loads((tmp_path / "d.json").read_text())["batches"]) == 1


def perturbed_loader(graphcare_run, target, seed=0):
    """A loader whose model has 1e-6 relative noise on every weight (the cross-machine floor) and one
    module (`target`) strongly changed."""
    def loader(run_dir, fold):
        model, is_adapter, splits, binding = replay_diff.default_loader(run_dir, fold)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(1 + 1e-6 * torch.randn(parameter.shape, generator=generator))
            dict(model.named_modules())[target].weight.add_(0.3)
        return model, is_adapter, splits, binding
    return loader


def last_linear_of(run_dir):
    bundle = runner.load_run_bundle(run_dir, "test")
    model, _ = runner.build_model(bundle.binding, bundle.state_dict)
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith("classifier")]
    return linears[-1]


def test_compare_marks_the_module_whose_weights_were_perturbed(graphcare_run, tmp_path, capsys):
    target = last_linear_of(graphcare_run)
    dump_a, dump_b = tmp_path / "a.json", tmp_path / "b.json"
    run_tool(graphcare_run, tmp_path, "--dump", str(dump_a), name="ra.json")
    out = tmp_path / "ob.json"
    assert replay_diff.main(["--run-dir", str(graphcare_run), "--out", str(out), "--dump", str(dump_b)],
                            loader=perturbed_loader(graphcare_run, target)) == 0
    table_path = tmp_path / "table.json"
    assert replay_diff.main(["--compare", str(dump_a), str(dump_b), "--out", str(table_path)]) == 0
    printed = capsys.readouterr().out
    table = json.loads(table_path.read_text())
    starts = [r for r in table["rows"] if r["mark"] == "START"]
    assert [r["name"] for r in starts] == [target], printed
    assert target in printed and "START" in printed
    # the marked module's inputs agree at the noise floor, its outputs jump
    row = starts[0]
    assert row["in_rel"] <= table["agree_within"] and row["out_rel"] > 100 * max(row["in_rel"], table["floor"])
    assert table["structure_notes"] == []
    # modules upstream of the perturbed one are not marked
    upstream = [r for r in table["rows"] if r["kind"] == "module" and r["call"] < row["call"] and r["mark"] == "START"]
    assert upstream == []


def test_compare_of_a_dump_with_itself_marks_nothing(graphcare_run, tmp_path):
    dump = tmp_path / "same.json"
    run_tool(graphcare_run, tmp_path, "--dump", str(dump))
    table = replay_diff.compare_dumps(json.loads(dump.read_text()), json.loads(dump.read_text()))
    assert all(r["mark"] == "" for r in table["rows"])
    assert all(r["out_rel"] == 0.0 for r in table["rows"])


def test_relative_difference_definition():
    base = {"shape": [4], "dtype": "float32", "sum": 4.0, "abs_sum": 4.0, "min": 0.5, "max": 2.0, "projection": 1.0}
    other = dict(base, projection=1.5, sum=4.1)
    assert replay_diff.relative_difference(base, base) == 0.0
    assert replay_diff.relative_difference(base, other) == pytest.approx(0.5 / (4.0 / 4 + 1e-12))
    assert replay_diff.relative_difference(base, dict(base, shape=[5])) == float("inf")


def test_compare_refuses_dumps_with_a_different_call_sequence(graphcare_run, tmp_path):
    dump = tmp_path / "d.json"
    run_tool(graphcare_run, tmp_path, "--dump", str(dump))
    a = json.loads(dump.read_text())
    b = copy.deepcopy(a)
    b["batches"][0]["calls"][1]["module"] = "something_else"
    with pytest.raises(ValueError, match="call sequence"):
        replay_diff.compare_dumps(a, b)


def test_refusal_to_overwrite_and_to_write_inside_the_run_dir(graphcare_run, tmp_path):
    existing = tmp_path / "taken.json"
    existing.write_text("{}")
    with pytest.raises(FileExistsError):
        replay_diff.main(["--run-dir", str(graphcare_run), "--out", str(existing)])
    fresh = tmp_path / "fresh.json"
    with pytest.raises(FileExistsError):    # the dump target is checked before any model is loaded
        replay_diff.main(["--run-dir", str(graphcare_run), "--out", str(fresh), "--dump", str(existing)])
    assert not fresh.exists() and existing.read_text() == "{}"
    inside = graphcare_run / "report.json"
    with pytest.raises(ValueError, match="inside the run directory"):
        replay_diff.main(["--run-dir", str(graphcare_run), "--out", str(inside)])
    assert not inside.exists()
    with pytest.raises(FileExistsError):
        replay_diff.main(["--compare", str(existing), str(existing), "--out", str(existing)])


def test_compare_needs_no_model_and_rejects_run_dir(tmp_path):
    with pytest.raises(SystemExit):
        replay_diff.main(["--compare", "a.json", "b.json", "--run-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        replay_diff.main(["--run-dir", str(tmp_path)])      # no --out


def test_stored_ids_that_differ_from_the_rebuilt_fold_are_refused(graphcare_run, tmp_path):
    run_dir = tmp_path / "swapped_run"
    shutil.copytree(graphcare_run, run_dir)
    stored = dict(np.load(run_dir / "test.npz"))
    stored["sample_ids"] = stored["sample_ids"][::-1].copy()
    np.savez(run_dir / "test.npz", **stored)
    with pytest.raises(ValueError, match="sample ids differ"):
        replay_diff.main(["--run-dir", str(run_dir), "--out", str(tmp_path / "r.json")])
    assert not (tmp_path / "r.json").exists()


# ---- the runner's replay rule, applied to the same numbers ----------------------------------------------------------

def shifted_replay(monkeypatch, delta):
    """Make the replay differ from the stored predictions: the first probability moves by `delta`."""
    real = runner.fold_probabilities

    def shifted(*args, **kwargs):
        proba = real(*args, **kwargs).copy()
        proba[0, 0] += delta
        return proba.astype(np.float32)

    monkeypatch.setattr(runner, "fold_probabilities", shifted)


def test_the_runner_rule_is_reported_and_exact_on_this_machine(graphcare_run, tmp_path):
    rule = run_tool(graphcare_run, tmp_path)["runner_rule"]
    assert rule == {"tolerance": runner.REPLAY_TOLERANCE, "stored_npz_hash_verified": True,
                    "stored_labels_match": True, "argmax_flips_total": 0, "argmax_flips_tie": 0,
                    "would_pass": True, "mode": "exact_hash"}


def test_a_replay_within_the_tolerance_of_a_verified_stored_npz_passes_the_runner_rule(graphcare_run, tmp_path,
                                                                                       monkeypatch):
    shifted_replay(monkeypatch, 5e-6)
    report = run_tool(graphcare_run, tmp_path)
    assert report["exact_hash_match"] is False and report["max_abs_diff"] == pytest.approx(5e-6, abs=1e-7)
    assert report["runner_rule"]["would_pass"] is True
    assert report["runner_rule"]["mode"] == "stored_predictions_tolerance"


def test_a_replay_beyond_the_tolerance_is_refused_by_the_runner_rule(graphcare_run, tmp_path, monkeypatch):
    shifted_replay(monkeypatch, 1e-3)
    rule = run_tool(graphcare_run, tmp_path)["runner_rule"]
    assert rule["would_pass"] is False and rule["mode"] == "refused"


def test_the_tool_reads_the_runners_tolerance(graphcare_run, tmp_path, monkeypatch):
    shifted_replay(monkeypatch, 5e-6)
    monkeypatch.setattr(runner, "REPLAY_TOLERANCE", 1e-9)
    rule = run_tool(graphcare_run, tmp_path)["runner_rule"]
    assert rule["tolerance"] == 1e-9 and rule["would_pass"] is False


def test_a_stored_npz_that_is_not_the_recorded_run_is_refused_by_the_runner_rule(graphcare_run, tmp_path,
                                                                                 monkeypatch):
    shifted_replay(monkeypatch, 5e-6)
    tampered = tmp_path / "tampered_run"
    shutil.copytree(graphcare_run, tampered)
    stored = dict(np.load(tampered / "test.npz"))
    stored["proba"] = stored["proba"].copy()
    stored["proba"][1, 0] += 1e-7          # inside the tolerance, but no longer the recorded predictions
    np.savez(tampered / "test.npz", **stored)
    rule = run_tool(tampered, tmp_path)["runner_rule"]
    assert rule["stored_npz_hash_verified"] is False and rule["would_pass"] is False
    relabelled = tmp_path / "relabelled_run"
    shutil.copytree(graphcare_run, relabelled)
    stored = dict(np.load(relabelled / "test.npz"))
    stored["y"] = (stored["y"] + 1) % stored["proba"].shape[1]
    np.savez(relabelled / "test.npz", **stored)
    rule = run_tool(relabelled, tmp_path, name="relabelled.json")["runner_rule"]
    assert rule["stored_npz_hash_verified"] is True and rule["stored_labels_match"] is False
    assert rule["would_pass"] is False


def make_stored(tmp_path, proba, y):
    path = tmp_path / "stored.npz"
    np.savez(path, proba=proba, y=np.array(y))
    return np.load(path)


def test_an_argmax_flip_is_accepted_only_on_a_stored_tie(tmp_path):
    from types import SimpleNamespace

    stored_proba = np.array([[0.5000, 0.5000 - 1e-5, 0.0], [0.7, 0.2, 0.1]], dtype=np.float32)
    rows = [SimpleNamespace(y=torch.tensor([0])), SimpleNamespace(y=torch.tensor([0]))]
    binding = {"selected_test": {"prediction_sha256": runner.proba_digest(stored_proba)}}
    stored = make_stored(tmp_path, stored_proba, [0, 0])
    flipped_on_tie = stored_proba.copy()
    flipped_on_tie[0] = [0.5000 - 1e-5 + 3e-6, 0.5000 - 3e-6, 0.0]       # row 0 flips, its stored gap is 1e-5
    rule = replay_diff.replay_rule(stored, flipped_on_tie, rows, binding, "test", exact=False)
    assert (rule["argmax_flips_total"], rule["argmax_flips_tie"], rule["would_pass"]) == (1, 1, True)
    flipped_off_tie = stored_proba.copy()
    flipped_off_tie[1] = [0.2, 0.7, 0.1]                                  # a real flip far outside the tolerance
    rule = replay_diff.replay_rule(stored, flipped_off_tie, rows, binding, "test", exact=False)
    assert (rule["argmax_flips_total"], rule["argmax_flips_tie"], rule["would_pass"]) == (1, 0, False)
    assert replay_diff.replay_rule(stored, flipped_off_tie, rows, binding, "test", exact=True)["would_pass"] is True
