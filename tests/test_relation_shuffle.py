"""Relation-shuffle probe of trained runs: plan, per-visit records, report."""
import json
from pathlib import Path

import numpy as np
import pytest

from comparison.explain import relation_shuffle as rs
from core import train
from tests import graphxai_fixtures as fixtures
from tests.graphxai_fixtures import artifact, run_args


def train_run(tmp_path, name="top2/gsat_seed7", **overrides):
    root, targets, canonical = artifact(tmp_path, count=24, test_rows=4)
    out = tmp_path / "runs" / name
    train.run(run_args(root, targets, canonical, out, **overrides))
    return out


def probe(run, out, **kwargs):
    kwargs = {"sample_patients": None, "draws": 2, **kwargs}
    return rs.probe_run(run, out, **kwargs)


def read_rows(out):
    return [json.loads(line) for line in (Path(out) / "visits.jsonl").read_text().splitlines()]


def test_probe_replays_the_run_and_records_every_arm(tmp_path):
    run = train_run(tmp_path)
    header = probe(run, tmp_path / "probe")
    rows = read_rows(tmp_path / "probe")
    assert header["replay"] in ("exact_hash", "stored_predictions_tolerance")
    assert header["visits"] == len(rows) == 4 and header["draws"] == 2
    assert set(rows[0]["arms"]) == set(rs.arms())
    assert [len(rows[0]["arms"][a]) for a in ("shuffle_all", "drop_all")] == [2, 1]


def test_unchanged_predictions_are_the_stored_predictions(tmp_path):
    run = train_run(tmp_path)
    probe(run, tmp_path / "probe")
    stored = np.load(run / "test.npz", allow_pickle=False)
    expected = dict(zip([str(s) for s in stored["sample_ids"]], stored["proba"].argmax(1).tolist()))
    assert {r["sample_id"]: r["pred"] for r in read_rows(tmp_path / "probe")} == expected


def rich_graph_from(builder, sample_id, value=70.0, **kwargs):
    """The fixture graph plus four complaints with two co_complaint edges, so a shuffle has something to swap."""
    graph = builder(sample_id, value, **kwargs)
    for name, token in (("c2", "cc:fever"), ("c3", "cc:cough"), ("c4", "cc:nausea")):
        graph["nodes"].append({"id": name, "kind": "complaint", "token": token})
        graph["edges"].append({"source": "v", "target": name, "relation": "reports_complaint", "informative": False})
    graph["edges"] += [{"source": "c", "target": "c2", "relation": "co_complaint", "informative": True},
                       {"source": "c3", "target": "c4", "relation": "co_complaint", "informative": True}]
    graph["coverage"].update(complaints=4, informative_edges=2)
    return graph


def test_shuffles_keep_the_edge_count_and_drop_all_leaves_none(tmp_path, monkeypatch):
    original = fixtures.clinical_graph
    monkeypatch.setattr(fixtures, "clinical_graph", lambda *a, **k: rich_graph_from(original, *a, **k))
    run = train_run(tmp_path)
    probe(run, tmp_path / "probe")
    rows = read_rows(tmp_path / "probe")
    moved = 0
    for row in rows:
        assert row["edges"] > 0
        for arm in ("shuffle_informative", "shuffle_structural", "shuffle_all"):
            assert all(r["edges"] == row["edges"] for r in row["arms"][arm])
        moved += sum(r["moved"] for r in row["arms"]["shuffle_informative"])
        assert row["arms"]["drop_all"][0]["edges"] == 0 and row["arms"]["drop_all"][0]["moved"] == 0
        assert row["arms"]["drop_informative"][0]["edges"] < row["edges"]
        assert all(r["moved"] == 0 for r in row["arms"]["drop_informative"])
    assert moved > 0


def test_a_graph_with_one_edge_per_relation_cannot_be_shuffled(tmp_path):
    run = train_run(tmp_path)
    probe(run, tmp_path / "probe")
    for row in read_rows(tmp_path / "probe"):
        for arm in ("shuffle_informative", "shuffle_structural", "shuffle_all"):
            assert all(r["moved"] == 0 and r["pred"] == row["pred"] and r["prob"] == pytest.approx(row["base_prob"])
                       for r in row["arms"][arm])


def test_probe_is_deterministic(tmp_path):
    run = train_run(tmp_path)
    probe(run, tmp_path / "a")
    probe(run, tmp_path / "b")
    assert (tmp_path / "a" / "visits.jsonl").read_text() == (tmp_path / "b" / "visits.jsonl").read_text()


def test_probe_refuses_a_run_trained_on_rewired_edges_and_an_occupied_output(tmp_path):
    # training-time rewiring needs --no-edge-payload, which only the black-box clinical_gnn accepts
    rewired = train_run(tmp_path, "top2/clinical_seed7", method="clinical_gnn", conv="edge_conditioned",
                        rewire_relation=["co_complaint"], no_edge_payload=True)
    with pytest.raises(ValueError, match="rewired"):
        probe(rewired, tmp_path / "x")


def test_probe_refuses_an_occupied_output_directory(tmp_path):
    run = train_run(tmp_path)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x")
    with pytest.raises(FileExistsError):
        probe(run, tmp_path / "busy")


def test_plan_only_prints_and_missing_runs_stop(tmp_path, capsys):
    train_run(tmp_path)
    runs = str(tmp_path / "runs")
    assert rs.main(["run", "--runs", runs, "--out", str(tmp_path / "out"), "--ks", "2", "--methods", "gsat",
                    "--seeds", "7"]) == 0
    assert "nothing executed" in capsys.readouterr().out and not (tmp_path / "out").exists()
    with pytest.raises(SystemExit, match="missing"):
        rs.main(["run", "--runs", runs, "--out", str(tmp_path / "out"), "--ks", "2", "--methods", "gsat",
                 "--seeds", "7", "2025"])


def test_run_and_report_end_to_end(tmp_path):
    train_run(tmp_path)
    runs, out, report = str(tmp_path / "runs"), tmp_path / "out", tmp_path / "report"
    assert rs.main(["run", "--runs", runs, "--out", str(out), "--ks", "2", "--methods", "gsat", "--seeds", "7",
                    "--sample-patients", "0", "--draws", "2", "--execute"]) == 0
    assert (out / "top2" / "gsat_seed7" / "relation_probe.json").is_file()
    assert rs.main(["report", "--probes", str(out), "--out", str(report), "--bootstrap", "20"]) == 0
    text = (report / "relation_shuffle.md").read_text()
    assert "**gsat**" in text and "drop_all" in text and "shuffle_informative" in text
    data = json.loads((report / "relation_shuffle.json").read_text())
    assert "2/gsat/drop_all" in data
    with pytest.raises(SystemExit):
        rs.main(["report", "--probes", str(out), "--out", str(report)])


def test_macro_f1_matches_a_hand_computation():
    y, pred = [0, 0, 1, 1, 2], [0, 1, 1, 1, 0]
    # class 0: tp1 fp1 fn1 -> 0.5; class 1: tp2 fp1 fn0 -> 0.8; class 2: tp0 -> 0
    assert rs.macro_f1(y, pred) == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert rs.macro_f1([0, 1], [0, 1]) == 1.0


def test_interval_is_withheld_for_few_patients():
    assert rs.interval({f"p{i}": 1.0 for i in range(9)}, bootstrap=50, seed=1)["ci"] is None
    assert rs.interval({f"p{i}": float(i) for i in range(12)}, bootstrap=50, seed=1)["ci"] is not None


def test_each_shuffle_draw_uses_its_own_seed_and_drop_arms_use_none(tmp_path, monkeypatch):
    import core.tensorize as tensorize
    run = train_run(tmp_path)
    calls = []
    real = tensorize.encode_graph

    def spy(graph, prep, membership, edge_mode="all", drop=(), rewire=(), seed=0, **kwargs):
        calls.append((bool(rewire), bool(drop), seed))
        return real(graph, prep, membership, edge_mode, drop, rewire, seed, **kwargs)

    monkeypatch.setattr(tensorize, "encode_graph", spy)
    probe(run, tmp_path / "probe", max_subjects=1, draws=3)
    shuffle_seeds = sorted({seed for rewired, _, seed in calls if rewired})
    assert shuffle_seeds == [rs.SEED_BASE, rs.SEED_BASE + 1, rs.SEED_BASE + 2]
    assert not any(rewired and dropped for rewired, dropped, _ in calls)


def test_a_model_that_cannot_reproduce_its_predictions_is_refused(tmp_path):
    run = train_run(tmp_path)
    binding = json.loads((run / "binding.json").read_text())
    binding["selected_test"]["prediction_sha256"] = "0" * 64
    (run / "binding.json").write_text(json.dumps(binding))
    (run / "test.npz").unlink()
    with pytest.raises(ValueError, match="REPLAY CHECK FAILED"):
        probe(run, tmp_path / "x")


def test_report_numbers_follow_the_rows():
    def row(patient, y, pred, base, arm_pred, arm_prob, moved=1, edges=10, arm_edges=10):
        return {"sample_id": patient, "patient": patient, "y": y, "pred": pred, "base_prob": base, "edges": edges,
                "arms": {"a": [{"pred": arm_pred, "prob": arm_prob, "moved": moved, "edges": arm_edges}]}}

    rows = [row("p1", 0, 0, 0.9, 1, 0.3),            # was right, now wrong, flips
            row("p2", 1, 0, 0.8, 0, 0.6),            # was wrong, still wrong, no flip
            row("p3", 1, 1, 0.7, 1, 0.7, moved=0),   # unchanged, nothing moved
            row("p4", 1, 0, 0.6, 1, 0.5)]            # was wrong, now right, flips
    patients, per_seed = rs.arm_values({7: ({}, rows)}, "a")
    assert patients["d_acc"] == {"p1": -1.0, "p2": 0.0, "p3": 0.0, "p4": 1.0}
    assert patients["flip"] == {"p1": 1.0, "p2": 0.0, "p3": 0.0, "p4": 1.0}
    assert patients["d_prob"]["p1"] == pytest.approx(-0.6) and patients["d_prob"]["p4"] == pytest.approx(-0.1)
    assert patients["moved"] == {"p1": 1.0, "p2": 1.0, "p3": 0.0, "p4": 1.0}
    assert per_seed[7]["base_acc"] == 0.5 and per_seed[7]["d_acc"] == 0.0
