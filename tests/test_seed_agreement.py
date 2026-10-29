"""Seed agreement of explanations: overlap, chance, rank correlation, grouping and the report."""
import json

import numpy as np
import pytest

from comparison.explain import seed_agreement as sa

KINDS = ["patient", "visit"] + ["measurement"] * 20 + ["knowledge"] * 3
ELIGIBLE = ["measurement", "complaint", "vital", "diagnosis"]


def record(sample_id, subject, vector, *, builtin=True):
    return {"sample_id": sample_id, "subject_id": subject, "builtin_available": builtin,
            "builtin_node_importance": list(vector), "graphxai": {},
            "fidelity_v2": {"builtin": {"fidelity_plus": {}}},
            "fidelity_v2_intervention": {"eligible_node_kinds": ELIGIBLE}}


def vector(seed):
    return np.random.default_rng(seed).random(len(KINDS)) + 0.01


def test_identical_vectors_agree_completely():
    v = np.arange(1, 21, dtype=float)
    out = sa.pair_agreement(v, v.copy(), "s")
    assert out["overlap_cut"] == 1.0 and out["overlap_k5"] == 1.0
    assert out["rank_correlation"] == pytest.approx(1.0)
    assert out["chance_cut"] == pytest.approx(4 / 20) and out["chance_k5"] == pytest.approx(5 / 20)


def test_reversed_ranking_has_no_overlap_and_negative_correlation():
    v = np.arange(1, 21, dtype=float)
    out = sa.pair_agreement(v, v[::-1].copy(), "s")
    assert out["overlap_cut"] == 0.0
    assert out["rank_correlation"] == pytest.approx(-1.0)


def test_independent_vectors_sit_near_chance():
    rng = np.random.default_rng(0)
    overlaps = [sa.pair_agreement(rng.random(100), rng.random(100), f"s{i}")["overlap_cut"] for i in range(200)]
    assert np.mean(overlaps) == pytest.approx(0.2, abs=0.03)


def test_all_zero_vector_is_left_out_and_constant_vector_has_no_rank_correlation():
    v = np.arange(1, 11, dtype=float)
    assert sa.pair_agreement(np.zeros(10), v, "s") is None
    out = sa.pair_agreement(np.ones(10), v, "s")
    assert out is not None and out["rank_correlation"] is None


def test_ties_at_the_cut_are_averaged_not_resolved_by_position():
    flat = np.ones(20)
    out = sa.pair_agreement(flat, flat.copy(), "s")
    assert out["overlap_cut"] == pytest.approx(0.2, abs=0.2)
    assert 0.0 < out["overlap_cut"] < 1.0


def test_protected_nodes_are_not_compared():
    base = vector(1)
    other = base.copy()
    other[[0, 1, 22, 23, 24]] = 100.0   # only patient, visit and knowledge nodes differ
    by_seed = {1: [record("a", "p1", base)], 2: [record("a", "p1", other)]}
    rows = sa.visit_values(by_seed, "builtin", {"a": KINDS})
    assert len(rows) == 1 and rows[0]["overlap_cut"] == 1.0


def test_visit_missing_in_one_seed_still_pairs_the_other_two():
    v = vector(2)
    by_seed = {1: [record("a", "p1", v)], 2: [record("a", "p1", v)], 3: []}
    assert len(sa.visit_values(by_seed, "builtin", {"a": KINDS})) == 1


def test_group_runs_reads_method_and_seed_from_folder_name():
    runs = {"top20/cei_mp2_seed1234": [], "top20/cei_mp2_seed7": [], "top20/gsat_seed7": [], "top20/other": []}
    grouped = sa.group_runs(runs)
    assert sorted(grouped["cei_mp2"]) == [7, 1234] and sorted(grouped["gsat"]) == [7] and "other" not in grouped


def test_node_kinds_reads_only_requested_graphs(tmp_path):
    path = tmp_path / "graphs.jsonl"
    path.write_text("\n".join(json.dumps({"sample_id": sid, "nodes": [{"kind": "visit"}, {"kind": k}]})
                              for sid, k in (("aa", "vital"), ("bb", "lab"))) + "\n")
    assert sa.node_kinds(path, ["aa"]) == {"aa": ["visit", "vital"]}


def test_report_and_cli_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    graphs = tmp_path / "graphs.jsonl"
    ids = [f"{i:02x}" for i in range(12)]
    graphs.write_text("\n".join(json.dumps({"sample_id": sid, "nodes": [{"kind": k} for k in KINDS]})
                                for sid in ids) + "\n")
    for seed in (1, 2):
        folder = runs / f"m_seed{seed}"
        folder.mkdir(parents=True)
        for i, sid in enumerate(ids):
            (folder / f"subject_{sid}.json").write_text(json.dumps(record(sid, f"p{i}", vector(i))))
    out = tmp_path / "out"
    assert sa.main(["--explanations", str(runs), "--graphs", str(graphs), "--out", str(out), "--bootstrap", "50"]) == 0
    data = json.loads((out / "seed_agreement.json").read_text())
    assert data["m/builtin"]["overlap_cut"]["mean"] == 1.0 and data["m/builtin"]["overlap_cut"]["n"] == 12
    assert "| builtin | 12 |" in (out / "seed_agreement.md").read_text()
    with pytest.raises(SystemExit):
        sa.main(["--explanations", str(runs), "--graphs", str(graphs), "--out", str(out)])


def test_vector_with_some_zero_importance_is_still_compared():
    v = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    assert sa.pair_agreement(v, v.copy(), "s")["overlap_cut"] == 1.0


def test_visit_without_a_graph_is_left_out_not_a_crash():
    v = vector(3)
    by_seed = {1: [record("a", "p1", v)], 2: [record("a", "p1", v)]}
    assert sa.visit_values(by_seed, "builtin", {}) == []
    assert sa.visit_values(by_seed, "builtin", {"a": KINDS[:-1]}) == []
