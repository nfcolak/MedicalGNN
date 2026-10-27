"""Similar-patient consistency: items, pair choice from inputs, agreement on shared items, report."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from comparison.explain import similar_patients as sp

ELIGIBLE = ["measurement", "complaint", "vital", "diagnosis"]
COMMON = [f"lab:{i}|high|index" for i in range(8)]


def visit(sid, subject, group, items):
    return {"sample_id": sid, "subject_id": subject, "group": group, "items": set(items)}


def test_class_groups_match_the_graph_type_analysis():
    source = Path(sp.__file__).parent / "graph_types" / "step3_clinical.py"
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CLASS_GROUPS":
            assert ast.literal_eval(node.value) == sp.CLASS_GROUPS
            return
    pytest.fail("CLASS_GROUPS not found in step3_clinical.py")


def test_items_carry_token_level_and_scope_and_only_eligible_kinds(tmp_path):
    path = tmp_path / "graphs.jsonl"
    nodes = [{"kind": "patient", "token": "patient"},
             {"kind": "measurement", "token": "lab:1", "level": "high", "scope": "index"},
             {"kind": "complaint", "token": "cc:dyspnea"},
             {"kind": "knowledge", "token": "knowledge:state:x"}]
    path.write_text(json.dumps({"sample_id": "aa", "nodes": nodes}) + "\n")
    read = sp.read_nodes(path, ["aa"])["aa"]
    assert sp.evidence_items(read, set(ELIGIBLE)) == {1: "lab:1|high|index", 2: "cc:dyspnea"}


def test_pairs_are_same_group_other_patients_nearest_first_and_baselines_exclude_neighbours():
    a = visit("a", "p1", "cardiac", COMMON)
    near = visit("n", "p2", "cardiac", COMMON[:7])
    far = visit("f", "p3", "cardiac", COMMON[:5] + ["x1", "x2", "x3", "x4"])
    own = visit("o", "p1", "cardiac", COMMON)                        # same patient: never paired
    other = visit("g", "p4", "infection", COMMON)
    few = visit("e", "p5", "cardiac", COMMON[:4])                    # fewer than MIN_SHARED shared items
    pairs = sp.choose_pairs([a, near, far, own, other, few], seed=1)
    similar = [o for x, o in pairs["similar"] if x == "a"]
    assert similar == ["n", "f"] and "o" not in similar and "e" not in similar and "g" not in similar
    assert all(o == "g" for x, o in pairs["other_group"] if x == "a")
    assert not {o for x, o in pairs["same_group"] if x == "a"} & set(similar)


def test_a_visit_without_similar_patients_gets_no_pairs():
    pairs = sp.choose_pairs([visit("a", "p1", "cardiac", COMMON), visit("b", "p2", "infection", COMMON)])
    assert pairs["similar"] == [] and pairs["same_group"] == [] and pairs["other_group"] == []


def test_pair_choice_is_deterministic_and_baselines_are_capped():
    visits = [visit(f"v{i:02d}", f"p{i}", "cardiac" if i % 2 else "infection", COMMON) for i in range(30)]
    first, second = sp.choose_pairs(visits, seed=7), sp.choose_pairs(visits, seed=7)
    assert first == second
    per_anchor = [o for x, o in first["other_group"] if x == "v01"]
    assert len(per_anchor) == sp.BASELINE_PAIRS and len(set(per_anchor)) == len(per_anchor)


def record(sid, subject, vector, group_class=9):
    return {"sample_id": sid, "subject_id": subject, "true_class_id": group_class, "builtin_available": True,
            "builtin_node_importance": list(vector), "graphxai": {},
            "fidelity_v2": {"builtin": {"fidelity_plus": {}}},
            "fidelity_v2_intervention": {"eligible_node_kinds": ELIGIBLE}}


def items_for(order):
    return {i: item for i, item in enumerate(order)}


def test_agreement_uses_shared_items_only_and_maximum_over_repeated_items():
    shared = COMMON[:6]
    items_a = items_for(shared + ["only_a"])
    items_b = items_for(shared[::-1] + ["only_b", shared[0]])      # different order, one repeated item
    rec_a = record("a", "p1", list(range(1, 8)))
    rec_b = record("b", "p2", [6, 5, 4, 3, 2, 1, 999, 0.1])         # the item only b has is huge
    vec_a = sp.item_vector(rec_a, "builtin", items_a)
    vec_b = sp.item_vector(rec_b, "builtin", items_b)
    assert vec_b[shared[0]] == 1 and "only_b" in vec_b             # repeated item keeps its largest value (1, not 0.1)
    out = sp.run_patient_values([rec_a, rec_b], "builtin", {"similar": [("a", "b")], "same_group": [], "other_group": []},
                                {"a": items_a, "b": items_b})
    # vector a ascends over `shared`; vector b is [6..1] over shared[::-1], i.e. also ascending over `shared`
    assert out["similar"]["rank_correlation"]["p1"] == pytest.approx(1.0)
    assert out["similar"]["overlap_cut"]["p1"] == 1.0


def test_pair_with_an_unexplained_visit_is_skipped():
    rec_a = record("a", "p1", list(range(1, 8)))
    out = sp.run_patient_values([rec_a], "builtin", {"similar": [("a", "missing")], "same_group": [], "other_group": []},
                                {"a": items_for(COMMON[:7]), "missing": items_for(COMMON[:7])})
    assert out["similar"]["lift"] == {}


def test_end_to_end_cli_similar_pairs_agree_more_than_other_group(tmp_path):
    rng = np.random.default_rng(0)
    graphs, runs = tmp_path / "graphs.jsonl", tmp_path / "runs"
    lines, recs = [], []
    for i in range(24):
        group_class = 9 if i < 12 else 15                           # cardiac / infection
        nodes = [{"kind": "patient", "token": "patient"}] + [{"kind": "measurement", "token": f"lab:{j}", "level": "high",
                                                              "scope": "index"} for j in range(10)]
        sid = f"{i:02x}"
        lines.append(json.dumps({"sample_id": sid, "nodes": nodes}))
        base = np.arange(10, dtype=float) if group_class == 9 else np.arange(10, dtype=float)[::-1]
        recs.append(record(sid, f"p{i}", [0.0] + list(base + rng.random(10) * 0.1), group_class))
    graphs.write_text("\n".join(lines) + "\n")
    for seed in (1, 2):
        folder = runs / f"m_seed{seed}"
        folder.mkdir(parents=True)
        for r in recs:
            (folder / f"subject_{r['sample_id']}.json").write_text(json.dumps(r))
    out = tmp_path / "out"
    assert sp.main(["--explanations", str(runs), "--graphs", str(graphs), "--out", str(out), "--bootstrap", "50"]) == 0
    data = json.loads((out / "similar_patients.json").read_text())
    similar = data["m/builtin"]["similar"]["rank_correlation"]["mean"]
    other = data["m/builtin"]["other_group"]["rank_correlation"]["mean"]
    assert similar > 0.9 and other < -0.9
    assert data["m/builtin"]["similar_minus_other_group"]["rank_correlation"]["mean"] > 1.5
    with pytest.raises(SystemExit):
        sp.main(["--explanations", str(runs), "--graphs", str(graphs), "--out", str(out)])
