"""Decoder of explanation records: words, names, case selection, signs, privacy, deletion effects."""
import json
from pathlib import Path

import numpy as np
import pytest

from comparison.explain import decode_examples as de
from comparison.explain import similar_patients as sp
from core import train
from tests.graphxai_fixtures import artifact, run_args

NODES = [{"kind": "patient", "token": "patient", "age": 52.0, "gender": "F", "race": "WHITE",
          "arrival_transport": "AMBULANCE"},
         {"kind": "visit", "token": "visit:index"},
         {"kind": "complaint", "token": "cc:dyspnea"},
         {"kind": "measurement", "token": "lab:50963", "value": 1522.0, "unit": "pg/mL", "level": "high", "flag": "abnormal",
          "ref_lower": 0.0, "ref_upper": 125.0, "time_hours": -0.2333, "scope": "index"},
         {"kind": "analyte", "token": "lab:50963"},
         {"kind": "knowledge", "token": "knowledge:state:troponin_elevated"},
         {"kind": "diagnosis", "token": "dx:icd10:I50"}]


def test_tokens_become_words_and_labs_keep_their_numbers():
    assert de.readable_token("cc:chest pain") == "complaint: chest pain"
    assert de.readable_token("dx:icd10:J44") == "diagnosis ICD-10 J44"
    assert de.readable_token("knowledge:state:hyperkalemia") == "knowledge: state / hyperkalemia"
    assert de.readable_token("lab:50912") == "lab 50912"
    row = de.decode_node(NODES[3], {})
    assert row["label"] == "lab 50963" and row["value"] == "1522 pg/mL" and row["level"] == "high"
    assert row["range"] == "0-125" and row["time_hours"] == -0.2 and row["scope"] == "index"
    assert de.decode_node(NODES[3], {"lab:50963": "NT-proBNP (Blood)"})["label"] == "NT-proBNP (Blood)"


def test_names_from_d_labitems_or_a_token_csv(tmp_path):
    (tmp_path / "labs.csv").write_text("itemid,label,fluid,category\n50963,NT-proBNP,Blood,Chemistry\n50912,Creatinine,,Chemistry\n")
    assert de.load_names(tmp_path / "labs.csv") == {"lab:50963": "NT-proBNP (Blood)", "lab:50912": "Creatinine"}
    (tmp_path / "t.csv").write_text("token,name\ndx:icd10:I50,Heart failure\n")
    assert de.load_names(tmp_path / "t.csv") == {"dx:icd10:I50": "Heart failure"}
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit):
        de.load_names(tmp_path / "bad.csv")
    assert de.load_names(None) == {}


def test_patient_line_gives_sex_and_a_decade_and_nothing_else():
    line = de.patient_line({"nodes": NODES}, 3)
    assert line == "female, age 50s, 3 ED visit(s) in the record"
    assert "WHITE" not in line and "AMBULANCE" not in line and "52" not in line
    assert de.patient_line({"nodes": [{"kind": "patient", "age": 93, "gender": "M"}]}, 1).startswith("male, age 90+")


def test_nearest_to_the_median_with_ties_by_sample_id():
    assert de.pick_nearest_median([("b", 0.9), ("a", 0.5), ("c", 0.7)])[0] == "c"
    assert de.pick_nearest_median([("b", 0.5), ("a", 0.5)])[0] == "a"
    assert de.pick_nearest_median([]) is None


def record(sid, true, pred, p, vector, *, subject="subj-1", signed=None, ig=None):
    detail = {} if signed is None else {"signed_node_contributions": list(signed)}
    return {"sample_id": sid, "subject_id": subject, "num_visits": 2, "true_class_id": true, "prediction_class_id": pred,
            "edge_dependence": {"probability_unmasked": p}, "builtin_available": True,
            "builtin_node_importance": list(vector), "builtin_detail": detail,
            "graphxai": {} if ig is None else {"IntegratedGradExplainer": {
                "status": "success", "node_explanation": {"node_importance": list(ig)}}}}


LABELS = sp.K20_LABELS
HF, SEPSIS, UTI = LABELS.index("Heart failure"), LABELS.index("Sepsis"), LABELS.index("UTI or pyelonephritis")


def methods_for(rows):
    return {"cei_mp2": {r["sample_id"]: r for r in rows}, "gsat": {r["sample_id"]: dict(r) for r in rows}}


def test_selection_one_class_per_group_nearest_median_and_only_visits_every_method_explained():
    v = [0.0] * 7
    rows = [record("a1", HF, HF, 0.60, v), record("a2", HF, HF, 0.80, v), record("a3", HF, HF, 0.99, v),
            record("a4", HF, SEPSIS, 0.55, v),                       # an error: not a correct example
            record("b1", SEPSIS, SEPSIS, 0.7, v), record("b2", UTI, UTI, 0.7, v)]
    by_method = methods_for(rows)
    del by_method["gsat"]["a2"]                                       # gsat did not explain a2
    cases = de.select_cases(by_method, LABELS)
    assert ("correct", "Heart failure", "a1") in cases or ("correct", "Heart failure", "a3") in cases
    assert ("correct", "Heart failure", "a2") not in cases
    classes = {name for _, name, _ in cases}
    assert "Heart failure" in classes and len({n for n in classes if n in ("Sepsis", "UTI or pyelonephritis")}) == 1
    errors = de.select_cases(by_method, LABELS, classes=["Heart failure"], with_errors=True)
    assert ("error", "Heart failure", "a4") in errors
    assert de.select_cases(by_method, LABELS, classes=["Heart failure"]) == [c for c in errors if c[0] == "correct"]


def test_the_reference_model_decides_correctness():
    v = [0.0] * 7
    rows = [record("a1", HF, HF, 0.6, v)]
    by_method = methods_for(rows)
    by_method["gsat"]["a1"]["prediction_class_id"] = SEPSIS           # gsat is wrong, the reference (cei_mp2) is right
    assert de.select_cases(by_method, LABELS, classes=["Heart failure"]) == [("correct", "Heart failure", "a1")]


def test_evidence_rows_rank_by_importance_skip_protected_kinds_and_carry_the_ledger_sign():
    graph = {"nodes": NODES}
    vector = [9.0, 9.0, 0.5, 1.0, 9.0, 0.5, 0.0]                      # patient/visit/analyte are huge but never shown
    signed = [0.0, 0.0, 0.5, -1.0, 0.0, 0.5, 0.0]
    rec = record("a", HF, HF, 0.7, vector, signed=signed, ig=[1, 1, 0.1, 0.2, 1, 0.3, 0.4])
    rows = de.evidence_rows(rec, "builtin", graph, {}, top=10)
    assert [r["token"] for r in rows] == ["lab:50963", "cc:dyspnea", "knowledge:state:troponin_elevated"]   # tie: position
    assert [r["signed_contribution"] for r in rows] == [-1.0, 0.5, 0.5]
    ig = de.evidence_rows(rec, "IntegratedGradExplainer", graph, {}, top=2)
    assert [r["token"] for r in ig] == ["dx:icd10:I50", "knowledge:state:troponin_elevated"]
    assert all(r["signed_contribution"] is None for r in ig)
    assert de.evidence_rows(rec, "GNNExplainer", graph, {}, top=3) == []
    with_effect = de.evidence_rows(rec, "builtin", graph, {}, top=1, effects={3: 0.25})
    assert with_effect[0]["effect"] == 0.25


def test_cli_writes_cases_without_identifiers(tmp_path):
    graph = {"sample_id": "aa11", "nodes": NODES}
    graphs = tmp_path / "graphs.jsonl"
    graphs.write_text(json.dumps(graph) + "\n")
    trained = tmp_path / "trained" / "top20" / "cei_mp2_seed1234"
    trained.mkdir(parents=True)
    (trained / "binding.json").write_text(json.dumps({"label_order": LABELS}))
    names = tmp_path / "labs.csv"
    names.write_text("itemid,label\n50963,NT-proBNP\n")
    for method in ("cei_mp2", "gsat"):
        folder = tmp_path / "expl" / "top20" / f"{method}_seed1234"
        folder.mkdir(parents=True)
        rec = record("aa11", HF, HF, 0.8, [0, 0, 0.5, 2.0, 0, 0.1, 0.2], subject="987654321",
                     signed=[0, 0, 0.5, 2.0, 0, 0.1, 0.2] if method == "cei_mp2" else None,
                     ig=[0, 0, 0.5, 2.0, 0, 0.1, 0.2])
        (folder / "subject_aa11.json").write_text(json.dumps(rec))
    out = tmp_path / "out"
    cli = ["--explanations", str(tmp_path / "expl"), "--graphs", str(graphs), "--runs", str(tmp_path / "trained"),
           "--names", str(names), "--out", str(out)]
    assert de.main(cli) == 0
    text = (out / "examples.md").read_text() + (out / "examples.json").read_text()
    assert "NT-proBNP" in text and "female, age 50s" in text and "Heart failure" in text
    for secret in ("987654321", "WHITE", "AMBULANCE", "52.0"):
        assert secret not in text
    cases = json.loads((out / "examples.json").read_text())
    assert len(cases) == 1 and set(cases[0]["methods"]) == {"cei_mp2", "gsat"}
    assert cases[0]["methods"]["cei_mp2"]["sources"]["builtin"][0]["signed_contribution"] == 2.0
    assert cases[0]["methods"]["gsat"]["sources"]["builtin"][0]["signed_contribution"] is None
    with pytest.raises(SystemExit):
        de.main(cli)


# ------------------------------------------------------------------------------------------- deletion effects
def train_run(tmp_path):
    root, targets, canonical = artifact(tmp_path, count=24, test_rows=4)
    out = tmp_path / "runs" / "top2" / "gsat_seed7"
    train.run(run_args(root, targets, canonical, out))
    return out


def test_deletion_effect_is_the_probability_drop_of_deleting_that_node(tmp_path):
    from core.explain import runner
    run = train_run(tmp_path)
    bundle = runner.load_run_bundle(run, "test")
    splits = runner.rebuild_split(bundle.binding, "test")
    model, _ = runner.build_model(bundle.binding, bundle.state_dict)
    wrapper = runner.build_wrapper(bundle.binding, model)
    data = splits["test"][0]
    sample_id = str(data.sample_id)
    scorer = splits.visit_records([sample_id]).scorer(wrapper.deletion_forward, data)
    predicted = int(scorer.probs(()).view(-1).argmax())
    # fixture graph positions: patient 0, visit 1, complaint 2, vital 3, measurement 4, analyte 5
    effects = de.deletion_effects(run, {sample_id: (predicted, {0, 1, 2, 3, 4, 5})})[sample_id]
    base = float(scorer.probs(()).view(-1)[predicted])
    for position in (2, 3, 4):
        expected = base - float(scorer.probs([position]).view(-1)[predicted])
        assert effects[position] == pytest.approx(expected)
    assert any(abs(effects[p]) > 0 for p in (2, 3, 4))
    for position in (0, 1, 5):
        assert effects[position] is None
