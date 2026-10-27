"""Audit report: weight randomisation and stability aggregated from the records' audit blocks."""
import json
from pathlib import Path

import pytest

from comparison.explain import audit_report as ar
from core import train
from core.explain import runner as explain
from tests.graphxai_fixtures import artifact, run_args


def audit(*, weight, draws, changed=False):
    return {"weight_randomisation": {"prediction_changed": changed, "sources": weight},
            "stability": {"sigma": 0.05, "draws": draws}}


def entry(spearman, jaccard):
    return {"spearman": spearman, "top_k_jaccard": jaccard, "k": [3, 3]}


def record(subject, block):
    return {"sample_id": f"s-{subject}", "subject_id": subject, "audit": block}


def test_visit_values_average_the_kept_draws_and_ignore_changed_or_failed_ones():
    block = audit(weight={"builtin": entry(0.1, 0.2), "GNNExplainer": {"status": "failed", "error": "x"}},
                  draws=[{"prediction_changed": False, "sources": {"builtin": entry(0.8, 0.6)}},
                         {"prediction_changed": False, "sources": {"builtin": entry(0.6, 0.4)}},
                         {"prediction_changed": True, "measured_nodes_perturbed": 4,
                          "sources": {"builtin": entry(0.0, 0.0)}},          # must not pull the average down
                         {"status": "failed", "error": "boom"}], changed=True)
    values = ar.visit_values(record("p", block))
    assert values["weight"] == {"builtin": {"spearman": 0.1, "top_k_jaccard": 0.2}}
    assert values["weight_changed"] == 1.0
    assert values["stability"]["builtin"]["spearman"] == pytest.approx(0.7)
    assert values["stability"]["builtin"]["top_k_jaccard"] == pytest.approx(0.5)
    assert values["stability_changed"] == pytest.approx(1 / 3)       # 1 of the 3 draws that carry a verdict


def test_a_constant_explanation_has_no_spearman_and_is_not_counted_as_zero():
    block = audit(weight={"builtin": entry(None, 0.5)}, draws=[])
    rows = [record(f"p{i}", block) for i in range(12)]
    assert ar.per_patient(rows, lambda v: v["weight"]["builtin"]["spearman"]) == {}
    assert len(ar.per_patient(rows, lambda v: v["weight"]["builtin"]["top_k_jaccard"])) == 12


def test_report_without_audited_visits_says_so(tmp_path):
    folder = tmp_path / "expl" / "top2" / "gsat_seed7"
    folder.mkdir(parents=True)
    (folder / "subject_a.json").write_text(json.dumps({"sample_id": "a", "subject_id": "p"}))
    out = tmp_path / "out"
    assert ar.main(["--explanations", str(tmp_path / "expl"), "--out", str(out)]) == 0
    assert "without `--audit-subjects`" in (out / "audit_report.md").read_text()


def test_report_numbers_and_cli(tmp_path):
    for seed in (1, 2):
        folder = tmp_path / "expl" / "top2" / f"m_seed{seed}"
        folder.mkdir(parents=True)
        for i in range(12):
            block = audit(weight={"builtin": entry(0.05, 0.1)},
                          draws=[{"prediction_changed": False, "sources": {"builtin": entry(0.9, 0.8)}}])
            (folder / f"subject_{i}.json").write_text(json.dumps(record(f"p{i}", block)))
    out = tmp_path / "out"
    assert ar.main(["--explanations", str(tmp_path / "expl"), "--out", str(out), "--bootstrap", "50"]) == 0
    data = json.loads((out / "audit_report.json").read_text())["m"]["builtin"]
    assert data["weight_spearman"]["mean"] == pytest.approx(0.05) and data["weight_spearman"]["n"] == 12
    assert data["stability_top_k_jaccard"]["mean"] == pytest.approx(0.8)
    with pytest.raises(SystemExit):
        ar.main(["--explanations", str(tmp_path / "expl"), "--out", str(out)])


def test_the_report_reads_the_audit_block_a_real_runner_writes(tmp_path):
    root, targets, canonical = artifact(tmp_path, count=24, test_rows=4)
    run_dir = tmp_path / "runs" / "top2" / "gsat_seed7"
    train.run(run_args(root, targets, canonical, run_dir))
    output = tmp_path / "expl" / "top2" / "gsat_seed7"
    manifest = explain.run_explanations(run_dir, output, max_subjects=2, steps=2, epochs=3, audit_subjects=2,
                                        stability_draws=1, random_repeats=2, tie_draws=2)
    assert manifest["failed_count"] == 0
    records = [json.loads((output / f).read_text()) for f in manifest["record_files"]]
    assert all("audit" in r for r in records)
    values = ar.visit_values(records[0])
    assert values["weight_changed"] is not None and values["stability_changed"] is not None
    assert {"builtin", "GradExplainer", "IntegratedGradExplainer", "GNNExplainer"} <= set(values["weight"])
    lines, data = ar.audit_report({"top2/gsat_seed7": records}, bootstrap=20, seed=1)
    assert "gsat" in data and any("weight randomisation" in line.lower() for line in lines)
    assert "IntegratedGradExplainerZeroBaseline" not in "\n".join(lines) and "IntegratedGradExplainer" in data["gsat"]
