"""comparison/explain/fair_report.py: the patient is the unit; paired differences use shared patients."""
from __future__ import annotations

import json

import numpy as np
import pytest

from comparison.explain import fair_report as report


def record(patient, visit, **sources):
    """A record with fidelity_v2 entries {source: (fidelity_plus prob, k, k_requested)}."""
    return {
        "subject_id": patient, "sample_id": f"{patient}-{visit}",
        "fidelity_v2": {
            name: {"fidelity_plus": {"prob": prob, "k": k, "k_requested": requested, "removable_node_count": 50},
                   "fidelity_minus": {"prob": 0.1}, "sparsity": {"mass90_sparsity": 0.5},
                   "vs_random": {"deletion_mean_change": {"difference": prob},
                                 "insertion_mean_change": {"difference": -prob}}}
            for name, (prob, k, requested) in sources.items()},
    }


def test_a_patient_counts_once_whatever_its_number_of_visits():
    records = [record("a", 1, s=(0.0, 1, 1)), record("a", 2, s=(0.0, 1, 1)), record("a", 3, s=(0.0, 1, 1)),
               record("b", 1, s=(1.0, 1, 1))]
    values = report.by_patient(records, lambda r: r["fidelity_v2"]["s"]["fidelity_plus"]["prob"])
    assert values == {"a": 0.0, "b": 1.0}                    # not (0 + 0 + 0 + 1) / 4
    assert report.interval(values, bootstrap=50, seed=1)["mean"] == pytest.approx(0.5)


def test_an_interval_is_withheld_below_ten_patients_and_given_from_ten():
    few = {str(i): float(i) for i in range(report.MIN_PATIENTS_FOR_INTERVAL - 1)}
    enough = {str(i): float(i) for i in range(report.MIN_PATIENTS_FOR_INTERVAL)}
    assert report.interval(few, bootstrap=100, seed=1)["ci"] is None
    low, high = report.interval(enough, bootstrap=100, seed=1)["ci"]
    assert low < np.mean(list(enough.values())) < high


def test_the_interval_is_reproducible_and_brackets_the_mean_of_a_known_sample():
    values = {str(i): float(i % 7) for i in range(200)}
    a = report.interval(values, bootstrap=500, seed=3)
    b = report.interval(values, bootstrap=500, seed=3)
    assert a == b
    assert a["ci"][0] < a["mean"] < a["ci"][1]
    values_array = np.asarray(list(values.values()))
    width = 2 * 1.96 * values_array.std(ddof=1) / np.sqrt(values_array.size)    # normal theory, about 0.55
    assert 0.8 * width < a["ci"][1] - a["ci"][0] < 1.2 * width                  # a 95% interval, not 50%


def test_a_paired_difference_uses_only_the_patients_both_sides_have():
    a = {"1": 5.0, "2": 5.0, "3": 100.0}
    b = {"1": 4.0, "2": 4.0, "4": -100.0}
    same = {str(i): 1.0 for i in range(report.MIN_PATIENTS_FOR_INTERVAL)}
    result = report.paired_interval(a, b, bootstrap=50, seed=1)
    assert result["n"] == 2 and result["mean"] == pytest.approx(1.0)
    big_a = {**same, "x": 3.0}
    big_b = {**{k: 0.0 for k in same}, "x": 1.0, "only_b": 50.0}
    paired = report.paired_interval(big_a, big_b, bootstrap=100, seed=1)
    assert paired["n"] == len(same) + 1
    assert paired["mean"] == pytest.approx((len(same) * 1.0 + 2.0) / (len(same) + 1))


def test_the_budget_section_reports_the_cap_by_support():
    records = [record(f"p{i}", 1, dense=(0.5, 10, 10), sparse=(0.0, 1, 10)) for i in range(12)]
    lines, data = report.section_budgets({"run": records}, bootstrap=20, seed=1)
    assert data["run"]["dense"]["capped_share"] == 0.0 and data["run"]["dense"]["k"] == 10
    assert data["run"]["sparse"]["capped_share"] == 1.0 and data["run"]["sparse"]["k"] == 1


def test_the_paired_section_gives_the_difference_between_two_explainers():
    records = [record(f"p{i}", 1, a=(0.6, 10, 10), b=(0.4, 10, 10)) for i in range(12)]
    lines, data = report.section_paired({"run": records}, bootstrap=50, seed=1)
    entry = data["run"]["fidelity+"]["paired"]["a - b"]
    assert entry["mean"] == pytest.approx(0.2) and entry["n"] == 12
    assert data["run"]["fidelity+"]["individual"]["a"]["mean"] == pytest.approx(0.6)


def test_the_same_explainer_is_compared_across_methods_on_shared_patients():
    run_a = [record(f"p{i}", 1, s=(0.9, 10, 10)) for i in range(12)]
    run_b = [record(f"p{i}", 1, s=(0.5, 10, 10)) for i in range(6, 18)]      # 6 shared patients
    lines, data = report.section_paired({"A": run_a, "B": run_b}, bootstrap=50, seed=1)
    entry = data["across_methods"]["fidelity+|s|A - B"]
    assert entry["n"] == 6 and entry["mean"] == pytest.approx(0.4) and entry["ci"] is None


def test_the_matched_budget_section_shows_rows_and_how_many_patients_reach_k():
    def with_budget(patient, support):
        rec = record(patient, 1, s=(0.5, 10, 10))
        rows = {}
        for k in (1, 3):
            rows[str(k)] = ({"status": "beyond_support", "support": support} if k > support else
                            {"plus": 0.5, "minus": 0.1, "plus_random_mean": 0.2, "minus_random_mean": 0.3,
                             "plus_minus_random": 0.3, "minus_minus_random": -0.2})
        rec["matched_budget"] = {"ks": [1, 3], "sources": {"s": {"support": support, "by_k": rows}}}
        return rec

    records = [with_budget(f"p{i}", 2 if i < 4 else 5) for i in range(12)]
    lines, data = report.section_matched({"run": records}, bootstrap=20, seed=1)
    assert data["run|s|k1"]["reached_share"] == 1.0
    assert data["run|s|k3"]["reached_share"] == pytest.approx(8 / 12)
    assert data["run|s|k1"]["plus_minus_random"]["mean"] == pytest.approx(0.3)
    assert any("reached by 67% of patients" in line for line in lines)


def test_the_cli_refuses_a_filled_output_directory(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "subject_a.json").write_text(json.dumps(record("a", 1, s=(0.5, 10, 10))))
    out = tmp_path / "out"
    out.mkdir()
    (out / "something").write_text("x")
    with pytest.raises(SystemExit):
        report.main(["--explanations", str(folder), "--out", str(out)])
