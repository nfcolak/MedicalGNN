"""Failure-type analysis (analysis/failure_types): strata by class name, sample-size rules, patient-cluster bootstrap."""
import csv
import os

import numpy as np
import pytest

os.environ.setdefault("FT_K", "20")
os.environ.setdefault("FT_EXPLAIN_ROOT", "/nonexistent")
os.environ.setdefault("FT_SUMMARY_DIR", "/nonexistent")
os.environ.setdefault("FT_WORK", "/nonexistent")

from analysis.failure_types import strata_report as sr  # noqa: E402
from comparison.explain import similar_patients as sp  # noqa: E402

KEYS = ("cei_final", "gsat", "graphcare", "protgnn")


def cell(n, est=0.1, lo=0.05, hi=0.15):
    return {"n": n, "patients": n, "est": est, "lo": lo, "hi": hi}


def test_sample_size_rules_of_a_cell():
    assert sr.fmt(cell(9)) == "-"
    assert sr.fmt(cell(10)).endswith("†") and sr.fmt(cell(29)).endswith("†")
    assert sr.fmt(cell(30)) == "+0.100 [+0.050, +0.150]"
    assert sr.fmt(cell(30), bold_if_excluded=True).startswith("**")                 # interval excludes 0
    assert not sr.fmt(cell(30, 0.0, -0.05, 0.05), bold_if_excluded=True).startswith("**")
    assert not sr.fmt(cell(29), bold_if_excluded=True).startswith("**")              # too few visits to interpret
    assert sr.fmt({"n": 50, "patients": 5, "est": None, "lo": None, "hi": None}) == "-"


def test_stratified_mean_sizes_and_missing_values():
    patient = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    values = np.array([1.0, 3.0, 5.0, np.nan, 2.0, 2.0, 4.0, 6.0])
    ax = sr.axis("a", "t", ["first half", "second half"], [np.arange(8) < 4, np.arange(8) >= 4])
    first, second = sr.Bootstrap(patient).stratified(values, ax)
    assert first["n"] == 3 and first["patients"] == 2 and first["est"] == pytest.approx(3.0)       # nan left out
    assert second["n"] == 4 and second["patients"] == 2 and second["est"] == pytest.approx(3.5)
    assert first["lo"] <= first["est"] <= first["hi"]
    again = sr.Bootstrap(patient).stratified(values, ax)
    assert again == [first, second]                                                                   # seeded


def test_the_interval_follows_patients_not_visits():
    # 10 patients with 5 identical visits each: the interval is that of 10 patients, far wider than of 50 visits
    patient = np.repeat(np.arange(10), 5)
    values = np.repeat(np.arange(10, dtype=float), 5)
    ax = sr.axis("a", "t", ["all"], [np.ones(50, bool)])
    cell_ = sr.Bootstrap(patient).stratified(values, ax)[0]
    naive = 1.96 * values.std(ddof=1) / np.sqrt(50)
    assert (cell_["hi"] - cell_["lo"]) / 2 > 1.5 * naive


def test_ok_in_at_least_two_seeds_and_mean_over_ok_seeds(tmp_path):
    path = tmp_path / "scores.csv"
    rows = [  # visit 0: ok in 3 seeds; visit 1: ok in 1 seed; visit 2: ok in 2 seeds (the third not ok)
        (0, 1, 0.1), (0, 1, 0.2), (0, 1, 0.3), (1, 1, 0.9), (1, 0, ""), (1, 0, ""), (2, 1, 0.2), (2, 1, 0.4), (2, 0, "")]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["visit_idx", "key", "seed", "source", "ok", "deletion_mean_change_vs_random"])
        for i, (visit, ok, value) in enumerate(rows):
            writer.writerow([visit, "gsat", i % 3, "IG", ok, value])
    values = sr.load_values(path, 3, {"gsat"})[("gsat", "IG", "del")]
    assert values[0] == pytest.approx(0.2) and np.isnan(values[1]) and values[2] == pytest.approx(0.3)


def features(names, n=None):
    n = n or len(names)
    rng = np.random.default_rng(0)
    feat = {"true_class": np.array(names), "visit_idx": np.arange(n), "patient_idx": np.arange(n) // 2,
            "num_visits": rng.integers(1, 6, n).astype(float), "n_nodes": rng.integers(30, 200, n).astype(float),
            "n_edges": rng.integers(0, 300, n).astype(float), "n_kind_measurement": rng.integers(0, 20, n).astype(float),
            "n_prior_dx": rng.integers(0, 3, n).astype(float)}
    for key in KEYS:
        feat["conf_" + key] = rng.uniform(0.3, 0.99, n)
        feat["correct_" + key] = (rng.random(n) < 0.7).astype(float)
        feat["pred_agree_" + key] = (rng.random(n) < 0.8).astype(float)
    return feat


def test_groups_follow_class_names_and_unknown_classes_form_one_other_stratum():
    names = ["Heart failure"] * 3 + ["Sepsis"] * 2 + ["Dehydration"] * 2        # Dehydration is a K=30 class, no group
    ax = {a["key"]: a for a in sr.shared_axes(features(names))}["group"]
    labels = dict(zip(ax["labels"], ax["masks"]))
    assert labels["cardiac"].sum() == 3 and labels["infection"].sum() == 2 and labels["other (no group)"].sum() == 2
    assert ax["labels"][-1] == "other (no group)"
    assert set(sp.GROUP_OF_NAME) >= {"Heart failure", "Sepsis"} and "Dehydration" not in sp.GROUP_OF_NAME


def test_axes_cover_every_visit_exactly_once():
    feat = features(["Heart failure", "Sepsis", "Anemia", "Head injury"] * 30)
    for ax in sr.shared_axes(feat) + sr.method_axes(feat, "gsat") + sr.pair_axes(feat, "cei_final", "gsat"):
        total = np.sum(ax["masks"], axis=0)
        assert (total == 1).all(), ax["key"]


def test_end_to_end_report_finds_a_planted_difference_and_withholds_small_cells(tmp_path, monkeypatch):
    names = (["Heart failure"] * 160 + ["Sepsis"] * 8 + ["Head injury"] * 100)
    feat = features(names)
    n = len(names)
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    cols = [c for c in feat if c != "true_class"]
    with open(data / "visit_features.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["visit_idx", "patient_idx", "true_class_id", "true_class"] + [c for c in cols if c not in ("visit_idx", "patient_idx")])
        for i in range(n):
            writer.writerow([i, int(feat["patient_idx"][i]), 0, names[i]] + [feat[c][i] for c in cols if c not in ("visit_idx", "patient_idx")])
    rng = np.random.default_rng(1)
    with open(data / "visit_scores.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["visit_idx", "key", "seed", "source", "ok", "deletion_mean_change_vs_random"])
        for i in range(n):
            for key in KEYS:
                for seed in range(3):
                    for source in sr.SOURCES:
                        value = rng.normal(0.2, 0.02)
                        if key == "cei_final" and names[i] == "Heart failure":
                            value += 0.1                                  # planted: CEI-GNN better only on cardiac
                        writer.writerow([i, key, seed, source, 1, value])
    monkeypatch.setattr(sr, "WORK", tmp_path)
    monkeypatch.setattr(sr, "K", 20)
    assert sr.main() == 0
    text = (out / "strata-report.md").read_text()
    section = text.split("## 3.")[1]
    cardiac = [l for l in section.split("### CEI-GNN minus GSAT: Diagnosis group")[1].splitlines() if l.startswith("| cardiac")][0]
    other_group = [l for l in section.split("### CEI-GNN minus GSAT: Diagnosis group")[1].splitlines() if l.startswith("| other")]
    assert "**+0.1" in cardiac                                           # planted +0.1, n >= 30, interval excludes 0
    sepsis = [l for l in section.split("### CEI-GNN minus GSAT: True class")[1].splitlines() if l.startswith("| Sepsis")][0]
    assert sepsis.count("-") >= 4 and "**" not in sepsis                 # 8 visits: withheld
    rows = list(csv.DictReader(open(out / "strata-cells.csv")))
    assert {r["table"] for r in rows} == {"score", "score_own", "difference"}
    assert any(r["metric"] == "del" and r["who"] == "CEI-GNN - GSAT" and r["stratum"] == "cardiac" for r in rows)


def test_driver_runs_both_steps_for_every_k_in_its_own_folder_and_stops_on_failure(tmp_path):
    from analysis.failure_types import run as driver
    calls = []

    class Done:
        def __init__(self, code):
            self.returncode = code

    def fake(command, env, cwd):
        calls.append((command[-1], env["FT_K"], env["FT_WORK"], env["FT_FOLD"]))
        return Done(0)

    args = ["--explanations", str(tmp_path / "e"), "--summary", str(tmp_path / "s"), "--work", str(tmp_path / "w")]
    assert driver.main(args + ["--ks", "10,30"], run=fake) == 0
    assert [(c[0].rsplit(".", 1)[-1], c[1]) for c in calls] == [("step1_visit_tables", "10"), ("strata_report", "10"),
                                                                ("step1_visit_tables", "30"), ("strata_report", "30")]
    assert calls[0][2].endswith("w/k10") and calls[2][2].endswith("w/k30") and calls[0][3] == "test"
    calls.clear()
    assert driver.main(args[:-1] + [str(tmp_path / "w2"), "--ks", "20"], run=lambda c, env, cwd: Done(3)) == 3
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    with pytest.raises(SystemExit):
        driver.main(args[:-1] + [str(tmp_path / "busy")], run=fake)
