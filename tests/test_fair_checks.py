"""Matched evidence budgets and the ProtGNN prototype check (core/explain/fair_checks.py)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from core.explain import fair_checks
from core.explain import fidelity_v2 as fv2
from core.explain import runner as explain
from tests.test_explain_runner import subject_kwargs, subject_setup, train_one


@pytest.fixture(scope="module")
def gsat_scorer(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("fair_gsat")
    run_dir = train_one(tmp_path, "run_gsat", method="gsat", conv=None)
    binding, splits, wrapper, records = subject_setup(run_dir)
    data = splits["test"][0]
    wrapper.set_context(explain._single_graph_batch(data))
    scorer = records.scorer(wrapper.deletion_forward, data)
    return scorer


def eligible_indices(scorer):
    return np.flatnonzero(fv2.removable_node_mask(scorer.node_type).numpy())


def test_matched_budget_scores_every_source_with_the_direct_fidelity_calls(gsat_scorer):
    scorer = gsat_scorer
    eligible = eligible_indices(scorer)
    rng = np.random.default_rng(0)
    dense = np.zeros(scorer.node_type.numel())
    dense[eligible] = rng.uniform(0.1, 1.0, size=eligible.size)
    target = int(scorer.probs(()).argmax())
    result = fair_checks.matched_budget(scorer, {"dense": dense}, target, ks=(1, 2), repeats=3, seed=7)
    assert result["ks"] == [1, 2]
    for k in (1, 2):
        row = result["sources"]["dense"]["by_k"][str(k)]
        plus = fv2.fidelity_plus_v2(scorer, dense, target, k=k, seed=7)
        minus = fv2.fidelity_minus_v2(scorer, dense, target, k=k, seed=7)
        assert plus["k"] == k and minus["k"] == k
        assert row["plus"] == pytest.approx(plus["prob"]) and row["minus"] == pytest.approx(minus["prob"])
        assert row["plus_minus_random"] == pytest.approx(row["plus"] - row["plus_random_mean"])
        assert row["minus_minus_random"] == pytest.approx(row["minus"] - row["minus_random_mean"])
    reference = fv2.random_reference(scorer, target, [1, 2], repeats=3, seed=7)
    assert result["sources"]["dense"]["by_k"]["2"]["plus_random_mean"] == pytest.approx(
        float(reference._draws("deletion", 2).mean()))
    assert result["sources"]["dense"]["by_k"]["2"]["minus_random_mean"] == pytest.approx(
        float(reference._draws("insertion", 2).mean()))


def test_a_source_is_not_padded_past_its_support(gsat_scorer):
    scorer = gsat_scorer
    eligible = eligible_indices(scorer)
    assert eligible.size >= 3, "the fixture visit must have at least three eligible nodes"
    sparse = np.zeros(scorer.node_type.numel())
    sparse[eligible[:2]] = 1.0                      # two nodes with importance
    target = int(scorer.probs(()).argmax())
    result = fair_checks.matched_budget(scorer, {"sparse": sparse}, target, ks=(1, 2, 3), repeats=3, seed=1)
    rows = result["sources"]["sparse"]["by_k"]
    assert result["sources"]["sparse"]["support"] == 2
    assert "plus" in rows["1"] and "plus" in rows["2"]
    assert rows["3"] == {"status": "beyond_support", "support": 2}


def test_all_zero_and_failing_sources_are_reported_not_hidden(gsat_scorer):
    scorer = gsat_scorer
    target = int(scorer.probs(()).argmax())
    zero = np.zeros(scorer.node_type.numel())
    wrong_length = np.ones(3)
    result = fair_checks.matched_budget(scorer, {"zero": zero, "bad": wrong_length}, target, ks=(1,), repeats=2, seed=1)
    assert result["sources"]["zero"]["status"] == "no_eligible_evidence"
    assert result["sources"]["bad"]["status"] == "failed"


def test_ks_above_the_number_of_eligible_nodes_are_left_out(gsat_scorer):
    scorer = gsat_scorer
    count = int(fv2.removable_node_mask(scorer.node_type).sum())
    target = int(scorer.probs(()).argmax())
    result = fair_checks.matched_budget(scorer, {}, target, ks=(1, count, count + 1, count + 50), repeats=2, seed=1)
    assert result["ks"] == sorted({1, count})


@pytest.fixture(scope="module")
def protgnn_setup(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("fair_protgnn")
    run_dir = train_one(tmp_path, "run_protgnn", method="protgnn", conv=None)
    return subject_setup(run_dir)


def protgnn_record(protgnn_setup, index=0, **extra):
    binding, splits, wrapper, records = protgnn_setup
    return explain.explain_subject(wrapper, splits["test"][index], **subject_kwargs(binding, records, **extra))


def test_records_carry_the_matched_budget_and_the_prototype_check(protgnn_setup):
    record = protgnn_record(protgnn_setup)
    assert set(record["matched_budget"]["ks"]) <= set(fair_checks.BUDGET_KS)
    assert "builtin" in record["matched_budget"]["sources"]
    check = record["protgnn_prototype_check"]
    detail = record["builtin_detail"]
    assert check["subgraph_size"] == len(detail["subgraph_nodes"])
    assert sum(check["subgraph_kinds"].values()) == check["subgraph_size"]
    assert check["prototype_index"] == detail["prototype_index"]
    assert check["explained_class"] == record["prediction_class_id"]
    assert check["subgraph_removable"] <= check["subgraph_size"]


def test_prototype_check_base_is_the_untouched_visit(protgnn_setup):
    binding, splits, wrapper, records = protgnn_setup
    record = protgnn_record(protgnn_setup)
    check = record["protgnn_prototype_check"]
    if check["status"] != "ok":
        pytest.skip("the tiny model's subgraph has no removable node on this visit")
    data = splits["test"][0]
    scorer = records.scorer(wrapper.deletion_forward, data)
    target = record["prediction_class_id"]
    assert check["base"]["probability"] == pytest.approx(float(scorer.probs(())[0, target]), abs=1e-5)
    adapter = wrapper.adapter
    weight = float(adapter.prototype_classifier.weight[target, check["prototype_index"]])
    assert check["base"]["contribution"] == pytest.approx(weight * check["base"]["activation"], rel=1e-5)
    for scenario in ("deleted", "kept_only"):
        assert 0.0 <= check[scenario]["probability"] <= 1.0


def test_prototype_check_deletes_and_keeps_the_same_numbers_of_nodes(protgnn_setup, monkeypatch):
    binding, splits, wrapper, records = protgnn_setup
    data = splits["test"][0]
    wrapper.set_context(explain._single_graph_batch(data))
    scorer = records.scorer(wrapper.deletion_forward, data)
    target = int(scorer.probs(()).argmax())
    detail = wrapper.builtin_detail()
    seen: list[list[int]] = []
    real = fair_checks.delete_from_record

    def spy(graph, membership, deleted, **kwargs):
        seen.append(list(deleted))
        return real(graph, membership, deleted, **kwargs)

    monkeypatch.setattr(fair_checks, "delete_from_record", spy)
    check = fair_checks.protgnn_prototype_check(wrapper.adapter, scorer, detail, target, repeats=3, seed=5)
    if check["status"] != "ok":
        pytest.skip("no removable node in the subgraph")
    eligible = set(eligible_indices(scorer).tolist())
    inside = [n for n in detail["subgraph_nodes"] if n in eligible]
    size, total = len(inside), len(eligible)
    assert seen[0] == [] and seen[1] == sorted(inside)
    assert seen[2] == sorted(eligible - set(inside))                     # kept only the subgraph
    random_calls = seen[3:]
    assert len(random_calls) == 6                                         # 3 deleted + 3 kept draws
    for position, call in enumerate(random_calls):
        assert len(call) == (size if position % 2 == 0 else total - size)
        assert set(call) <= eligible


def test_a_subgraph_without_removable_nodes_is_stated(protgnn_setup):
    binding, splits, wrapper, records = protgnn_setup
    data = splits["test"][0]
    wrapper.set_context(explain._single_graph_batch(data))
    scorer = records.scorer(wrapper.deletion_forward, data)
    structural = [i for i in range(scorer.node_type.numel())
                  if not bool(fv2.removable_node_mask(scorer.node_type)[i])][:2]
    detail = dict(wrapper.builtin_detail(), subgraph_nodes=structural)
    check = fair_checks.protgnn_prototype_check(wrapper.adapter, scorer, detail, 0, repeats=2, seed=1)
    assert check["status"] == "subgraph_has_no_removable_node"
    assert check["subgraph_removable"] == 0 and check["subgraph_size"] == len(structural)


def test_the_records_stay_json_serialisable(protgnn_setup):
    record = protgnn_record(protgnn_setup)
    json.dumps(record["matched_budget"])
    json.dumps(record["protgnn_prototype_check"])
