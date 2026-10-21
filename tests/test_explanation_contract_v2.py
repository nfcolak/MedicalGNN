"""Every refusal of the v2 explanation cohort contract, plus the cases it accepts."""
import copy

import pytest

from core.contracts import sample_ids_sha256
from core.explain import explanation_contract_v2 as contract

FOLD_IDS = ["s1", "s2", "s3", "s4"]


def binding(fold="test", ids=FOLD_IDS):
    return {"split_sample_ids_sha256": {fold: sample_ids_sha256(ids)},
            "preprocessing_sha256": "p" * 64}


def cohort(fold="test", explained=("s1", "s2")):
    return contract.build_cohort_v2(binding(fold), fold=fold, fold_sample_ids=FOLD_IDS,
                                    explained_sample_ids=explained)


# ------------------------------------------------------------------ build_cohort_v2
@pytest.mark.parametrize("fold", contract.ALLOWED_FOLDS)
def test_every_allowed_fold_builds_a_cohort_bound_to_the_run(fold):
    built = cohort(fold)
    assert built["schema"] == contract.COHORT_SCHEMA
    assert built["schema_version"] == contract.COHORT_SCHEMA_VERSION
    assert built["fold"] == fold
    assert built["fold_sample_count"] == 4 and built["explained_count"] == 2
    assert built["fold_sample_ids_sha256"] == sample_ids_sha256(FOLD_IDS)
    assert built["preprocessing_sha256"] == "p" * 64
    assert built["is_full_fold"] is False


def test_the_whole_fold_is_recorded_as_a_full_fold():
    assert cohort(explained=FOLD_IDS)["is_full_fold"] is True


@pytest.mark.parametrize("fold", ["train", "val", "", None])
def test_a_fold_outside_the_allowed_set_is_refused(fold):
    with pytest.raises(ValueError, match="explanations may only use folds"):
        contract.build_cohort_v2(binding(), fold=fold, fold_sample_ids=FOLD_IDS,
                                 explained_sample_ids=["s1"])


def test_fold_ids_that_differ_from_the_recorded_hash_are_refused():
    with pytest.raises(ValueError, match="do not match binding.json"):
        contract.build_cohort_v2(binding(), fold="test", fold_sample_ids=FOLD_IDS[::-1],
                                 explained_sample_ids=["s1"])
    with pytest.raises(ValueError, match="do not match binding.json"):
        contract.build_cohort_v2(binding(), fold="test", fold_sample_ids=FOLD_IDS[:3],
                                 explained_sample_ids=["s1"])


def test_a_fold_the_binding_never_recorded_is_refused():
    with pytest.raises(ValueError, match="do not match binding.json"):
        contract.build_cohort_v2(binding("validation"), fold="test", fold_sample_ids=FOLD_IDS,
                                 explained_sample_ids=["s1"])
    with pytest.raises(ValueError, match="do not match binding.json"):
        contract.build_cohort_v2({"preprocessing_sha256": "p" * 64}, fold="test",
                                 fold_sample_ids=FOLD_IDS, explained_sample_ids=["s1"])


def test_duplicate_or_foreign_explained_ids_are_refused():
    with pytest.raises(ValueError, match="unique members of the fold"):
        cohort(explained=("s1", "s1"))
    with pytest.raises(ValueError, match="unique members of the fold"):
        cohort(explained=("s1", "not-in-fold"))


# ------------------------------------------------------------------ validate_cohort_v2
def test_a_cohort_built_for_the_run_validates():
    contract.validate_cohort_v2(cohort(), binding())


@pytest.mark.parametrize("key,value", [("schema", "something.else"), ("schema_version", 1)])
def test_a_cohort_of_another_schema_or_version_is_refused(key, value):
    broken = {**cohort(), key: value}
    with pytest.raises(ValueError, match="not a 'medgnn.explanation_cohort' version 2 cohort"):
        contract.validate_cohort_v2(broken, binding())


def test_a_cohort_with_a_fold_outside_the_allowed_set_is_refused():
    broken = {**cohort(), "fold": "train"}
    with pytest.raises(ValueError, match="cohort fold must be one of"):
        contract.validate_cohort_v2(broken, binding())


def test_a_cohort_whose_fold_hash_differs_from_the_binding_is_refused():
    broken = {**cohort(), "fold_sample_ids_sha256": "0" * 64}
    with pytest.raises(ValueError, match="cohort fold hash differs"):
        contract.validate_cohort_v2(broken, binding())
    other_run = binding()
    other_run["split_sample_ids_sha256"]["test"] = sample_ids_sha256(FOLD_IDS[:2])
    with pytest.raises(ValueError, match="cohort fold hash differs"):
        contract.validate_cohort_v2(cohort(), other_run)


def test_a_cohort_whose_preprocessing_hash_differs_from_the_binding_is_refused():
    other_run = copy.deepcopy(binding())
    other_run["preprocessing_sha256"] = "q" * 64
    with pytest.raises(ValueError, match="cohort preprocessing hash differs"):
        contract.validate_cohort_v2(cohort(), other_run)


def test_a_cohort_with_repeated_ids_or_a_wrong_count_is_refused():
    repeated = {**cohort(), "explained_sample_ids": ["s1", "s1"]}
    with pytest.raises(ValueError, match="not unique or the count is wrong"):
        contract.validate_cohort_v2(repeated, binding())
    miscounted = {**cohort(), "explained_count": 3}
    with pytest.raises(ValueError, match="not unique or the count is wrong"):
        contract.validate_cohort_v2(miscounted, binding())


# ------------------------------------------------------------------ v1 node-explanation schema
class _Linear(__import__("torch").nn.Module):
    """One-graph toy model: logits = sum of the node rows times a fixed matrix."""

    def __init__(self):
        import torch

        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [0.5, 2.0], [-0.3, 0.2]]))

    def forward(self, x, edge_index, batch=None, **kwargs):
        del edge_index, batch, kwargs
        return (x.sum(dim=0) @ self.weight)[None]


def _v1_record(**build):
    import torch

    from core.explain import explanation_contract as v1

    torch.manual_seed(3)
    x = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    model = _Linear().eval()
    explanation = v1.build_node_explanation(model, x, edge_index,
                                            batch=torch.zeros(6, dtype=torch.long), **build)
    prediction = explanation["target"]["class_id"]
    record = {"schema": v1.EXPLANATION_SCHEMA, "schema_version": 1, "method": "gsat",
              "subject_id": "s1", "topology": "star", "seed": 0, "true_class_id": prediction,
              "prediction_class_id": prediction, "attribution_method": "input_gradient",
              "node_explanation": explanation}
    return v1, record


def test_a_v1_record_built_with_the_v1_fidelity_passes_the_v1_validator():
    v1, record = _v1_record(v1_fidelity=True)
    assert set(record["node_explanation"]) == set(v1.EXPLANATION_METRIC_KEYS)
    assert v1.validate_explanation_record(record, method="gsat", topology="star", seed=0) == "s1"


def test_a_record_without_the_v1_fidelity_is_refused_with_the_reason():
    v1, record = _v1_record()
    assert "fidelity_plus" not in record["node_explanation"]
    with pytest.raises(ValueError, match="v1_fidelity=True"):
        v1.validate_explanation_record(record, method="gsat", topology="star", seed=0)


def test_the_historical_cei_export_schema_accepts_a_v1_fidelity_explanation():
    from cei.studies.cei_graphxai import _validate_algorithm_record

    # without provenance the validator stops after the schema checks, at the provenance check
    _, record = _v1_record(v1_fidelity=True)
    with pytest.raises(ValueError, match="provenance is missing"):
        _validate_algorithm_record("GradExplainer", {"status": "success",
                                                     "node_explanation": record["node_explanation"]})
    _, plain = _v1_record()
    with pytest.raises(ValueError, match="genuine node explanation schema"):
        _validate_algorithm_record("GradExplainer", {"status": "success",
                                                     "node_explanation": plain["node_explanation"]})
