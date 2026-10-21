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
