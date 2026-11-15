"""Synthetic split ancestry, determinism and export acceptance tests."""

from itertools import permutations

import pytest

from openmed.training.synthetic.split_planner import (
    SyntheticSplitPlan,
    SyntheticSplitRecord,
    plan_synthetic_splits,
)


def digest(value):
    return f"{value:064x}"


def row(n, *, trajectory=None, lineage=None, parents=(), strata=(), synthetic=True):
    return SyntheticSplitRecord(
        digest(n),
        digest(n if trajectory is None else trajectory),
        digest(n if lineage is None else lineage),
        tuple(map(digest, parents)),
        strata,
        synthetic,
    )


def plan(rows, **kwargs):
    options = dict(fractions={"train": 0.5, "test": 0.5}, seed=7)
    options.update(kwargs)
    return plan_synthetic_splits(rows, **options)


def test_all_modalities_and_derivation_descendants_remain_together():
    records = [row(1), row(2, trajectory=1), row(3, parents=(2,)), row(4, lineage=3)]
    result = plan(records + [row(5), row(6)])
    assert len({result.assignments[digest(i)] for i in range(1, 5)}) == 1
    assert result.group_count == 3


def test_input_and_fraction_mapping_order_do_not_change_membership():
    rows = [row(1), row(2, parents=(1,)), row(3), row(4)]
    expected = plan(rows).to_manifest()
    for order in permutations(rows):
        assert (
            plan(order, fractions={"test": 0.5, "train": 0.5}).to_manifest() == expected
        )


def test_feasible_language_and_rare_strata_are_preserved_without_splitting_groups():
    rows = [row(1, strata=("hi", "rare")), row(2, strata=("hi", "rare"))]
    result = plan(
        rows, required_strata={"train": ("hi", "rare"), "test": ("hi", "rare")}
    )
    assert result.unmet_strata == ()
    assert set(result.assignments.values()) == {"train", "test"}


def test_impossible_strata_are_reported_and_export_fails():
    result = plan(
        [row(1, strata=("rare",)), row(2, trajectory=1, strata=("rare",))],
        required_strata={"train": ("rare",), "test": ("rare",)},
    )
    assert len(set(result.assignments.values())) == 1
    assert len(result.unmet_strata) == 1
    with pytest.raises(ValueError, match="unsatisfied"):
        result.to_manifest()


def test_unknown_stratum_is_not_fabricated():
    result = plan([row(1)], required_strata={"test": ("absent",)})
    assert result.unmet_strata == (("test", "absent"),)


@pytest.mark.parametrize(
    "records,message",
    [
        ([row(1), row(1)], "duplicate artifact"),
        ([row(1, parents=(2,))], "unresolved ancestry"),
        ([row(1, parents=(1,))], "cyclic"),
        ([row(1, parents=(2,)), row(2, parents=(1,))], "cyclic"),
    ],
)
def test_invalid_ancestry_fails_before_allocation(records, message):
    with pytest.raises(ValueError, match=message):
        plan(records)


def test_duplicate_parent_and_non_synthetic_inputs_fail():
    with pytest.raises(ValueError, match="duplicate ancestry"):
        row(1, parents=(2, 2))
    with pytest.raises(ValueError, match="explicitly synthetic"):
        row(1, synthetic=False)
    with pytest.raises(ValueError, match="explicitly synthetic"):
        row(1, synthetic=1)


@pytest.mark.parametrize(
    "fractions",
    [
        {},
        {"train": 0},
        {"train": float("nan")},
        {"train": float("inf")},
        {"train": True},
        {"train": 0.9},
        {"train": -1, "test": 2},
    ],
)
def test_invalid_fractions_fail(fractions):
    with pytest.raises(ValueError):
        plan([], fractions=fractions)


@pytest.mark.parametrize("seed", [-1, True, 0.1, "7"])
def test_invalid_seed_fails(seed):
    with pytest.raises(ValueError, match="seed"):
        plan([], seed=seed)


def test_unknown_requirement_partition_fails():
    with pytest.raises(ValueError, match="unknown partition"):
        plan([], required_strata={"validation": ("rare",)})


def test_manifest_contains_only_synthetic_digests_codes_and_counts():
    result = plan([row(1)]).to_manifest()
    assert set(result) == {"schema_version", "synthetic", "group_count", "assignments"}
    assert result["synthetic"] is True
    assert set(result["assignments"]) == {digest(1)}


def test_deep_ancestry_uses_no_recursion():
    records = [row(1)] + [row(i, parents=(i - 1,)) for i in range(2, 1501)]
    result = plan(records)
    assert result.group_count == 1
    assert len(set(result.assignments.values())) == 1


def test_plan_assignments_cannot_be_mutated_after_validation():
    result = plan([row(1)])
    with pytest.raises(TypeError):
        result.assignments[digest(1)] = "other"


@pytest.mark.parametrize(
    "assignments,groups,unmet",
    [
        ({"source_text": "train"}, 1, ()),
        ({digest(1): "invalid partition"}, 1, ()),
        ({digest(1): "train"}, True, ()),
        ({digest(1): "train"}, 0, ()),
        ({digest(1): "train"}, 2, ()),
        ({}, 1, ()),
        ({}, 0, (("train", "invalid stratum"),)),
        ({}, 0, (("train",),)),
        ({}, 0, (("train", "rare"), ("train", "rare"))),
    ],
)
def test_public_plan_constructor_rejects_invalid_export_metadata(
    assignments, groups, unmet
):
    with pytest.raises(ValueError):
        SyntheticSplitPlan(assignments, groups, unmet)


def test_direct_constructor_snapshots_mutable_input_metadata():
    assignments = {digest(1): "train"}
    unmet = [["test", "rare"]]
    result = SyntheticSplitPlan(assignments, 1, unmet, "unsatisfied")
    assignments[digest(1)] = "changed"
    unmet[0][1] = "changed"
    assert result.assignments[digest(1)] == "train"
    assert result.unmet_strata == (("test", "rare"),)
    with pytest.raises(ValueError, match="unsatisfied"):
        result.to_manifest()


def test_record_ancestry_and_strata_snapshot_mutable_sequences():
    parents = [digest(1)]
    strata = ["rare"]
    record = SyntheticSplitRecord(digest(2), digest(2), digest(2), parents, strata)
    parents.append(digest(3))
    strata.append("different")
    assert record.parent_digests == (digest(1),)
    assert record.strata == ("rare",)


def test_feasible_overlapping_strata_use_scarcity_before_assignment():
    records = [row(1, strata=("x", "y")), row(2, strata=("x",))]
    result = plan(
        records,
        fractions={"train": 0.5, "validation": 0.5},
        seed=0,
        required_strata={"train": ("x",), "validation": ("y",)},
    )
    assert result.assignments == {digest(1): "validation", digest(2): "train"}
    assert result.search_status == "complete"
    result.to_manifest()


def test_search_budget_exhaustion_is_explicit_and_blocks_export(monkeypatch):
    import openmed.training.synthetic.split_planner as planner

    monkeypatch.setattr(planner, "MAX_SEARCH_STATES", 0)
    result = plan([row(1, strata=("x",))], required_strata={"train": ("x",)})
    assert result.search_status == "budget_exhausted"
    with pytest.raises(ValueError, match="unsatisfied"):
        result.to_manifest()


def test_public_constructor_rejects_unknown_search_status():
    with pytest.raises(ValueError, match="search status"):
        SyntheticSplitPlan({}, 0, (), "fabricated")


def test_large_feasible_cohort_is_not_rejected_by_candidate_search_cap():
    records = [row(i, strata=("common",)) for i in range(1, 140)]
    result = plan(records, required_strata={"train": ("common",), "test": ("common",)})
    assert result.search_status == "complete"
    assert result.unmet_strata == ()
    result.to_manifest()
