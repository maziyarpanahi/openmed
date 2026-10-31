"""Non-IID exact quota, skew, reproducibility and augmenting-path controls."""

from collections import Counter
from itertools import permutations

import pytest

from openmed.training.federated.non_iid_simulator import (
    InstitutionArchetype,
    SyntheticPartitionExample,
    simulate_non_iid_partitions,
)


def example(n, language="hi", label="ID", specialty="general"):
    return SyntheticPartitionExample(f"{n:064x}", language, (label,), specialty)


def profile(code, count, *, languages=None, labels=None, specialties=None):
    return InstitutionArchetype(
        code, count, languages or {}, labels or {}, specialties or {}
    )


def test_all_four_skews_and_actual_partition_statistics_are_preserved():
    rows = [
        example(
            i,
            "hi" if i <= 3 else "te",
            "RARE" if i <= 3 else "COMMON",
            "specialist" if i <= 3 else "general",
        )
        for i in range(1, 7)
    ]
    profiles = [
        profile(
            "specialist_center",
            3,
            languages={"hi": 1, "te": 0},
            labels={"RARE": 100, "COMMON": 0},
            specialties={"specialist": 10, "general": 0},
        ),
        profile("general_center", 2, languages={"hi": 0, "te": 1}),
    ]
    result = simulate_non_iid_partitions(rows, profiles, seed=7)
    first = result["partitions"]["specialist_center"]
    assert first["sample_count"] == 3
    assert first["language_counts"] == {"hi": 3}
    assert first["label_counts"] == {"RARE": 3}
    assert first["specialty_counts"] == {"specialist": 3}
    assert result["partitions"]["general_center"]["sample_count"] == 2
    assert len(result["unassigned_example_digests"]) == 1


def test_input_and_profile_order_do_not_change_membership():
    rows = [example(1), example(2), example(3, "te")]
    profiles = [profile("a", 1), profile("b", 1)]
    expected = simulate_non_iid_partitions(rows, profiles, seed=7)
    for order in permutations(rows):
        assert simulate_non_iid_partitions(order, profiles[::-1], seed=7) == expected


def test_weighted_label_prevalence_changes_assignment_on_fixed_seed():
    rows = [example(i, label="RARE" if i <= 50 else "COMMON") for i in range(1, 101)]
    scarce = simulate_non_iid_partitions(
        rows, [profile("a", 20, labels={"RARE": 1000, "COMMON": 1})], seed=7
    )
    common = simulate_non_iid_partitions(
        rows, [profile("a", 20, labels={"RARE": 1, "COMMON": 1000})], seed=7
    )
    assert scarce["partitions"]["a"]["label_counts"]["RARE"] >= 18
    assert common["partitions"]["a"]["label_counts"]["COMMON"] >= 18


def test_overlap_is_impossible_and_exact_stats_match_membership():
    rows = [example(i, label="RARE" if i % 2 else "COMMON") for i in range(1, 11)]
    result = simulate_non_iid_partitions(
        rows, [profile("a", 3), profile("b", 7)], seed=3
    )
    memberships = [
        digest
        for part in result["partitions"].values()
        for digest in part["example_digests"]
    ]
    assert len(memberships) == len(set(memberships)) == 10
    by_id = {row.example_digest: row for row in rows}
    for part in result["partitions"].values():
        assert part["label_counts"] == dict(
            Counter(by_id[digest].labels[0] for digest in part["example_digests"])
        )


def test_augmenting_assignment_preserves_feasible_restricted_quotas():
    rows = [example(1, "hi"), example(2, "te"), example(3, "hi")]
    profiles = [profile("flexible", 1), profile("restricted", 2, languages={"te": 0})]
    result = simulate_non_iid_partitions(rows, profiles, seed=0)
    assert result["partitions"]["restricted"]["language_counts"] == {"hi": 2}
    assert result["partitions"]["flexible"]["language_counts"] == {"te": 1}


def test_excess_or_impossible_quota_fails_instead_of_relaxing_constraints():
    with pytest.raises(ValueError):
        simulate_non_iid_partitions([example(1)], [profile("a", 2)], seed=0)
    with pytest.raises(ValueError):
        simulate_non_iid_partitions(
            [example(1)], [profile("a", 1, languages={"hi": 0})], seed=0
        )


def test_duplicate_and_non_synthetic_examples_are_rejected():
    with pytest.raises(ValueError):
        simulate_non_iid_partitions([example(1), example(1)], [], seed=0)
    with pytest.raises(ValueError):
        SyntheticPartitionExample("1" * 64, "hi", ("ID",), "general", False)


def test_seed_and_manifest_identity_are_recorded_without_source_payloads():
    result = simulate_non_iid_partitions([example(1)], [profile("a", 1)], seed=17)
    assert result["seed"] == 17
    assert len(result["fixture_manifest_digest"]) == 64
    assert "text" not in result
    assert "patient" not in result


def test_equal_eligibility_profiles_need_reassignment_to_preserve_all_quotas():
    rows = [
        example(1, specialty="shared"),
        example(2, specialty="a_only"),
        example(3, specialty="bc_only"),
    ]
    profiles = [
        profile("a", 1, specialties={"shared": 1, "a_only": 1, "bc_only": 0}),
        profile("b", 1, specialties={"shared": 1, "a_only": 0, "bc_only": 1}),
        profile("c", 1, specialties={"shared": 1, "a_only": 0, "bc_only": 1}),
    ]
    for seed in range(20):
        result = simulate_non_iid_partitions(rows, profiles, seed=seed)
        assert result["partitions"]["a"]["example_digests"] == [f"{2:064x}"]
        assert result["assigned_count"] == 3


def test_nested_label_input_has_controlled_validation_error():
    with pytest.raises(ValueError):
        SyntheticPartitionExample("0" * 64, "en", (["ID"],), "general")
