"""Offline native-review queue acceptance and negative controls."""

from itertools import permutations

import pytest

from openmed.training.languages.scarce_label_queue import (
    ReviewCandidate,
    select_native_review,
)


def row(
    n,
    *,
    label="RARE",
    language="hi",
    scarcity=1.0,
    disagreement=0.5,
    cluster=1,
    signature=1,
):
    return ReviewCandidate(
        f"{n:064x}",
        language,
        label,
        scarcity,
        disagreement,
        f"{cluster:064x}",
        f"{signature:064x}",
    )


def choose(rows, **kwargs):
    args = dict(
        language_budgets={"hi": 10}, label_budgets={"RARE": 10}, total_budget=10
    )
    args.update(kwargs)
    return select_native_review(rows, **args)


def test_scarcity_precedes_common_label_uncertainty():
    selected = choose(
        [row(1, label="COMMON", scarcity=0, disagreement=1), row(2)],
        label_budgets={"RARE": 1, "COMMON": 1},
        total_budget=1,
    )
    assert selected.selected == (row(2),)


def test_simultaneous_language_label_and_total_budgets():
    rows = [row(1), row(2), row(3, language="te"), row(4, label="OTHER")]
    result = choose(
        rows,
        language_budgets={"hi": 1, "te": 1},
        label_budgets={"RARE": 1, "OTHER": 1},
        total_budget=2,
    )
    assert len(result.selected) == 1
    assert result.language_counts == {"hi": 1}
    assert result.label_counts == {"RARE": 1}
    # Language and label caps independently prohibit each remaining candidate.


def test_diversity_and_digest_ties_are_permutation_invariant():
    rows = [row(1), row(2), row(3, cluster=2), row(4, cluster=3)]
    expected = (rows[0], rows[2], rows[3], rows[1])
    for order in permutations(rows):
        assert choose(order).selected == expected


def test_missing_categories_and_zero_budget_do_not_authorize_review():
    assert choose([row(1, language="te")]).selected == ()
    assert choose([row(1, label="OTHER")]).selected == ()
    assert choose([row(1)], total_budget=0).selected == ()
    assert choose([]).considered_count == 0


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_invalid_ranking_evidence_fails(score):
    with pytest.raises(ValueError, match="ranking evidence"):
        row(1, scarcity=score)


@pytest.mark.parametrize("budget", [-1, True, 1.5, "1"])
def test_invalid_budgets_fail(budget):
    with pytest.raises(ValueError, match="budgets"):
        choose([row(1)], total_budget=budget)
    with pytest.raises(ValueError, match="budgets"):
        choose([row(1)], label_budgets={"RARE": budget})


def test_duplicate_references_fail():
    with pytest.raises(ValueError, match="duplicate"):
        choose([row(1), row(1, scarcity=0)])


def test_serialized_result_contains_only_counts_codes_and_digests():
    result = choose([row(1)]).to_dict()
    assert set(result) == {
        "schema_version",
        "selected_digests",
        "considered_count",
        "selected_count",
        "language_counts",
        "label_counts",
    }
    assert result["selected_digests"] == [f"{1:064x}"]
    assert "text" not in result


def test_reference_and_category_validation():
    with pytest.raises(ValueError, match="SHA-256"):
        ReviewCandidate("not-a-digest", "hi", "RARE", 1, 1, "0" * 64, "0" * 64)
    with pytest.raises(ValueError, match="category"):
        row(1, label="invalid category")


def test_error_signatures_rank_independently_of_diversity_clusters():
    rows = [row(1, cluster=1), row(2, cluster=2), row(3, cluster=1, signature=2)]
    # A new error signature outranks a new cluster for the repeated signature.
    assert choose(rows).selected == (rows[0], rows[2], rows[1])


def test_language_and_region_aliases_share_one_budget():
    result = choose(
        [row(1, language="HI"), row(2, language="hi-IN")], language_budgets={"hi_IN": 1}
    )
    assert len(result.selected) == 1
    assert result.language_counts == {"hi": 1}
    with pytest.raises(ValueError, match="duplicate"):
        choose([], language_budgets={"hi": 1, "HI": 1})
    with pytest.raises(ValueError, match="explicit"):
        row(1, language="auto")


def test_report_counts_are_immutable_snapshots():
    result = choose([row(1)])
    with pytest.raises(TypeError):
        result.language_counts["hi"] = 99
    exported = result.to_dict()
    exported["language_counts"]["hi"] = 99
    assert result.language_counts["hi"] == 1


def test_oversized_candidates_fail_before_selection():
    with pytest.raises(ValueError, match="candidate limit"):
        choose([row(1)] * 10_001)


def test_budget_category_count_is_bounded():
    with pytest.raises(ValueError, match="category limit"):
        choose([], label_budgets={f"LABEL_{i}": 1 for i in range(257)})


def test_pathological_ties_respect_work_limit(monkeypatch):
    import openmed.training.languages.scarce_label_queue as queue

    monkeypatch.setattr(queue, "MAX_RANK_UPDATES", 0)
    with pytest.raises(ValueError, match="work limit"):
        choose([row(1), row(2)])


def test_public_selection_snapshots_and_validates_accounting():
    from dataclasses import replace

    result = choose([row(1)])
    values = [row(1)]
    copied = replace(result, selected=values)
    values.clear()
    assert copied.selected == (row(1),)
    with pytest.raises(ValueError):
        replace(result, label_counts={"ARBITRARY SENTINEL": -1})
    with pytest.raises(ValueError):
        replace(result, considered_count=-1)
    with pytest.raises(ValueError):
        replace(result, language_counts={"hi": 2})
