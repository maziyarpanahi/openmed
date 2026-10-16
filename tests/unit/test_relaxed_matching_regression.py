import random
from dataclasses import replace
from itertools import permutations, product

import pytest

from openmed.eval.metrics import EvalSpan, compute_relaxed_span_f1
from openmed.eval.relation_metrics import EvalRelation, compute_relation_f1


def spans():
    return [EvalSpan(0, 10, "NAME"), EvalSpan(10, 20, "NAME")], [
        EvalSpan(0, 20, "NAME"),
        EvalSpan(0, 5, "NAME"),
    ]


def test_relaxed_span_f1_is_permutation_invariant_and_maximal():
    gold, predicted = spans()
    for g in permutations(gold):
        for p in permutations(predicted):
            result = compute_relaxed_span_f1(g, p)
            assert (
                result.true_positives,
                result.false_positives,
                result.false_negatives,
                result.f1,
            ) == (2, 0, 0, 1.0)


def test_relaxed_relation_f1_reassigns_earlier_matches():
    gold, predicted = spans()
    tail = EvalSpan(30, 35, "LOCATION")
    gold = [EvalRelation("RELATED_TO", h, tail, fixture_id="synthetic") for h in gold]
    predicted = [
        EvalRelation("RELATED_TO", h, tail, fixture_id="synthetic") for h in predicted
    ]
    for g in permutations(gold):
        for p in permutations(predicted):
            result = compute_relation_f1(g, p, match="relaxed")
            assert result.true_positives == 2
            assert result.f1 == 1.0
    assert compute_relation_f1(gold, predicted, match="strict").true_positives == 0


def test_matching_does_not_cross_labels_fixtures_or_relation_types():
    a = EvalSpan(0, 5, "NAME")
    b = EvalSpan(10, 15, "LOCATION")
    assert (
        compute_relaxed_span_f1([a], [replace(a, label="LOCATION")]).true_positives == 0
    )
    assert (
        compute_relaxed_span_f1([a], [replace(a, start=5, end=8)]).true_positives == 0
    )
    rel = EvalRelation("RELATED_TO", a, b, fixture_id="one")
    for other in (
        replace(rel, fixture_id="two"),
        replace(rel, relation_type="OTHER"),
        replace(rel, head=b, tail=a),
    ):
        assert compute_relation_f1([rel], [other], match="relaxed").true_positives == 0


def test_duplicate_predictions_are_not_reused_and_empty_inputs_work():
    a = EvalSpan(0, 5, "NAME")
    result = compute_relaxed_span_f1([a, a], [a])
    assert (result.true_positives, result.false_negatives) == (1, 1)
    assert compute_relaxed_span_f1([], [a]).false_positives == 1
    assert compute_relaxed_span_f1([a], []).false_negatives == 1
    assert compute_relaxed_span_f1([], []).true_positives == 0


def test_matcher_against_exhaustive_small_graph_oracle():
    matching = pytest.importorskip("openmed.eval._matching")

    def oracle(rows, used=0):
        if not rows:
            return 0
        return max(
            [oracle(rows[1:], used)]
            + [
                1 + oracle(rows[1:], used | (1 << right))
                for right in rows[0]
                if not used & (1 << right)
            ]
        )

    for bits in product((False, True), repeat=9):
        rows = [[r for r in range(3) if bits[3 * left + r]] for left in range(3)]
        assert matching.maximum_matching_count(
            len(rows), 3, lambda left, right: right in rows[left]
        ) == oracle(rows)


def test_long_augmenting_path_does_not_use_python_recursion():
    matching = pytest.importorskip("openmed.eval._matching")
    count = 1500
    rows = [[i, i + 1] for i in range(count - 1)] + [[0]]
    assert (
        matching.maximum_matching_count(
            len(rows), count, lambda left, right: right in rows[left]
        )
        == count
    )


def test_rectangular_matchings_against_seeded_small_graph_oracle():
    matching = pytest.importorskip("openmed.eval._matching")
    rng = random.Random(314159)

    def oracle(rows, used=frozenset()):
        if not rows:
            return 0
        options = [oracle(rows[1:], used)]
        options.extend(
            1 + oracle(rows[1:], used | {right})
            for right in rows[0]
            if right not in used
        )
        return max(options)

    for left_count, right_count in product(range(6), repeat=2):
        for _ in range(8):
            rows = [
                [right for right in range(right_count) if rng.random() < 0.4]
                for _ in range(left_count)
            ]
            expected = oracle(rows)
            assert (
                matching.maximum_matching_count(
                    len(rows), right_count, lambda left, right: right in rows[left]
                )
                == expected
            )
            reordered = [list(reversed(row)) for row in reversed(rows)]
            assert (
                matching.maximum_matching_count(
                    len(reordered),
                    right_count,
                    lambda left, right: right in reordered[left],
                )
                == expected
            )


def test_dense_blocked_component_is_searched_once_for_all_unmatched_roots():
    matching = pytest.importorskip("openmed.eval._matching")
    maximum_matching_count = matching.maximum_matching_count

    left_count, right_count = 200, 100
    calls = 0

    def eligible(left, right):
        nonlocal calls
        calls += 1
        return right < right_count // 2

    assert maximum_matching_count(left_count, right_count, eligible) == 50
    assert calls <= 2 * left_count * right_count


def test_dense_matcher_uses_linear_auxiliary_memory():
    import tracemalloc

    matching = pytest.importorskip("openmed.eval._matching")
    maximum_matching_count = matching.maximum_matching_count

    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        assert maximum_matching_count(3000, 3000, lambda left, right: True) == 3000
        _, peak = tracemalloc.get_traced_memory()
        peak -= baseline
    finally:
        if not tracing:
            tracemalloc.stop()
    # Storing the 9M eligible edges would exceed this allocation bound.
    assert peak < 2_000_000


def test_strict_relation_behavior_and_duplicate_counts_are_unchanged():
    head, tail = EvalSpan(0, 3, "NAME"), EvalSpan(10, 13, "LOCATION")
    relation = EvalRelation("RELATED_TO", head, tail, fixture_id="one")
    other_fixture = replace(relation, fixture_id="two")
    # Existing strict relation eligibility compares endpoint offsets, not labels.
    relabeled = replace(relation, head=replace(head, label="LOCATION"))
    for gold in permutations([relation, relation, other_fixture]):
        for predicted in permutations([relabeled, other_fixture]):
            result = compute_relation_f1(gold, predicted, match="strict")
            assert (
                result.true_positives,
                result.false_positives,
                result.false_negatives,
            ) == (2, 0, 1)


def test_multilingual_group_scoring_keeps_fixture_coordinates_separate():
    from openmed.eval.harness import (
        BenchmarkFixture,
        FixtureResult,
        _multilingual_group_rows,
    )

    gold, predicted = spans()
    fixtures = [
        BenchmarkFixture(
            str(i), "x" * 40, tuple(gold), metadata={"benchmark": "synthetic"}
        )
        for i in range(2)
    ]
    results = [FixtureResult("0", tuple(predicted), 0.0), FixtureResult("1", (), 0.0)]
    row = _multilingual_group_rows(
        fixtures, results, group_keys=("benchmark", "language")
    )[0]
    relaxed = row["relaxed_span_f1"]
    assert (
        relaxed["true_positives"],
        relaxed["false_positives"],
        relaxed["false_negatives"],
    ) == (2, 0, 2)
    assert row["fixture_count"] == 2
    assert row["span_count"] == 4


def test_helper_only_changes_invalidate_eval_cache(monkeypatch):
    pytest.importorskip("openmed.eval._matching")
    from openmed.eval import cache

    before = cache.eval_code_hash()
    original = cache._module_source
    monkeypatch.setattr(
        cache,
        "_module_source",
        lambda module: (
            b"synthetic helper revision"
            if module.__name__ == "openmed.eval._matching"
            else original(module)
        ),
    )
    assert cache.eval_code_hash() != before
