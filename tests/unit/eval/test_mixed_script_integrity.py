"""Regression coverage for mixed-script span integrity."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Iterator

from openmed.eval.mixed_script_integrity import (
    GRAPHEME_BOUNDARY_FAILURE,
    NONDETERMINISTIC_RUN_FAILURE,
    RUNNER_FAILURE,
    SPAN_MISMATCH_FAILURE,
    SPAN_ORDER_FAILURE,
    SURROGATE_STABILITY_FAILURE,
    MixedScriptSpan,
    default_mixed_script_fixtures,
    evaluate_mixed_script_integrity,
)


def _json_strings(payload: object) -> Iterator[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key)
            yield from _json_strings(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _json_strings(item)
    else:
        yield str(payload)


def test_default_fixtures_cover_required_boundaries_and_pass() -> None:
    fixtures = default_mixed_script_fixtures()
    report = evaluate_mixed_script_integrity(fixtures)

    coverage = {tag for fixture in fixtures for tag in fixture.coverage}
    assert {"Latin", "Devanagari", "Han", "Arabic"} <= coverage
    assert {"combining-mark", "bidi", "script-transition", "grapheme"} <= coverage
    assert report.fixture_count == 5
    assert report.span_count == 10
    assert report.deterministic is True
    assert report.passed is True


def test_report_is_deterministic_json_and_contains_no_raw_values() -> None:
    fixtures = default_mixed_script_fixtures()

    first = evaluate_mixed_script_integrity(fixtures, iterations=5).to_dict()
    second = evaluate_mixed_script_integrity(fixtures, iterations=5).to_dict()
    encoded = json.dumps(first, ensure_ascii=False, sort_keys=True)

    assert first == second
    assert json.loads(encoded) == first
    serialized_values = set(_json_strings(first))
    for fixture in fixtures:
        assert fixture.text not in encoded
        for span in fixture.spans:
            assert fixture.text[span.start : span.end] not in serialized_values
            assert span.surrogate not in serialized_values
            assert span.surrogate not in repr(span)


def test_combining_mark_split_fails_grapheme_and_exact_source_checks() -> None:
    fixture = default_mixed_script_fixtures()[0]

    def split_combining_mark(_fixture) -> tuple[MixedScriptSpan, ...]:
        first, second = fixture.spans
        # ``Jose\N{COMBINING ACUTE ACCENT}``: offset +4 falls between the
        # Latin base character and its combining mark.
        return (replace(first, end=first.start + 4), second)

    result = evaluate_mixed_script_integrity(
        (fixture,), runner=split_combining_mark
    ).fixture_results[0]

    assert result.passed is False
    assert SPAN_MISMATCH_FAILURE in result.failures
    assert f"span-0:{GRAPHEME_BOUNDARY_FAILURE}" in result.failures
    assert all(fixture.text not in failure for failure in result.failures)


def test_reversed_spans_fail_exact_ordering() -> None:
    fixture = default_mixed_script_fixtures()[2]

    result = evaluate_mixed_script_integrity(
        (fixture,), runner=lambda item: tuple(reversed(item.spans))
    ).fixture_results[0]

    assert result.passed is False
    assert SPAN_MISMATCH_FAILURE in result.failures
    assert f"span-1:{SPAN_ORDER_FAILURE}" in result.failures


def test_repeated_entity_requires_one_stable_surrogate() -> None:
    fixture = default_mixed_script_fixtures()[3]

    def drift_surrogate(item) -> tuple[MixedScriptSpan, ...]:
        first, second = item.spans
        return (first, replace(second, surrogate="different synthetic replacement"))

    result = evaluate_mixed_script_integrity(
        (fixture,), runner=drift_surrogate
    ).fixture_results[0]

    assert result.passed is False
    assert any(
        failure.endswith(SURROGATE_STABILITY_FAILURE) for failure in result.failures
    )
    assert "different synthetic replacement" not in json.dumps(result.to_dict())


def test_stateful_runner_is_detected_without_exposing_values() -> None:
    fixture = default_mixed_script_fixtures()[4]
    calls = 0

    def alternating_runner(item) -> tuple[MixedScriptSpan, ...]:
        nonlocal calls
        calls += 1
        return item.spans if calls % 2 else tuple(reversed(item.spans))

    report = evaluate_mixed_script_integrity(
        (fixture,), runner=alternating_runner, iterations=3
    )

    assert report.deterministic is False
    assert report.passed is False
    assert report.failures == (NONDETERMINISTIC_RUN_FAILURE,)


def test_runner_exception_is_sanitized() -> None:
    fixture = default_mixed_script_fixtures()[0]
    raw_value = fixture.text[fixture.spans[0].start : fixture.spans[0].end]

    def failing_runner(_fixture):
        raise RuntimeError(f"failed on {raw_value}")

    report = evaluate_mixed_script_integrity((fixture,), runner=failing_runner)
    encoded = json.dumps(report.to_dict(), ensure_ascii=False)

    assert report.passed is False
    assert report.fixture_results[0].failures == (RUNNER_FAILURE,)
    assert raw_value not in encoded


def test_default_evaluation_never_opens_a_network_socket(monkeypatch) -> None:
    def reject_network(*_args, **_kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr("socket.socket", reject_network)

    assert evaluate_mixed_script_integrity().passed is True
