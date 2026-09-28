"""Offline regression tests for the synthetic clinical NLI negation challenge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval.nli_negation_challenge import (
    DEFAULT_NLI_NEGATION_CASES,
    NEGATION_PATTERNS,
    NliNegationCase,
    NliNegationChallengeError,
    NliNegationGateError,
    assert_false_entailment_gate,
    assert_nli_negation_gate,
    check_false_entailment_gate,
    default_nli_negation_cases,
    load_nli_negation_fixtures,
    run_nli_negation_challenge,
    write_nli_negation_report,
)


def _gold_predictions(cases):
    return {case.case_id: case.gold_label for case in cases}


def test_default_corpus_is_synthetic_and_covers_all_negation_patterns() -> None:
    cases = default_nli_negation_cases()

    assert cases is DEFAULT_NLI_NEGATION_CASES
    assert len(cases) == 7
    assert {case.pattern for case in cases} == set(NEGATION_PATTERNS)
    assert all(case.synthetic for case in cases)
    assert all(case.contains_real_phi is False for case in cases)
    assert all(case.premise and case.hypothesis for case in cases)
    assert any(
        case.pattern == "nested" and case.gold_label == "neutral" for case in cases
    )
    assert {case.gold_label for case in cases if case.pattern == "double"} == {
        "entailment",
        "contradiction",
    }
    assert {case.case_id: case.gold_label for case in cases} == {
        "simple-contradiction": "contradiction",
        "simple-entailment": "entailment",
        "nested-neutral": "neutral",
        "double-contradiction": "contradiction",
        "double-entailment": "entailment",
        "section-scoped-contradiction": "contradiction",
        "section-scoped-entailment": "entailment",
    }
    section_case = next(
        case for case in cases if case.case_id == "section-scoped-contradiction"
    )
    assert section_case.section == "negative_findings"
    assert "ACTIVE PROBLEMS:" in section_case.premise
    assert all(
        any(case.pattern == pattern and case.requires_non_entailment for case in cases)
        for pattern in NEGATION_PATTERNS
    )


def test_perfect_local_runner_reports_accuracy_and_zero_false_entailment() -> None:
    cases = default_nli_negation_cases()
    expected = _gold_predictions(cases)

    def runner(premise: str, hypothesis: str) -> str:
        for case in cases:
            if (case.premise, case.hypothesis) == (premise, hypothesis):
                return case.gold_label
        raise AssertionError("unexpected synthetic case")

    report = run_nli_negation_challenge(runner=runner)

    assert report.case_count == len(cases)
    assert report.fixture_count == len(cases)
    assert report.correct_count == len(cases)
    assert report.aggregate_accuracy == 1.0
    assert report.accuracy == 1.0
    assert report.false_entailment_count == 0
    assert report.false_entailment_rate == 0.0
    assert report.gate.false_entailment_gate_passed is True
    assert report.gate.aggregate_accuracy_gate_passed is True
    assert report.passed is True
    assert report.prediction_hash
    assert set(report.by_pattern) == set(NEGATION_PATTERNS)
    assert {
        pattern: metrics.case_count for pattern, metrics in report.by_pattern.items()
    } == {
        "simple": 2,
        "nested": 1,
        "double": 2,
        "section_scoped": 2,
    }
    assert set(expected) == {case.case_id for case in cases}


def test_false_entailment_gate_is_independent_from_aggregate_accuracy() -> None:
    cases = default_nli_negation_cases()
    predictions = {
        case.case_id: (
            "entailment" if case.requires_non_entailment else case.gold_label
        )
        for case in cases
    }

    report = run_nli_negation_challenge(predictions=predictions)

    assert report.aggregate_accuracy < 1.0
    assert report.gate.aggregate_accuracy_gate_passed is True
    assert report.false_entailment_count == report.false_entailment_case_count
    assert report.false_entailment_rate == 1.0
    assert report.false_entailment_gate_passed is False
    assert report["false_entailment_gate_passed"] is False
    assert report.gate.false_entailment_gate_passed is False
    assert check_false_entailment_gate(report) is False
    with pytest.raises(NliNegationGateError, match="false-entailment"):
        assert_false_entailment_gate(report)


def test_accuracy_floor_remains_visible_when_false_entailment_gate_passes() -> None:
    cases = default_nli_negation_cases()
    predictions = _gold_predictions(cases)
    predictions["simple-entailment"] = "neutral"

    report = run_nli_negation_challenge(
        predictions=predictions,
        minimum_aggregate_accuracy=1.0,
    )

    assert report.aggregate_accuracy < 1.0
    assert report.gate.aggregate_accuracy_gate_passed is False
    assert report.gate.false_entailment_gate_passed is True
    assert check_false_entailment_gate(report) is True
    assert_false_entailment_gate(report)
    with pytest.raises(NliNegationGateError, match="negation gate"):
        assert_nli_negation_gate(report)


def test_report_is_order_independent_for_precomputed_predictions() -> None:
    cases = default_nli_negation_cases()
    predictions = _gold_predictions(cases)

    first = run_nli_negation_challenge(predictions=predictions, cases=cases)
    second = run_nli_negation_challenge(
        predictions=predictions,
        cases=tuple(reversed(cases)),
    )

    assert first.to_json() == second.to_json()
    assert first.to_dict() == second.to_dict()


def test_one_argument_runner_and_prediction_mapping_outputs_are_supported() -> None:
    report = run_nli_negation_challenge(
        runner=lambda case: {"label": case.gold_label},
    )

    assert report.aggregate_accuracy == 1.0
    assert report.abstention_count == 0


def test_abstention_is_incorrect_but_not_false_entailment() -> None:
    cases = default_nli_negation_cases()
    predictions = [None] * len(cases)

    report = run_nli_negation_challenge(predictions=predictions)

    assert report.correct_count == 0
    assert report.abstention_count == len(cases)
    assert report.aggregate_accuracy == 0.0
    assert report.false_entailment_count == 0
    assert report.false_entailment_rate == 0.0
    assert report.passed is True


def test_label_aliases_and_local_fixture_loader_are_deterministic(
    tmp_path: Path,
) -> None:
    case = NliNegationCase(
        case_id="local-case",
        pattern="simple",
        premise="No evidence of condition zeta is present.",
        hypothesis="Condition zeta is present.",
        gold_label="contradiction",
    )
    path = tmp_path / "negation.jsonl"
    path.write_text(json.dumps(case.to_mapping()) + "\n", encoding="utf-8")

    loaded = load_nli_negation_fixtures(path)
    report = run_nli_negation_challenge(
        predictions=["contradict"],
        cases=loaded,
    )

    assert loaded == (case,)
    assert report.aggregate_accuracy == 1.0
    assert report.false_entailment_rate == 0.0


def test_report_and_errors_never_echo_source_text_or_predictor_exceptions() -> None:
    sentinel = "synthetic-private-marker-4815"
    case = NliNegationCase(
        case_id="private-case",
        pattern="simple",
        premise=f"No evidence of {sentinel} is present.",
        hypothesis=f"{sentinel} is present.",
        gold_label="contradiction",
    )
    report = run_nli_negation_challenge(
        predictions=["contradiction"],
        cases=[case],
    )
    serialized = report.to_json() + report.to_markdown()

    assert sentinel not in serialized
    assert case.case_id not in serialized
    assert "premise" not in serialized
    assert "hypothesis" not in serialized

    def failing_runner(_: str, __: str) -> str:
        raise RuntimeError(sentinel)

    with pytest.raises(NliNegationChallengeError) as error:
        run_nli_negation_challenge(runner=failing_runner, cases=[case])
    assert sentinel not in str(error.value)


def test_unsafe_fixture_and_prediction_errors_do_not_echo_values(
    tmp_path: Path,
) -> None:
    sentinel = "synthetic-private-marker-9271"
    unsafe = {
        "case_id": "unsafe-case",
        "pattern": "simple",
        "premise": sentinel,
        "hypothesis": "condition is present",
        "gold_label": "contradiction",
        "synthetic": False,
    }
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps([unsafe]), encoding="utf-8")

    with pytest.raises(NliNegationChallengeError) as fixture_error:
        load_nli_negation_fixtures(path)
    assert sentinel not in str(fixture_error.value)

    with pytest.raises(NliNegationChallengeError) as prediction_error:
        run_nli_negation_challenge(
            predictions=[sentinel],
            cases=[
                NliNegationCase(
                    case_id="prediction-case",
                    pattern="simple",
                    premise="No evidence of condition eta is present.",
                    hypothesis="Condition eta is present.",
                    gold_label="contradiction",
                )
            ],
        )
    assert sentinel not in str(prediction_error.value)


def test_report_rendering_and_file_writes_are_stable(tmp_path: Path) -> None:
    report = run_nli_negation_challenge(
        predictions=_gold_predictions(default_nli_negation_cases())
    )
    json_path = write_nli_negation_report(
        report,
        tmp_path / "nested" / "negation.json",
    )
    markdown_path = write_nli_negation_report(
        report,
        tmp_path / "nested" / "negation.md",
        format="markdown",
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report.to_dict()
    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert markdown_path.read_text(encoding="utf-8") == report.to_markdown()
    assert report.to_json() == report.to_json()
    assert report.to_markdown().startswith("# Clinical NLI Negation Challenge\n")
    assert "False-entailment gate" in report.to_markdown()


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_oversized_gate_thresholds_fail_cleanly(value: int) -> None:
    with pytest.raises(NliNegationChallengeError):
        run_nli_negation_challenge(
            predictions=_gold_predictions(default_nli_negation_cases()),
            max_false_entailment_rate=value,
        )


@pytest.mark.parametrize("error_type", [RuntimeError, NliNegationChallengeError])
def test_predictor_tracebacks_suppress_all_source_exceptions(error_type) -> None:
    import traceback

    def runner(case):
        raise error_type("synthetic-private-5550199")

    with pytest.raises(NliNegationChallengeError) as caught:
        run_nli_negation_challenge(runner=runner)
    assert "synthetic-private-5550199" not in "".join(
        traceback.format_exception(caught.value)
    )


def test_public_negation_labels_are_qualified() -> None:
    from openmed.eval import NLI_NEGATION_LABELS

    assert NLI_NEGATION_LABELS == (
        "entailment",
        "contradiction",
        "neutral",
        "abstention",
    )


@pytest.mark.parametrize(
    "field", ["fixture_set_hash", "case_count", "aggregate_accuracy"]
)
def test_report_rejects_uncontrolled_or_inconsistent_evidence(field: str) -> None:
    from dataclasses import replace

    report = run_nli_negation_challenge(
        predictions=_gold_predictions(default_nli_negation_cases())
    )
    value = 0.0 if field == "aggregate_accuracy" else "synthetic-private-5550199"
    with pytest.raises(NliNegationChallengeError):
        replace(report, **{field: value})


def test_report_rejects_misattributed_pattern() -> None:
    from dataclasses import replace

    report = run_nli_negation_challenge(
        predictions=_gold_predictions(default_nli_negation_cases())
    )
    patterns = dict(report.by_pattern)
    patterns["simple"] = report.by_pattern["double"]
    with pytest.raises(NliNegationChallengeError):
        replace(report, by_pattern=patterns)


def test_gate_rejects_forged_decision() -> None:
    from dataclasses import replace

    report = run_nli_negation_challenge(
        predictions=_gold_predictions(default_nli_negation_cases())
    )
    with pytest.raises(NliNegationChallengeError):
        replace(report.gate, false_entailment_gate_passed=False)
