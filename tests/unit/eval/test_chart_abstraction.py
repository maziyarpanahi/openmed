"""Synthetic scorer controls; no models, restricted data or network access."""

from dataclasses import replace

import pytest

from openmed.agent.workflows.abstraction_evidence import (
    AbstractionEvidenceChain,
    ReviewerState,
    SourceKind,
    SourceLocation,
    TransformationKind,
)
from openmed.eval.suites.chart_abstraction import (
    AbstractionFieldType,
    AbstractionMetric,
    ChartAbstractionGoldField,
    ChartAbstractionPrediction,
    run_chart_abstraction_benchmark,
    synthetic_chart_abstraction_gold,
)

FACT = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


def gold(value="Synthetic Alpha", kind=AbstractionFieldType.TEXT):
    return ChartAbstractionGoldField("case_1", "registry.label", kind, value)


def prediction(value="Synthetic Alpha", **kwargs):
    return ChartAbstractionPrediction("case_1", "registry.label", value, **kwargs)


def chain(*, sources=None, field_id="registry.label", digest=FACT):
    if sources is None:
        sources = (SourceLocation(OTHER, 5, 12),)
    return AbstractionEvidenceChain(
        field_id,
        sources,
        digest,
        TransformationKind.RULE,
        OTHER,
        0.0,
        ReviewerState.PENDING,
    )


def test_synthetic_gold_is_deterministic_and_all_fields_are_scored():
    rows = synthetic_chart_abstraction_gold()
    outputs = tuple(
        ChartAbstractionPrediction(row.case_id, row.field_id, row.value) for row in rows
    )
    report = run_chart_abstraction_benchmark(rows, outputs)
    assert (
        report.to_json()
        == run_chart_abstraction_benchmark(
            tuple(reversed(rows)), tuple(reversed(outputs))
        ).to_json()
    )
    assert rows == synthetic_chart_abstraction_gold()
    assert report.overall.field_count == 12
    assert report.overall.exact_agreement == AbstractionMetric(8, 8)
    assert report.overall.normalized_agreement == AbstractionMetric(8, 8)
    assert report.overall.abstention_correctness == AbstractionMetric(12, 12)
    assert report.overall.evidence_support == AbstractionMetric(0, 8)
    assert len(report.by_field) == 4
    assert len(report.by_field_type) == 4
    for _, scores in (*report.by_field, *report.by_field_type):
        assert scores.field_count == 3
        assert scores.answerable_count == 2
        assert scores.unanswerable_count == 1
        assert scores.exact_agreement == AbstractionMetric(2, 2)


@pytest.mark.parametrize(
    "expected,actual,kind,exact,normalized",
    [
        ("Synthetic Alpha", " synthetic  ALPHA ", AbstractionFieldType.TEXT, 0, 1),
        ("café", "CAFE\u0301", AbstractionFieldType.TEXT, 0, 1),
        ("code_a", "ＣＯＤＥ＿Ａ", AbstractionFieldType.CATEGORICAL, 0, 1),
        (12, "12.00", AbstractionFieldType.NUMBER, 0, 1),
        (12, 12.0, AbstractionFieldType.NUMBER, 0, 1),
        (12, "12 mg", AbstractionFieldType.NUMBER, 0, 0),
        (12, "NaN", AbstractionFieldType.NUMBER, 0, 0),
        (True, 1, AbstractionFieldType.BOOLEAN, 0, 0),
        (False, False, AbstractionFieldType.BOOLEAN, 1, 1),
        ("code_a", "code_b", AbstractionFieldType.CATEGORICAL, 0, 0),
        ("", "", AbstractionFieldType.TEXT, 1, 1),
    ],
)
def test_exact_and_normalized_agreement(expected, actual, kind, exact, normalized):
    scores = run_chart_abstraction_benchmark(
        [gold(expected, kind)], [prediction(actual)]
    ).overall
    assert scores.exact_agreement == AbstractionMetric(exact, 1)
    assert scores.normalized_agreement == AbstractionMetric(normalized, 1)


def test_abstention_errors_are_separate_and_missing_is_not_correct_abstention():
    rows = (
        gold(),
        replace(gold(None), case_id="case_2"),
        replace(gold(None), case_id="case_3"),
        replace(gold(None), case_id="case_4"),
    )
    outputs = (
        prediction(None),
        replace(prediction("invented answer"), case_id="case_2"),
        replace(prediction(None), case_id="case_3"),
    )
    scores = run_chart_abstraction_benchmark(rows, outputs).overall
    assert scores.answerable_abstention == AbstractionMetric(1, 1)
    assert scores.unanswerable_answer == AbstractionMetric(1, 3)
    assert scores.abstention_correctness == AbstractionMetric(1, 4)
    assert scores.exact_agreement == AbstractionMetric(0, 1)
    assert scores.missing_count == 1
    assert scores.abstained_count == 2
    assert scores.answered_count == 1


@pytest.mark.parametrize(
    "evidence,digest,supported",
    [
        (None, None, 0),
        (chain(sources=()), FACT, 0),
        (
            chain(sources=(SourceLocation(OTHER, 0, 2, SourceKind.GENERATED_TEXT),)),
            FACT,
            0,
        ),
        (chain(field_id="registry.other"), FACT, 0),
        (chain(), None, 0),
        (chain(), OTHER, 0),
        (chain(), FACT, 1),
        (
            chain(
                sources=(
                    SourceLocation(OTHER, 0, 2, SourceKind.GENERATED_TEXT),
                    SourceLocation(OTHER, 5, 12),
                )
            ),
            FACT,
            1,
        ),
    ],
)
def test_support_requires_a_fact_bound_clinical_source(evidence, digest, supported):
    scores = run_chart_abstraction_benchmark(
        [gold()],
        [prediction(normalized_fact_digest=digest, evidence_chain=evidence)],
    ).overall
    assert scores.exact_agreement == AbstractionMetric(1, 1)
    assert scores.evidence_support == AbstractionMetric(supported, 1)


def test_source_support_does_not_assert_accuracy_or_review_approval():
    scores = run_chart_abstraction_benchmark(
        [gold()],
        [prediction("different", normalized_fact_digest=FACT, evidence_chain=chain())],
    ).overall
    assert scores.exact_agreement == AbstractionMetric(0, 1)
    assert scores.evidence_support == AbstractionMetric(1, 1)


def test_abstained_and_missing_outputs_do_not_have_source_support_trials():
    for outputs in (
        [],
        [prediction(None, normalized_fact_digest=FACT, evidence_chain=chain())],
    ):
        scores = run_chart_abstraction_benchmark([gold(None)], outputs).overall
        assert scores.evidence_support == AbstractionMetric(0, 0)
        assert scores.evidence_support.interval is None
        assert scores.exact_agreement.interval is None


def test_wilson_interval_known_vector_and_empty_denominator():
    assert AbstractionMetric(5, 10).interval == pytest.approx((0.23659309, 0.76340691))
    assert AbstractionMetric(0, 0).to_dict()["interval"] is None
    assert AbstractionMetric(0, 1).interval[1] == pytest.approx(0.79345069)
    assert AbstractionMetric(1, 1).interval[0] == pytest.approx(0.20654931)
    with pytest.raises(ValueError, match="^invalid_metric_counts$"):
        AbstractionMetric(2, 1)


@pytest.mark.parametrize(
    "rows,outputs,code",
    [
        ([], [], "invalid_gold_fields"),
        ([gold(), gold()], [], "duplicate_gold_field"),
        ([gold()], [prediction(), prediction()], "duplicate_prediction_field"),
        (
            [gold()],
            [replace(prediction(), field_id="registry.unknown")],
            "unknown_prediction_field",
        ),
        (
            [gold()],
            [replace(prediction(), case_id="unknown")],
            "unknown_prediction_field",
        ),
        (
            [gold(), replace(gold(1, AbstractionFieldType.NUMBER), case_id="case_2")],
            [],
            "inconsistent_field_type",
        ),
        (["private-payload"], [], "invalid_gold_fields"),
        ([gold()], ["private-payload"], "invalid_predictions"),
    ],
)
def test_invalid_joins_fail_with_controlled_codes(rows, outputs, code):
    with pytest.raises(ValueError, match=f"^{code}$"):
        run_chart_abstraction_benchmark(rows, outputs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {}, []])
def test_non_scalar_or_nonfinite_values_are_rejected(value):
    with pytest.raises(ValueError, match="^invalid_value$"):
        prediction(value)


def test_reports_repr_and_errors_do_not_leak_input_canaries():
    canary = "SYNTHETIC_PRIVATE_NAME_123 /private/source credential-test"
    row = replace(gold(canary), case_id=canary)
    output = replace(
        prediction(canary),
        case_id=canary,
        normalized_fact_digest=FACT,
        evidence_chain=chain(),
    )
    report = run_chart_abstraction_benchmark([row], [output])
    for rendered in (repr(row), repr(output), repr(report), report.to_json()):
        assert canary not in rendered
        assert FACT not in rendered
        assert OTHER not in rendered
    with pytest.raises(ValueError, match="^invalid_field_id$") as error:
        replace(output, field_id=canary)
    assert canary not in str(error.value)
    with pytest.raises(ValueError, match="^invalid_fact_digest$") as error:
        replace(output, normalized_fact_digest=canary)
    assert canary not in str(error.value)
    allowed_labels = {
        "overall",
        "by_field",
        "by_field_type",
        "registry.label",
        "text",
        "field_count",
        "answerable_count",
        "unanswerable_count",
        "missing_count",
        "answered_count",
        "abstained_count",
        "exact_agreement",
        "normalized_agreement",
        "abstention_correctness",
        "answerable_abstention",
        "unanswerable_answer",
        "evidence_support",
        "success_count",
        "total_count",
        "interval",
    }

    def assert_content_free(value):
        if isinstance(value, dict):
            assert set(value) <= allowed_labels
            for item in value.values():
                assert_content_free(item)
        elif isinstance(value, list):
            for item in value:
                assert_content_free(item)
        else:
            assert value is None or type(value) in (int, float)

    assert_content_free(report.to_dict())
