from __future__ import annotations

import pytest

from openmed.eval.error_analysis import (
    FAITHFULNESS_DISCLAIMER,
    faithfulness_report,
)
from openmed.eval.harness import BenchmarkFixture, run_benchmark
from openmed.eval.metrics import (
    EvalSpan,
    compute_metrics_bundle,
    compute_span_grounded_faithfulness,
)


def test_grounded_facts_have_zero_ungrounded_rate() -> None:
    text = "Patient has hypertension and takes metformin 500 mg."
    diagnosis = _span(text, "hypertension")
    medication = _span(text, "metformin 500 mg")

    result = compute_span_grounded_faithfulness(
        [
            {
                "id": "dx-1",
                "fact_type": "diagnosis",
                "value": "hypertension",
                "supporting_span": {
                    "start": diagnosis[0],
                    "end": diagnosis[1],
                    "text": "hypertension",
                },
            },
            {
                "id": "med-1",
                "fact_type": "medication",
                "value": "metformin 500 mg",
                "supporting_span": {
                    "start": medication[0],
                    "end": medication[1],
                    "text": "metformin 500 mg",
                },
            },
        ],
        source_text=text,
    )

    assert result.ungrounded_fact_rate == 0.0
    assert result.ungrounded_facts == 0
    assert result.by_fact_type["diagnosis"]["rate"] == 0.0


@pytest.mark.parametrize(
    ("support", "reason"),
    [
        (None, "missing_supporting_span"),
        ({"start": 12, "end": 12}, "supporting_span_out_of_bounds"),
        ({"start": 12, "end": 99}, "supporting_span_out_of_bounds"),
    ],
)
def test_missing_or_invalid_supporting_spans_are_ungrounded(
    support: dict[str, int] | None,
    reason: str,
) -> None:
    fact: dict[str, object] = {
        "fact_type": "diagnosis",
        "value": "hypertension",
    }
    if support is not None:
        fact["supporting_span"] = support

    result = compute_span_grounded_faithfulness(
        [fact],
        source_text="Patient has hypertension.",
    )

    assert result.ungrounded_fact_rate == 1.0
    assert result.by_fact_type["diagnosis"] == {
        "rate": 1.0,
        "total": 1,
        "ungrounded": 1,
    }
    assert result.findings[0].reason == reason


def test_normalized_contiguous_span_reconstructs_fact_value() -> None:
    text = "Patient takes metformin 500 mg."
    start, end = _span(text, "metformin 500 mg")
    facts = [
        {
            "fact_type": "medication",
            "value": "metformin 500mg",
            "supporting_span": {"start": start, "end": end},
        }
    ]

    result = compute_span_grounded_faithfulness(facts, source_text=text)
    bundle = compute_metrics_bundle(
        [],
        [],
        extracted_facts=facts,
        source_text=text,
    )

    assert result.ungrounded_fact_rate == 0.0
    assert bundle["faithfulness"]["ungrounded_fact_rate"] == 0.0


def test_supporting_span_text_mismatch_is_ungrounded() -> None:
    text = "Patient has hypertension."
    start, end = _span(text, "hypertension")

    result = compute_span_grounded_faithfulness(
        [
            {
                "id": "dx-1",
                "fact_type": "diagnosis",
                "value": "hypertension",
                "supporting_span": {
                    "start": start,
                    "end": end,
                    "text": "diabetes",
                },
            }
        ],
        source_text=text,
        fixture_id="note-1",
    )

    assert result.ungrounded_fact_rate == 1.0
    finding = result.findings[0]
    assert finding.reason == "supporting_span_text_mismatch"
    assert finding.start == start
    assert finding.end == end


def test_fabricated_diagnosis_is_reported_with_disclaimer_and_offsets() -> None:
    text = "Patient has hypertension."
    start, end = _span(text, "hypertension")

    report = faithfulness_report(
        [
            {
                "fact_type": "diagnosis",
                "name": "pneumonia",
                "supporting_span": {
                    "start": start,
                    "end": end,
                    "text": "hypertension",
                },
            }
        ],
        source_text=text,
        fixture_id="note-1",
    )

    payload = report.to_dict()
    assert payload["disclaimer"] == FAITHFULNESS_DISCLAIMER
    assert payload["faithfulness"]["ungrounded_fact_rate"] == 1.0
    assert payload["ungrounded_facts"][0]["start"] == start
    assert payload["ungrounded_facts"][0]["end"] == end
    assert payload["ungrounded_facts"][0]["fact_id"] == "fact-0"
    assert payload["ungrounded_facts"][0]["span_text"] == "hypertension"
    assert "pneumonia" not in report.to_json(include_span_text=False)
    assert "assistive safeguard" in report.to_markdown()
    assert "not a clinical decision" in report.to_markdown()


def test_harness_attaches_faithfulness_metric_from_fixture_metadata() -> None:
    text = "Patient has hypertension."
    start, end = _span(text, "hypertension")
    fixture = BenchmarkFixture(
        fixture_id="note-1",
        text=text,
        gold_spans=(
            EvalSpan(start=start, end=end, label="CONDITION", text="hypertension"),
        ),
        metadata={
            "extracted_facts": [
                {
                    "id": "dx-1",
                    "fact_type": "diagnosis",
                    "value": "hypertension",
                    "supporting_span": {
                        "start": start,
                        "end": end,
                        "text": "hypertension",
                    },
                }
            ]
        },
    )

    report = run_benchmark(
        [fixture],
        suite="faithfulness",
        model_name="unit-model",
        runner=lambda fixture, _model, _device: fixture.gold_spans,
    )

    assert report.metrics["faithfulness"]["ungrounded_fact_rate"] == 0.0
    assert report.metrics["faithfulness"]["total_facts"] == 1


def _span(text: str, value: str) -> tuple[int, int]:
    start = text.index(value)
    return start, start + len(value)
