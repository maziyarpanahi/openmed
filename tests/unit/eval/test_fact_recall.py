"""Focused tests for the structured clinical-fact recall metric."""

from __future__ import annotations

from openmed.eval.metrics import (
    build_summary_faithfulness_report,
    compute_metrics_bundle,
    extract_clinical_facts,
    fact_recall,
    summary_faithfulness_metrics,
)


def test_fact_recall_reports_omissions_and_unsupported_facts() -> None:
    source = (
        ("problem", "hypertension"),
        ("medication", "lisinopril"),
        ("lab", "creatinine", "1.1", "mg/dL"),
    )
    summary = (
        ("problem", "hypertension"),
        ("lab", "creatinine", "1.1", "mg/dL"),
        ("problem", "pneumonia"),
    )

    result = fact_recall(source, summary)

    assert result.recall == 2 / 3
    assert ("medication", "lisinopril") in result["omitted"]
    assert ("problem", "pneumonia") in result["unsupported"]


def test_fact_recall_normalizes_fact_type_aliases_and_duplicate_facts() -> None:
    result = fact_recall(
        [
            {"fact_type": "diagnosis", "value": "Type 2 diabetes"},
            {"fact_type": "diagnosis", "value": "type 2 diabetes"},
        ],
        [("problem", "TYPE 2 DIABETES")],
    )

    assert result == {
        "recall": 1.0,
        "omitted": [],
        "unsupported": [],
    }


def test_extract_clinical_facts_uses_om043_relation_extractors() -> None:
    text = "Pneumonia. Lisinopril 10 mg daily. Sodium 130 mmol/L (135-145) L."
    spans = [
        _span(text, "Pneumonia", "PROBLEM"),
        _span(text, "Lisinopril", "MEDICATION"),
        _span(text, "10 mg", "DOSAGE"),
        _span(text, "daily", "FREQUENCY"),
        _span(text, "Sodium", "LAB_TEST"),
        _span(text, "130", "LAB_VALUE"),
        _span(text, "mmol/L", "UNIT"),
        _span(text, "135-145", "REFERENCE_RANGE"),
        _span(text, "L", "ABNORMAL_FLAG", start=text.rindex("L")),
    ]

    facts = extract_clinical_facts(text, spans)

    assert ("problem", "Pneumonia") in facts
    assert ("medication", "Lisinopril") in facts
    assert ("lab", "Sodium", "130", "mmol/L", "low") in facts


def test_summary_faithfulness_metrics_can_be_serialized_with_rouge() -> None:
    source_text = "Hypertension."
    summary_text = "Hypertension. Pneumonia."
    source_spans = [_span(source_text, "Hypertension", "PROBLEM")]
    summary_spans = [
        _span(summary_text, "Hypertension", "PROBLEM"),
        _span(summary_text, "Pneumonia", "PROBLEM"),
    ]

    metrics = summary_faithfulness_metrics(
        source_text,
        summary_text,
        source_spans,
        summary_spans,
        rouge={"rougeL": 0.75},
    )
    report = build_summary_faithfulness_report(
        extract_clinical_facts(source_text, source_spans),
        extract_clinical_facts(summary_text, summary_spans),
        rouge={"rougeL": 0.75},
    )

    assert metrics["fact_recall"]["recall"] == 1.0
    assert metrics["fact_recall"]["unsupported"] == [["problem", "Pneumonia"]]
    payload = report.to_dict()
    assert payload["metrics"]["rouge"] == {"rougeL": 0.75}
    assert payload["metrics"]["fact_recall"]["unsupported"] == [
        ["problem", "Pneumonia"]
    ]


def test_metrics_bundle_exposes_fact_recall() -> None:
    bundle = compute_metrics_bundle(
        [],
        [],
        source_facts=[("medication", "lisinopril")],
        summary_facts=[("problem", "pneumonia")],
    )

    assert bundle["fact_recall"]["recall"] == 0.0
    assert bundle["fact_recall"]["omitted"] == [["medication", "lisinopril"]]
    assert bundle["fact_recall"]["unsupported"] == [["problem", "pneumonia"]]


def _span(
    text: str,
    surface: str,
    label: str,
    *,
    start: int | None = None,
) -> dict[str, object]:
    offset = text.index(surface) if start is None else start
    return {
        "text": surface,
        "label": label,
        "start": offset,
        "end": offset + len(surface),
        "score": 1.0,
    }
