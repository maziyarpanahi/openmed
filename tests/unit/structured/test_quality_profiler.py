from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from openmed.structured.quality import (
    QualityGateError,
    assert_profile_gate,
    profile_extracted_batch,
    render_profile_summary,
)

NOTE_TEXT = (
    "Alice reports diabetes. Aspirin started. Glucose 95 mg/dL. "
    "Lactate 8.0 mmol/L. HR 300 bpm. Mystery term noted."
)


def _entity(
    surface: str,
    *,
    domain: str,
    concept_id: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    start = NOTE_TEXT.index(surface)
    entity = {
        "text": surface,
        "domain_id": domain,
        "start": start,
        "end": start + len(surface),
        "concept_id": concept_id,
    }
    entity.update(extra)
    return entity


def _fixture_records() -> list[dict[str, Any]]:
    return [
        {
            "document_id": "secret-note-1",
            "person_id": "secret-patient-1",
            "note_date": "2026-01-02",
            "note_text": NOTE_TEXT,
            "entities": [
                _entity("diabetes", domain="Condition", concept_id=201826),
                _entity("Aspirin", domain="Drug", concept_id=1112807),
                _entity(
                    "Glucose",
                    domain="Measurement",
                    concept_id=3004410,
                    value=95,
                    unit="mg/dL",
                    reference_range="70-99 mg/dL",
                ),
                _entity(
                    "Lactate",
                    domain="Measurement",
                    concept_id=3004249,
                    value=8.0,
                    unit="mmol/L",
                    reference_range="0.5-2.2 mmol/L",
                    metadata={"effective_date": "2099-01-01"},
                ),
                _entity("HR 300 bpm", domain="Measurement", concept_id=3027018),
                _entity("Mystery term", domain="Condition"),
            ],
        }
    ]


def test_profile_reports_known_coverage_and_domain_counts() -> None:
    report = profile_extracted_batch(
        _fixture_records(),
        completeness_floor=0.8,
        reference_date=date(2026, 1, 2),
    )

    assert report.pipeline_gate["passed"] is True
    assert report.totals["spans"] == 6
    assert report.totals["grounded_spans"] == 5
    assert report.totals["ungrounded_spans"] == 1
    assert report.overall_completeness_score == pytest.approx(0.916667)

    coverage = {item.domain: item for item in report.domain_grounding}
    assert coverage["condition"].to_dict() == {
        "domain": "condition",
        "total": 2,
        "grounded": 1,
        "ungrounded": 1,
        "coverage": 0.5,
    }
    assert coverage["drug"].grounded == 1
    assert coverage["drug"].coverage == 1.0
    assert coverage["measurement"].grounded == 3
    assert coverage["measurement"].coverage == 1.0


def test_profile_flags_seeded_plausibility_issues_without_raw_values() -> None:
    report = profile_extracted_batch(
        _fixture_records(),
        reference_date=date(2026, 1, 2),
    )
    payload = json.dumps(report.to_dict(), sort_keys=True)

    reasons = {
        issue["reason"]
        for check in report.to_dict()["checks"]
        for issue in check["issues"]
    }
    assert "out_of_reference_range_high" in reasons
    assert "vital_value_out_of_plausible_range" in reasons
    assert "date_after_reference_window" in reasons

    assert NOTE_TEXT not in payload
    assert "Alice" not in payload
    assert "secret-note-1" not in payload
    assert "secret-patient-1" not in payload
    assert "8.0" not in payload
    assert "2099-01-01" not in payload
    assert "HR 300 bpm" not in payload


def test_profile_gate_blocks_batches_below_floor() -> None:
    report = profile_extracted_batch(
        _fixture_records(),
        completeness_floor=0.95,
        reference_date=date(2026, 1, 2),
    )

    assert report.status == "fail"
    with pytest.raises(QualityGateError, match="quality profile gate failed"):
        assert_profile_gate(report)


def test_human_summary_is_phi_free() -> None:
    report = profile_extracted_batch(
        _fixture_records(),
        reference_date=date(2026, 1, 2),
    )

    summary = render_profile_summary(report)

    assert "completeness=0.917" in summary
    assert "Alice" not in summary
    assert NOTE_TEXT not in summary
