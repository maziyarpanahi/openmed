"""Tests for redaction preview records."""

from __future__ import annotations

import json
from datetime import datetime

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.core.redaction_preview import (
    redaction_preview,
    render_redaction_preview,
)


def _entity(
    text: str,
    label: str,
    start: int,
    end: int,
    redacted_text: str,
) -> PIIEntity:
    return PIIEntity(
        text=text,
        label=label,
        start=start,
        end=end,
        confidence=0.99,
        entity_type=label,
        original_text=text,
        redacted_text=redacted_text,
        action="mask",
    )


def _result() -> DeidentificationResult:
    text = "Maria Garcia called 555-0101."
    return DeidentificationResult(
        original_text=text,
        deidentified_text="[NAME] called [PHONE].",
        pii_entities=[
            _entity("555-0101", "PHONE", 20, 28, "[PHONE]"),
            _entity("Maria Garcia", "NAME", 0, 12, "[NAME]"),
        ],
        method="mask",
        timestamp=datetime(2026, 6, 25),
    )


def test_redaction_preview_orders_change_records_by_offsets():
    preview = redaction_preview(_result().original_text, _result())

    assert preview["change_count"] == 2
    assert [
        (change["start"], change["end"], change["label"], change["action"])
        for change in preview["changes"]
    ] == [
        (0, 12, "NAME", "mask"),
        (20, 28, "PHONE", "mask"),
    ]
    assert preview["changes"][0]["offsets"] == {"start": 0, "end": 12}


def test_offsets_only_mode_omits_plaintext_surfaces_and_is_json_serializable():
    result = _result()
    preview = redaction_preview(result.original_text, result)
    payload = json.dumps(preview, sort_keys=True)

    assert "Maria Garcia" not in payload
    assert "555-0101" not in payload
    assert "[NAME]" not in payload
    assert "[PHONE]" not in payload
    assert "original_text" not in preview["changes"][0]
    assert "replacement_text" not in preview["changes"][0]
    assert preview["changes"][0]["surface_hash"].startswith("sha256:")
    assert preview["changes"][0]["replacement_hash"].startswith("sha256:")


def test_full_mode_includes_surfaces_for_local_review():
    result = _result()
    preview = redaction_preview(result.original_text, result, offsets_only=False)

    assert preview["mode"] == "full"
    assert preview["changes"][0]["original_text"] == "Maria Garcia"
    assert preview["changes"][0]["replacement_text"] == "[NAME]"
    assert '"Maria Garcia" -> "[NAME]"' in preview["summary"]


def test_preview_renders_readable_multi_span_offsets_summary():
    result = _result()
    preview = redaction_preview(result.original_text, result)

    summary = render_redaction_preview(preview)

    assert summary.startswith("Redaction preview: 2 changes")
    assert "0:12 NAME mask" in summary
    assert "20:28 PHONE mask" in summary
    assert "surface=sha256:" in summary
    assert "Maria Garcia" not in summary


def test_redaction_preview_accepts_serialized_result_payload():
    result = _result()
    payload = result.to_dict()

    preview = redaction_preview(result.original_text, payload)

    assert preview["change_count"] == 2
    assert preview["changes"][1]["replacement_hash"].startswith("sha256:")


def test_redaction_preview_uses_mapping_when_entity_replacement_is_absent():
    result = _result()
    entity = result.pii_entities[0]
    entity.redacted_text = None
    result.mapping = {"[PHONE]": "555-0101"}

    preview = redaction_preview(result.original_text, result, offsets_only=False)

    assert preview["changes"][1]["replacement_text"] == "[PHONE]"
