"""Tests for single-document de-identification redaction previews."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openmed.core.audit import hash_text
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.core.redaction_preview import redaction_preview


def _result() -> DeidentificationResult:
    text = "Patient John Doe called 555-1234."

    # Deliberately unordered: preview output must be ordered by offsets.
    entities = [
        PIIEntity(
            text="555-1234",
            label="PHONE",
            start=24,
            end=32,
            confidence=0.99,
            redacted_text="[PHONE]",
            action="mask",
        ),
        PIIEntity(
            text="John Doe",
            label="NAME",
            start=8,
            end=16,
            confidence=0.95,
            redacted_text="[NAME]",
            action="mask",
        ),
    ]

    return DeidentificationResult(
        original_text=text,
        deidentified_text="Patient [NAME] called [PHONE].",
        pii_entities=entities,
        method="mask",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_redaction_preview_returns_sorted_full_change_records():
    result = _result()

    preview = redaction_preview(result.original_text, result)

    assert [change.start for change in preview.changes] == [8, 24]
    assert [change.end for change in preview.changes] == [16, 32]

    assert preview.to_dict() == {
        "offsets_only": False,
        "changes": [
            {
                "start": 8,
                "end": 16,
                "label": "NAME",
                "action": "mask",
                "original": "John Doe",
                "replacement": "[NAME]",
                "original_hash": hash_text("John Doe"),
                "replacement_hash": hash_text("[NAME]"),
            },
            {
                "start": 24,
                "end": 32,
                "label": "PHONE",
                "action": "mask",
                "original": "555-1234",
                "replacement": "[PHONE]",
                "original_hash": hash_text("555-1234"),
                "replacement_hash": hash_text("[PHONE]"),
            },
        ],
    }


def test_offsets_only_preview_contains_no_plaintext_surfaces():
    result = _result()

    preview = redaction_preview(result.original_text, result, offsets_only=True)
    payload = preview.to_dict()
    serialized = preview.to_json()

    assert payload["offsets_only"] is True
    assert set(payload["changes"][0]) == {
        "start",
        "end",
        "label",
        "action",
        "original_hash",
        "replacement_hash",
    }

    for surface in ("John Doe", "555-1234", "[NAME]", "[PHONE]"):
        assert surface not in serialized

    assert json.loads(serialized) == payload


def test_preview_renders_readable_full_and_offsets_only_summaries():
    result = _result()

    full_text = redaction_preview(result.original_text, result).to_text()
    offsets_only_text = redaction_preview(
        result.original_text,
        result,
        offsets_only=True,
    ).to_text()

    assert "2 redaction change(s)" in full_text
    assert "[8:16] NAME (mask)" in full_text
    assert "John Doe" in full_text
    assert "[NAME]" in full_text

    assert "offsets-only" in offsets_only_text
    assert "John Doe" not in offsets_only_text
    assert "[NAME]" not in offsets_only_text
    assert "sha256:" in offsets_only_text
