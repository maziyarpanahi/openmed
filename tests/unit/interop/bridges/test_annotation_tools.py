"""Tests for clean-room annotation tool row adapters."""

from __future__ import annotations

import pytest

from openmed.eval.annotation import AnnotationInterchangeError
from openmed.interop.bridges.annotation_tools import (
    export_fact_correction_rows,
    export_registry_label_rows,
    import_fact_correction_rows,
    import_registry_label_rows,
)


def test_fact_correction_rows_round_trip_without_source_text() -> None:
    rows = (
        {
            "annotation_id": "annotation_aaaaaaaaaaaa",
            "document_id": "document_aaaaaaaaaaaa",
            "evidence_ids": ["evidence_aaaaaaaaaaaa"],
            "fact_id": "fact_aaaaaaaaaaaa",
            "field": "assertion",
            "reason_code": "reviewed_correction",
            "replacement_code": "absent",
        },
    )

    envelope = import_fact_correction_rows(rows)

    assert export_fact_correction_rows(envelope) == rows
    assert "source_text" not in envelope.to_dict()["records"][0]


def test_registry_label_rows_round_trip() -> None:
    rows = (
        {
            "annotation_id": "annotation_bbbbbbbbbbbb",
            "document_id": "document_bbbbbbbbbbbb",
            "evidence_ids": [],
            "label": "eligible",
            "record_id": "record_aaaaaaaaaaaa",
            "registry_id": "registry_aaaaaaaaaaaa",
        },
    )

    envelope = import_registry_label_rows(rows)

    assert export_registry_label_rows(envelope) == rows


def test_tool_rows_reject_unknown_or_raw_fields() -> None:
    with pytest.raises(AnnotationInterchangeError, match="fields"):
        import_fact_correction_rows(
            (
                {
                    "annotation_id": "annotation_aaaaaaaaaaaa",
                    "document_id": "document_aaaaaaaaaaaa",
                    "fact_id": "fact_aaaaaaaaaaaa",
                    "field": "assertion",
                    "reason_code": "reviewed_correction",
                    "replacement_code": "absent",
                    "evidence_ids": [],
                    "source_text": "synthetic-canary",
                },
            )
        )
