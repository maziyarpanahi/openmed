from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.interop.elasticsearch import (
    DEFAULT_ELASTICSEARCH_GROK_PATTERNS,
    ElasticsearchProcessorDiagnostics,
    ElasticsearchRedactionConfig,
    ElasticsearchRedactionError,
    ElasticsearchRedactionProcessor,
    UnsupportedDynamicFieldError,
    build_ingest_pipeline,
)

_SYNTHETIC_VALUE = "synthetic-person-001@example.test"


def _fake_redactor(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        deidentified_text=text.replace(_SYNTHETIC_VALUE, "[EMAIL]"),
        pii_entities=[object()] if _SYNTHETIC_VALUE in text else [],
    )


def test_pipeline_is_explicit_deterministic_and_offline() -> None:
    processor = ElasticsearchRedactionProcessor(
        fields={
            "message": ("%{EMAILADDRESS:email}",),
            "clinical.note": ("%{IP:client_ip}",),
        }
    )

    first = processor.to_ingest_pipeline()
    second = processor.to_ingest_pipeline()

    assert first == second
    assert first["processors"] == [
        {
            "redact": {
                "field": "message",
                "patterns": ["%{EMAILADDRESS:email}"],
                "prefix": "[REDACTED]",
                "suffix": "",
                "ignore_missing": True,
                "tag": "openmed-redaction-0",
            }
        },
        {
            "redact": {
                "field": "clinical.note",
                "patterns": ["%{IP:client_ip}"],
                "prefix": "[REDACTED]",
                "suffix": "",
                "ignore_missing": True,
                "tag": "openmed-redaction-1",
            }
        },
    ]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert processor.config.patterns == DEFAULT_ELASTICSEARCH_GROK_PATTERNS


@pytest.mark.parametrize("field", ["message.*", "message[0]", "{{field_name}}"])
def test_dynamic_field_paths_are_rejected(field: str) -> None:
    with pytest.raises(UnsupportedDynamicFieldError, match="dynamic"):
        ElasticsearchRedactionConfig(fields=[field])


def test_selected_fields_are_redacted_without_mutating_input_and_diagnostics_are_counts_only() -> (
    None
):
    document = {
        "_index": "synthetic-index",
        "_source": {
            "message": f"contact {_SYNTHETIC_VALUE}",
            "metadata": {"kind": "synthetic"},
        },
    }
    original = deepcopy(document)
    processor = ElasticsearchRedactionProcessor(fields=["message"])

    result = processor.process(document, redactor=_fake_redactor)

    assert result.document["_source"]["message"] == "contact [EMAIL]"
    assert result.document["_source"]["metadata"] == {"kind": "synthetic"}
    assert document == original
    assert result.diagnostics == ElasticsearchProcessorDiagnostics(
        documents_processed=1,
        fields_configured=1,
        fields_processed=1,
        fields_redacted=1,
        fields_skipped=0,
        spans_redacted=1,
        dynamic_fields_rejected=0,
    )
    assert result.to_dict() == {
        "documents_processed": 1,
        "fields_configured": 1,
        "fields_processed": 1,
        "fields_redacted": 1,
        "fields_skipped": 0,
        "spans_redacted": 1,
        "dynamic_fields_rejected": 0,
    }
    assert _SYNTHETIC_VALUE not in json.dumps(result.to_dict())


def test_dynamic_document_values_fail_closed_without_exposing_value() -> None:
    processor = ElasticsearchRedactionProcessor(fields=["message"])
    document: dict[str, Any] = {"_source": {"message": {"runtime": _SYNTHETIC_VALUE}}}

    with pytest.raises(UnsupportedDynamicFieldError) as exc_info:
        processor.diagnose(document)

    assert _SYNTHETIC_VALUE not in str(exc_info.value)
    assert "dynamic" in str(exc_info.value)


def test_redactor_failures_are_value_free() -> None:
    processor = ElasticsearchRedactionProcessor(fields=["message"])

    def failing_redactor(text: str) -> str:
        raise RuntimeError(f"model detail: {text}")

    with pytest.raises(ElasticsearchRedactionError) as exc_info:
        processor.process(
            {"message": _SYNTHETIC_VALUE},
            redactor=failing_redactor,
        )

    assert str(exc_info.value) == "failed to redact a configured ingest field"
    assert _SYNTHETIC_VALUE not in str(exc_info.value)


def test_missing_and_empty_fields_are_counted_as_skipped() -> None:
    processor = ElasticsearchRedactionProcessor(
        fields=["message", "clinical.note", "empty"]
    )

    diagnostics = processor.diagnose(
        {"_source": {"empty": "", "clinical": {"note": None}}}
    )

    assert diagnostics.to_dict() == {
        "documents_processed": 1,
        "fields_configured": 3,
        "fields_processed": 0,
        "fields_redacted": 0,
        "fields_skipped": 3,
        "spans_redacted": 0,
        "dynamic_fields_rejected": 0,
    }


def test_builder_and_selected_fields_alias_are_equivalent() -> None:
    config = ElasticsearchRedactionConfig(
        selected_fields=["message"],
        patterns=["%{EMAILADDRESS:email}"],
    )

    assert (
        build_ingest_pipeline(["message"], patterns=["%{EMAILADDRESS:email}"])
        == config.to_pipeline()
    )
