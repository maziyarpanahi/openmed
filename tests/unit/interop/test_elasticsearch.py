from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.interop import elasticsearch as elasticsearch_adapter
from openmed.interop.elasticsearch import (
    DEFAULT_ELASTICSEARCH_GROK_PATTERNS,
    ElasticsearchFieldRule,
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


def test_configuration_and_serialization_are_bounded(monkeypatch) -> None:
    monkeypatch.setattr(elasticsearch_adapter, "_MAX_FIELDS", 2)
    with pytest.raises(ValueError, match="too many entries"):
        ElasticsearchRedactionConfig(fields=["one", "two", "three"])

    monkeypatch.setattr(elasticsearch_adapter, "_MAX_PATTERNS_PER_FIELD", 2)
    with pytest.raises(ValueError, match="too many entries"):
        ElasticsearchRedactionConfig(
            fields=["message"],
            patterns=["one", "two", "three"],
        )

    config = ElasticsearchRedactionConfig(fields=["message"], patterns=["one"])
    with pytest.raises(ValueError, match="indent must be bounded"):
        config.to_json(indent=True)
    with pytest.raises(ValueError, match="indent must be bounded"):
        config.to_json(indent=17)


def test_document_copy_is_bounded_and_rejects_cycles(monkeypatch) -> None:
    monkeypatch.setattr(elasticsearch_adapter, "_MAX_DOCUMENT_ITEMS", 2)
    processor = ElasticsearchRedactionProcessor(fields=["message"])

    with pytest.raises(
        ElasticsearchRedactionError,
        match="failed to copy the ingest document",
    ):
        processor.diagnose({"one": 1, "two": 2, "three": 3})

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(
        ElasticsearchRedactionError,
        match="failed to copy the ingest document",
    ):
        processor.diagnose(cyclic)


def test_result_representation_hides_document_values() -> None:
    document = {"message": _SYNTHETIC_VALUE, "unselected": _SYNTHETIC_VALUE}
    processor = ElasticsearchRedactionProcessor(fields=["message"])

    result = processor.process(document, redactor=_fake_redactor)

    assert _SYNTHETIC_VALUE not in repr(result)
    assert result.document["unselected"] == _SYNTHETIC_VALUE


def test_hostile_boundary_failures_are_value_free() -> None:
    class SensitiveAbort(BaseException):
        pass

    class HostileDocument(dict):
        def items(self):
            raise SensitiveAbort(_SYNTHETIC_VALUE)

    processor = ElasticsearchRedactionProcessor(fields=["message"])

    with pytest.raises(ElasticsearchRedactionError) as copy_error:
        processor.diagnose(HostileDocument(message=_SYNTHETIC_VALUE))
    assert str(copy_error.value) == "failed to copy the ingest document"
    assert _SYNTHETIC_VALUE not in str(copy_error.value)

    def aborting_redactor(text: str):
        raise SensitiveAbort(text)

    with pytest.raises(ElasticsearchRedactionError) as redact_error:
        processor.process(
            {"message": _SYNTHETIC_VALUE},
            redactor=aborting_redactor,
        )
    assert str(redact_error.value) == "failed to redact a configured ingest field"
    assert _SYNTHETIC_VALUE not in str(redact_error.value)

    def spoofed_adapter_error(text: str):
        raise ElasticsearchRedactionError(text)

    with pytest.raises(ElasticsearchRedactionError) as spoofed_error:
        processor.process(
            {"message": _SYNTHETIC_VALUE},
            redactor=spoofed_adapter_error,
        )
    assert str(spoofed_error.value) == "failed to redact a configured ingest field"
    assert _SYNTHETIC_VALUE not in str(spoofed_error.value)


@pytest.mark.parametrize("fatal_error", [KeyboardInterrupt, SystemExit])
def test_process_preserves_interpreter_control_exceptions(fatal_error) -> None:
    def aborting_redactor(text: str):
        del text
        raise fatal_error

    processor = ElasticsearchRedactionProcessor(fields=["message"])

    with pytest.raises(fatal_error):
        processor.process(
            {"message": _SYNTHETIC_VALUE},
            redactor=aborting_redactor,
        )


def test_optional_entity_metadata_failure_does_not_expose_source() -> None:
    class SensitiveAbort(BaseException):
        pass

    class RedactionResult:
        deidentified_text = "[REDACTED]"

        @property
        def pii_entities(self):
            raise SensitiveAbort(_SYNTHETIC_VALUE)

    processor = ElasticsearchRedactionProcessor(fields=["message"])
    result = processor.process(
        {"message": _SYNTHETIC_VALUE},
        redactor=lambda text: RedactionResult(),
    )

    assert result.document["message"] == "[REDACTED]"
    assert result.diagnostics.spans_redacted == 1


def test_redactor_output_and_span_diagnostics_are_bounded(monkeypatch) -> None:
    monkeypatch.setattr(elasticsearch_adapter, "_MIN_OUTPUT_CHARS", 4)
    processor = ElasticsearchRedactionProcessor(fields=["message"])

    with pytest.raises(
        ElasticsearchRedactionError,
        match="failed to redact a configured ingest field",
    ):
        processor.process(
            {"message": "x"},
            redactor=lambda text: "x" * 9,
        )

    monkeypatch.setattr(elasticsearch_adapter, "_MAX_SPANS_PER_FIELD", 2)
    result = processor.process(
        {"message": "x"},
        redactor=lambda text: SimpleNamespace(
            deidentified_text="[R]",
            pii_entities=[object(), object(), object()],
        ),
    )
    assert result.diagnostics.spans_redacted == 2

    unchanged = processor.process(
        {"message": "x"},
        redactor=lambda text: SimpleNamespace(
            deidentified_text=text,
            pii_entities=[object(), object(), object()],
        ),
    )
    assert unchanged.diagnostics.fields_redacted == 0
    assert unchanged.diagnostics.spans_redacted == 0


def test_process_rejects_a_non_callable_redactor() -> None:
    processor = ElasticsearchRedactionProcessor(fields=["message"])

    with pytest.raises(TypeError, match="redactor must be callable"):
        processor.process({"message": _SYNTHETIC_VALUE}, redactor=object())


def test_prebuilt_rules_and_configs_are_snapshotted() -> None:
    rule = ElasticsearchFieldRule(
        field="message",
        patterns=("%{EMAILADDRESS:email}",),
    )
    config = ElasticsearchRedactionConfig(fields=[rule])
    processor = ElasticsearchRedactionProcessor(config)

    object.__setattr__(rule, "field", _SYNTHETIC_VALUE)
    object.__setattr__(config, "_fields", (_SYNTHETIC_VALUE,))

    assert processor.fields == ("message",)
    assert processor.to_ingest_pipeline()["processors"][0]["redact"]["field"] == (
        "message"
    )


def test_runtime_configuration_is_snapshotted_before_callbacks() -> None:
    processor = ElasticsearchRedactionProcessor(fields=["message", "note"])
    calls = 0

    def mutating_redactor(text: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            object.__setattr__(processor.config, "_rules", ())
        return "[REDACTED]"

    result = processor.process(
        {"message": _SYNTHETIC_VALUE, "note": _SYNTHETIC_VALUE},
        redactor=mutating_redactor,
    )

    assert result.document == {
        "message": "[REDACTED]",
        "note": "[REDACTED]",
    }
    assert calls == 2
    with pytest.raises(
        ElasticsearchRedactionError, match="processor configuration is invalid"
    ):
        processor.diagnose({"message": _SYNTHETIC_VALUE})


def test_corrupted_metadata_fails_without_echoing_values() -> None:
    config = ElasticsearchRedactionConfig(fields=["message"])
    object.__setattr__(config, "_fields", (_SYNTHETIC_VALUE,))
    with pytest.raises(ValueError) as config_error:
        config.to_dict()
    assert str(config_error.value) == "Elasticsearch configuration is invalid"
    assert _SYNTHETIC_VALUE not in str(config_error.value)

    diagnostics = ElasticsearchProcessorDiagnostics(
        documents_processed=1,
        fields_configured=1,
        fields_processed=1,
    )
    object.__setattr__(diagnostics, "spans_redacted", _SYNTHETIC_VALUE)
    with pytest.raises(ValueError) as diagnostics_error:
        diagnostics.to_dict()
    assert str(diagnostics_error.value) == "processor diagnostics are invalid"
    assert _SYNTHETIC_VALUE not in str(diagnostics_error.value)


def test_document_requires_bounded_json_scalars(monkeypatch) -> None:
    processor = ElasticsearchRedactionProcessor(fields=["message"])
    for value in (float("nan"), float("inf"), 1 << 80):
        with pytest.raises(
            ElasticsearchRedactionError,
            match="failed to copy the ingest document",
        ):
            processor.diagnose({"metadata": value})

    monkeypatch.setattr(elasticsearch_adapter, "_MAX_DOCUMENT_TOTAL_BYTES", 8)
    with pytest.raises(
        ElasticsearchRedactionError,
        match="failed to copy the ingest document",
    ):
        processor.diagnose({"metadata": "123456789"})


def test_redacted_document_aggregate_size_is_revalidated(monkeypatch) -> None:
    monkeypatch.setattr(elasticsearch_adapter, "_MAX_DOCUMENT_TOTAL_BYTES", 20)
    processor = ElasticsearchRedactionProcessor(fields=["message"])

    with pytest.raises(
        ElasticsearchRedactionError,
        match="failed to copy the ingest document",
    ):
        processor.process(
            {"message": "x"},
            redactor=lambda text: "y" * 21,
        )


def test_duplicate_field_paths_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate paths"):
        ElasticsearchRedactionConfig(fields=["message", "message"])
