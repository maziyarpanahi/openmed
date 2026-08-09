from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, get_adapter
from openmed.interop import opensearch as opensearch_adapter
from openmed.interop.opensearch import (
    OpenSearchRedactionError,
    OpenSearchRedactionProcessor,
)

SOURCE_MARKERS = ("Synthetic Person", "synthetic-555-0100")


def fake_deidentifier(text: str, **kwargs):
    assert kwargs["policy"] == "hipaa_safe_harbor"
    assert kwargs["method"] == "mask"
    redacted = text.replace("Synthetic Person", "[PERSON]")
    redacted = redacted.replace("synthetic-555-0100", "[PHONE]")
    return SimpleNamespace(deidentified_text=redacted)


def test_registry_exposes_opensearch_without_client_dependency() -> None:
    adapter = get_adapter("opensearch")

    assert adapter.__name__ == "openmed.interop.opensearch"
    assert adapter_spec("opensearch").extra == ""
    assert adapter.OpenSearchRedactionProcessor is OpenSearchRedactionProcessor


def test_processor_redacts_only_selected_nested_fields_and_copies_document() -> None:
    document = {
        "message": "Synthetic Person called synthetic-555-0100",
        "title": "Synthetic Person",
        "metadata": {"note": "Synthetic Person"},
    }
    processor = OpenSearchRedactionProcessor(
        fields=("message", "metadata.note"),
        deidentifier=fake_deidentifier,
    )

    redacted = processor.execute(document)

    assert redacted == {
        "message": "[PERSON] called [PHONE]",
        "title": "Synthetic Person",
        "metadata": {"note": "[PERSON]"},
    }
    assert document["message"] == "Synthetic Person called synthetic-555-0100"
    assert document["metadata"]["note"] == "Synthetic Person"


def test_processor_reports_counts_without_source_values() -> None:
    document = {"message": "Synthetic Person", "tags": ["synthetic-555-0100"]}
    processor = OpenSearchRedactionProcessor(
        fields=("message", "tags"),
        deidentifier=fake_deidentifier,
    )

    redacted, report = processor.process_with_report(document)

    assert redacted == {"message": "[PERSON]", "tags": ["[PHONE]"]}
    assert report.to_dict() == {
        "adapter": "opensearch",
        "policy": "hipaa_safe_harbor",
        "fields": ["message", "tags"],
        "values_seen": 2,
        "values_redacted": 2,
        "spans_redacted": 2,
    }
    report_text = repr(report.to_dict())
    assert all(marker not in report_text for marker in SOURCE_MARKERS)


def test_policy_validation_is_explicit_and_value_free() -> None:
    invalid_policy = "synthetic-invalid-policy"

    with pytest.raises(OpenSearchRedactionError) as error:
        OpenSearchRedactionProcessor(
            policy=invalid_policy,
            deidentifier=fake_deidentifier,
        )

    assert str(error.value) == "policy is invalid"
    assert invalid_policy not in str(error.value)


def test_deidentifier_errors_do_not_include_source_values() -> None:
    source = "Synthetic Person has synthetic-555-0100"

    def failing_deidentifier(text: str, **kwargs):
        del text, kwargs
        raise RuntimeError(source)

    processor = OpenSearchRedactionProcessor(
        field="message",
        deidentifier=failing_deidentifier,
    )

    with pytest.raises(OpenSearchRedactionError) as error:
        processor.process({"message": source})

    assert str(error.value) == "redaction failed"
    assert source not in str(error.value)


def test_missing_fields_can_be_ignored_without_network_egress(monkeypatch) -> None:
    def blocked(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unexpected socket egress")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)

    processor = OpenSearchRedactionProcessor(
        fields=("message", "optional_note"),
        ignore_missing=True,
        deidentifier=fake_deidentifier,
    )

    redacted, report = processor.process_with_report({"message": "Synthetic Person"})

    assert redacted == {"message": "[PERSON]"}
    assert report.values_seen == 1


def test_default_deidentifier_is_configured_cache_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_default_deidentifier(text: str, **kwargs):
        captured.update(kwargs)
        return text.replace("Synthetic Person", "[PERSON]")

    monkeypatch.setattr(
        opensearch_adapter,
        "_default_deidentifier",
        fake_default_deidentifier,
    )

    redacted = OpenSearchRedactionProcessor().process({"text": "Synthetic Person"})

    assert redacted == {"text": "[PERSON]"}
    assert captured["policy"] == "hipaa_safe_harbor"
    assert captured["method"] == "mask"
    assert getattr(captured["config"], "local_only") is True


def test_selected_non_text_values_raise_safe_error() -> None:
    processor = OpenSearchRedactionProcessor(
        field="message",
        deidentifier=fake_deidentifier,
    )

    with pytest.raises(
        OpenSearchRedactionError, match="selected field must contain text"
    ):
        processor.process({"message": {"raw": "Synthetic Person"}})
