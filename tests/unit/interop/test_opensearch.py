from __future__ import annotations

import socket
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, get_adapter
from openmed.interop import opensearch as opensearch_adapter
from openmed.interop.opensearch import (
    OpenSearchRedactionError,
    OpenSearchRedactionProcessor,
    RedactionReport,
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

    def fake_deidentify(text: str, **kwargs):
        captured.update(kwargs)
        return text.replace("Synthetic Person", "[PERSON]")

    monkeypatch.setattr(
        opensearch_adapter,
        "_default_deidentifier",
        lambda: fake_deidentify,
    )

    redacted = OpenSearchRedactionProcessor(
        deidentify_kwargs={
            "audit": True,
            "config": object(),
            "keep_mapping": True,
            "use_safety_sweep": False,
        }
    ).process({"text": "Synthetic Person"})

    assert redacted == {"text": "[PERSON]"}
    assert captured["policy"] == "hipaa_safe_harbor"
    assert captured["method"] == "mask"
    assert getattr(captured["config"], "local_only") is True
    assert captured["audit"] is False
    assert captured["keep_mapping"] is False
    assert captured["use_safety_sweep"] is True


def test_selected_non_text_values_raise_safe_error() -> None:
    processor = OpenSearchRedactionProcessor(
        field="message",
        deidentifier=fake_deidentifier,
    )

    with pytest.raises(
        OpenSearchRedactionError, match="selected field must contain text"
    ):
        processor.process({"message": {"raw": "Synthetic Person"}})


def test_nested_selected_sequences_are_rejected() -> None:
    processor = OpenSearchRedactionProcessor(
        field="message",
        deidentifier=fake_deidentifier,
    )

    with pytest.raises(
        OpenSearchRedactionError, match="selected field must contain text"
    ):
        processor.process({"message": [["Synthetic Person"]]})


def test_selected_values_are_bounded(monkeypatch) -> None:
    monkeypatch.setattr(opensearch_adapter, "_MAX_SELECTED_VALUES", 2)
    processor = OpenSearchRedactionProcessor(
        field="message",
        deidentifier=fake_deidentifier,
    )

    with pytest.raises(OpenSearchRedactionError, match="too many values"):
        processor.process({"message": ["one", "two", "three"]})


def test_document_copy_is_bounded_and_rejects_cycles(monkeypatch) -> None:
    monkeypatch.setattr(opensearch_adapter, "_MAX_DOCUMENT_ITEMS", 2)
    processor = OpenSearchRedactionProcessor(
        ignore_missing=True,
        deidentifier=fake_deidentifier,
    )

    with pytest.raises(OpenSearchRedactionError, match="document could not be copied"):
        processor.process({"one": 1, "two": 2, "three": 3})

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(OpenSearchRedactionError, match="document could not be copied"):
        processor.process(cyclic)


def test_fields_and_deidentifier_options_are_bounded() -> None:
    class EndlessFields(Sequence[str]):
        def __getitem__(self, index):
            return f"field_{index}"

        def __len__(self):
            raise AssertionError("length must not be trusted")

    with pytest.raises(OpenSearchRedactionError, match="too many fields"):
        OpenSearchRedactionProcessor(fields=EndlessFields())

    options = {f"option_{index}": index for index in range(65)}
    with pytest.raises(OpenSearchRedactionError, match="too many entries"):
        OpenSearchRedactionProcessor(deidentify_kwargs=options)


def test_hostile_boundary_failures_are_value_free() -> None:
    source = "Synthetic Person has synthetic-555-0100"

    class HostileBoundary(BaseException):
        pass

    class HostileDocument(dict):
        def items(self):
            raise HostileBoundary(source)

    def aborting_deidentifier(text: str, **kwargs):
        del text, kwargs
        raise HostileBoundary(source)

    processor = OpenSearchRedactionProcessor(
        field="message",
        deidentifier=aborting_deidentifier,
    )

    with pytest.raises(OpenSearchRedactionError) as error:
        processor.process({"message": source})

    assert str(error.value) == "redaction failed"
    assert source not in str(error.value)

    with pytest.raises(OpenSearchRedactionError) as copy_error:
        processor.process(HostileDocument(message=source))

    assert str(copy_error.value) == "document could not be copied"
    assert source not in str(copy_error.value)


def test_default_loader_failure_is_value_free(monkeypatch) -> None:
    source = "Synthetic Person has synthetic-555-0100"

    class LoaderFailure(BaseException):
        pass

    def fail_loader():
        raise LoaderFailure(source)

    monkeypatch.setattr(opensearch_adapter, "_default_deidentifier", fail_loader)

    with pytest.raises(OpenSearchRedactionError) as error:
        OpenSearchRedactionProcessor().process({"text": source})

    assert str(error.value) == "redaction failed"
    assert source not in str(error.value)


@pytest.mark.parametrize("fatal_error", [KeyboardInterrupt, SystemExit])
def test_process_preserves_interpreter_control_exceptions(fatal_error) -> None:
    def aborting_deidentifier(text: str, **kwargs):
        del text, kwargs
        raise fatal_error

    processor = OpenSearchRedactionProcessor(
        field="message",
        deidentifier=aborting_deidentifier,
    )

    with pytest.raises(fatal_error):
        processor.process({"message": "Synthetic Person"})


def test_redaction_output_expansion_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(opensearch_adapter, "_MIN_OUTPUT_CHARS", 4)

    def expanding_deidentifier(text: str, **kwargs):
        del text, kwargs
        return "x" * 9

    processor = OpenSearchRedactionProcessor(
        field="message",
        deidentifier=expanding_deidentifier,
    )

    with pytest.raises(OpenSearchRedactionError, match="redaction failed"):
        processor.process({"message": "x"})


def test_deidentifier_options_are_deeply_snapshotted_and_fresh() -> None:
    supplied = {"nested": {"labels": ["original"]}, "payload": b"bounded"}
    received: list[dict[str, object]] = []

    def mutating_deidentifier(text: str, **kwargs):
        received.append(kwargs)
        kwargs["nested"]["labels"].append("callback mutation")
        return text.replace("Synthetic Person", "[PERSON]")

    processor = OpenSearchRedactionProcessor(
        fields=("message", "note"),
        deidentifier=mutating_deidentifier,
        deidentify_kwargs=supplied,
    )
    supplied["nested"]["labels"][0] = "caller mutation"

    document = {"message": "Synthetic Person", "note": "Synthetic Person"}
    assert processor.process(document) == {
        "message": "[PERSON]",
        "note": "[PERSON]",
    }
    assert processor.process(document) == {
        "message": "[PERSON]",
        "note": "[PERSON]",
    }
    expected = {"labels": ["original", "callback mutation"]}
    assert [options["nested"] for options in received] == [expected] * 4
    assert len({id(options["nested"]) for options in received}) == 4


@pytest.mark.parametrize(
    "options",
    [
        {"callback": lambda: None},
        {"number": float("inf")},
        {"number": 1 << 80},
        {"text": "reserved"},
    ],
)
def test_deidentifier_options_reject_unsafe_values(options) -> None:
    with pytest.raises(OpenSearchRedactionError):
        OpenSearchRedactionProcessor(
            deidentifier=fake_deidentifier,
            deidentify_kwargs=options,
        )

    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(OpenSearchRedactionError, match="bounded data options"):
        OpenSearchRedactionProcessor(
            deidentifier=fake_deidentifier,
            deidentify_kwargs={"cyclic": cyclic},
        )


def test_default_reserved_options_are_ignored_before_snapshotting() -> None:
    processor = OpenSearchRedactionProcessor(
        deidentify_kwargs={
            "audit": object(),
            "config": object(),
            "keep_mapping": object(),
            "method": object(),
            "policy": object(),
            "use_safety_sweep": object(),
        }
    )

    assert processor._deidentify_items == ()


def test_document_requires_bounded_json_scalars(monkeypatch) -> None:
    processor = OpenSearchRedactionProcessor(
        ignore_missing=True,
        deidentifier=fake_deidentifier,
    )

    for value in (float("nan"), float("inf"), 1 << 80):
        with pytest.raises(
            OpenSearchRedactionError, match="document could not be copied"
        ):
            processor.process({"metadata": value})

    monkeypatch.setattr(opensearch_adapter, "_MAX_DOCUMENT_TOTAL_BYTES", 4)
    with pytest.raises(OpenSearchRedactionError, match="document could not be copied"):
        processor.process({"note": "12345"})


def test_corrupted_processor_and_report_state_fail_value_free() -> None:
    source = "Synthetic Person has synthetic-555-0100"
    processor = OpenSearchRedactionProcessor(
        field="message",
        deidentifier=fake_deidentifier,
    )
    processor._deidentify_items = source

    with pytest.raises(OpenSearchRedactionError) as processor_error:
        processor.process({"message": source})
    assert str(processor_error.value) == "processor configuration is invalid"
    assert source not in str(processor_error.value)

    report = RedactionReport(
        policy="hipaa_safe_harbor",
        fields=("message",),
        values_seen=1,
        values_redacted=1,
        spans_redacted=1,
    )
    object.__setattr__(report, "values_seen", source)
    with pytest.raises(OpenSearchRedactionError) as report_error:
        report.to_dict()
    assert str(report_error.value) == "report is invalid"
    assert source not in str(report_error.value)
