"""Offline tests for the scriptable dataflow-tool processor."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.integrations import dataflow_tool_processor as processor


def _redact(text: str) -> tuple[str, list[object]]:
    replacements = {
        "Jane Roe": "[NAME]",
        "John Doe": "[NAME]",
        "555-0100": "[PHONE]",
        "jane.roe@example.org": "[EMAIL]",
    }
    entities: list[object] = []
    for raw_value, replacement in replacements.items():
        if raw_value in text:
            text = text.replace(raw_value, replacement)
            entities.append(object())
    return text, entities


def _batch_result(texts: list[str]) -> SimpleNamespace:
    items = []
    for text in texts:
        redacted, entities = _redact(text)
        items.append(
            SimpleNamespace(
                success=True,
                result=SimpleNamespace(
                    deidentified_text=redacted,
                    pii_entities=entities,
                ),
            )
        )
    return SimpleNamespace(items=items)


@pytest.fixture(autouse=True)
def _empty_pipeline_cache() -> None:
    processor.clear_pipeline_cache()
    yield
    processor.clear_pipeline_cache()


def test_callable_redacts_configured_fields_and_reuses_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = object()
    pipeline_loads: list[Any] = []
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_create_pipeline(config: Any | None) -> object:
        pipeline_loads.append(config)
        return pipeline

    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((list(texts), dict(kwargs)))
        return _batch_result(list(texts))

    monkeypatch.setattr(processor, "_create_pipeline", fake_create_pipeline)
    original_records = [
        {
            "record_id": "synthetic-1",
            "note": "Patient Jane Roe called 555-0100",
            "patient": {"contact": "jane.roe@example.org"},
            "facility": "example-clinic",
        },
        {
            "record_id": "synthetic-2",
            "note": "John Doe requested follow-up",
            "patient": {"contact": None},
            "facility": "example-clinic",
        },
    ]
    incoming_attributes = [
        {"route": "clinical", "mime.type": "application/json"},
        {"route": "follow-up", "mime.type": "application/json"},
    ]

    outputs = [
        processor.process_flow_file(
            record,
            attributes,
            fields=("note", "patient.contact", "missing"),
            process_batch_fn=fake_process_batch,
            policy="hipaa_safe_harbor",
            use_safety_sweep=False,
        )
        for record, attributes in zip(original_records, incoming_attributes)
    ]

    assert len(outputs) == len(original_records)
    first_record, first_attributes = outputs[0]
    second_record, second_attributes = outputs[1]
    assert first_record == {
        "record_id": "synthetic-1",
        "note": "Patient [NAME] called [PHONE]",
        "patient": {"contact": "[EMAIL]"},
        "facility": "example-clinic",
    }
    assert second_record["note"] == "[NAME] requested follow-up"
    assert second_record["patient"]["contact"] is None
    assert original_records[0]["note"] == "Patient Jane Roe called 555-0100"
    assert original_records[0]["patient"]["contact"] == "jane.roe@example.org"

    assert first_attributes == {
        **incoming_attributes[0],
        processor.RECORD_COUNT_ATTRIBUTE: "1",
        processor.FIELD_COUNT_ATTRIBUTE: "2",
        processor.ENTITY_COUNT_ATTRIBUTE: "3",
    }
    assert second_attributes == {
        **incoming_attributes[1],
        processor.RECORD_COUNT_ATTRIBUTE: "1",
        processor.FIELD_COUNT_ATTRIBUTE: "1",
        processor.ENTITY_COUNT_ATTRIBUTE: "1",
    }
    assert pipeline_loads == [None]
    assert [texts for texts, _ in calls] == [
        [
            "Patient Jane Roe called 555-0100",
            "jane.roe@example.org",
        ],
        ["John Doe requested follow-up"],
    ]
    assert all(kwargs["loader"] is pipeline for _, kwargs in calls)
    assert all(kwargs["operation"] == "deidentify" for _, kwargs in calls)
    assert all(kwargs["continue_on_error"] is False for _, kwargs in calls)
    assert all(kwargs["policy"] == "hipaa_safe_harbor" for _, kwargs in calls)
    assert all(kwargs["use_safety_sweep"] is False for _, kwargs in calls)
    assert [kwargs["batch_size"] for _, kwargs in calls] == [2, 1]

    for attributes in (first_attributes, second_attributes):
        emitted_counts = {
            key: value
            for key, value in attributes.items()
            if key.startswith("openmed.redaction.")
        }
        assert emitted_counts
        assert all(value.isdecimal() for value in emitted_counts.values())
        serialized = json.dumps(emitted_counts)
        assert "Jane Roe" not in serialized
        assert "John Doe" not in serialized
        assert "555-0100" not in serialized
        assert "jane.roe@example.org" not in serialized


def test_json_lines_entrypoint_preserves_record_count_and_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_loads = 0

    def fake_create_pipeline(config: Any | None) -> object:
        nonlocal pipeline_loads
        pipeline_loads += 1
        return object()

    monkeypatch.setattr(processor, "_create_pipeline", fake_create_pipeline)
    envelopes = [
        {
            "record": {"id": "one", "note": "Jane Roe"},
            "attributes": {"route": "a"},
        },
        {
            "record": {"id": "two", "note": "Call 555-0100"},
            "attributes": {"route": "b"},
        },
        {
            "record": {"id": "three", "note": "No identifiers"},
            "attributes": {"route": "c"},
        },
    ]
    input_stream = io.StringIO(
        "\n".join(json.dumps(envelope) for envelope in envelopes) + "\n"
    )
    output_stream = io.StringIO()

    emitted = processor.process_json_lines(
        input_stream,
        output_stream,
        fields="note",
        process_batch_fn=lambda texts, **kwargs: _batch_result(list(texts)),
    )

    output_envelopes = [
        json.loads(line) for line in output_stream.getvalue().splitlines()
    ]
    assert emitted == len(envelopes)
    assert len(output_envelopes) == len(envelopes)
    assert [item["record"]["id"] for item in output_envelopes] == [
        "one",
        "two",
        "three",
    ]
    assert [item["attributes"]["route"] for item in output_envelopes] == [
        "a",
        "b",
        "c",
    ]
    assert output_envelopes[0]["record"]["note"] == "[NAME]"
    assert output_envelopes[1]["record"]["note"] == "Call [PHONE]"
    assert pipeline_loads == 1


def test_missing_and_non_string_fields_pass_through_without_loading_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_create_pipeline(config: Any | None) -> object:
        raise AssertionError("pipeline should not load without redaction targets")

    monkeypatch.setattr(processor, "_create_pipeline", fail_create_pipeline)
    record = {"id": "synthetic", "note": None, "sequence": 4}

    output, attributes = processor.script_processor(
        record,
        {"route": "passthrough"},
        fields=("note", "sequence", "missing"),
    )

    assert output == record
    assert attributes == {
        "route": "passthrough",
        processor.RECORD_COUNT_ATTRIBUTE: "1",
        processor.FIELD_COUNT_ATTRIBUTE: "0",
        processor.ENTITY_COUNT_ATTRIBUTE: "0",
    }


def test_failures_do_not_expose_raw_field_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(processor, "_create_pipeline", lambda config: object())
    raw_value = "Patient Jane Roe called 555-0100"

    def fail_batch(texts: list[str], **kwargs: Any) -> None:
        raise RuntimeError(f"model failed on {texts[0]}")

    with pytest.raises(processor.DataflowToolProcessorError) as exc_info:
        processor.process_json_lines(
            [
                json.dumps(
                    {
                        "record": {"note": raw_value},
                        "attributes": {"route": "clinical"},
                    }
                )
            ],
            io.StringIO(),
            fields="note",
            process_batch_fn=fail_batch,
        )

    message = str(exc_info.value)
    assert "input line 1" in message
    assert "Jane Roe" not in message
    assert "555-0100" not in message


def test_cardinality_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(processor, "_create_pipeline", lambda config: object())

    with pytest.raises(
        processor.DataflowToolProcessorError,
        match="result count",
    ):
        processor.process_flow_file(
            {"note": "Jane Roe"},
            fields="note",
            process_batch_fn=lambda texts, **kwargs: SimpleNamespace(items=[]),
        )
