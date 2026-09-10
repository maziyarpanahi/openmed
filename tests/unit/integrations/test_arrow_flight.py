from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
flight = pytest.importorskip("pyarrow.flight")

from openmed.integrations.arrow_flight import (  # noqa: E402
    ARROW_FLIGHT_DESCRIPTOR_VERSION,
    ArrowFlightDeidentificationServer,
    make_deidentify_descriptor,
)


def test_do_exchange_redacts_each_record_batch_without_buffering_stream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_names = ("Jane Doe", "John Roe", "Sam Poe")
    calls: list[dict[str, Any]] = []

    def fake_process_batch(texts, **kwargs):
        calls.append({"texts": tuple(texts), **kwargs})
        redacted_texts = []
        for text in texts:
            for name in raw_names:
                text = text.replace(name, "[NAME]")
            redacted_texts.append(text)
        items = [
            SimpleNamespace(
                result=SimpleNamespace(deidentified_text=text),
                error=None,
            )
            for text in redacted_texts
        ]
        return SimpleNamespace(items=items)

    server = ArrowFlightDeidentificationServer(
        ("127.0.0.1", 0),
        batch_size=32,
        process_batch_fn=fake_process_batch,
    )
    client = flight.connect(("127.0.0.1", server.port))
    schema = pa.schema(
        [
            pa.field("record_id", pa.int64(), nullable=False),
            pa.field("clinical_note", pa.string()),
            pa.field("status", pa.string()),
        ],
        metadata={b"dataset": b"synthetic"},
    )
    first = pa.RecordBatch.from_pylist(
        [
            {
                "record_id": 1,
                "clinical_note": "Patient Jane Doe called today",
                "status": "open",
            },
            {
                "record_id": 2,
                "clinical_note": "John Roe has a follow-up",
                "status": "closed",
            },
        ],
        schema=schema,
    )
    second = pa.RecordBatch.from_pylist(
        [
            {
                "record_id": 3,
                "clinical_note": "Sam Poe requested records",
                "status": "open",
            },
            {
                "record_id": 4,
                "clinical_note": None,
                "status": "pending",
            },
        ],
        schema=schema,
    )

    caplog.set_level(logging.DEBUG)
    try:
        writer, reader = client.do_exchange(
            make_deidentify_descriptor(
                "clinical_note",
                policy="hipaa_safe_harbor",
            )
        )
        writer.begin(schema)

        writer.write_batch(first)
        first_response = reader.read_chunk().data
        assert first_response.column("clinical_note").to_pylist() == [
            "Patient [NAME] called today",
            "[NAME] has a follow-up",
        ]
        assert len(calls) == 1

        writer.write_batch(second)
        second_response = reader.read_chunk().data
        assert second_response.column("clinical_note").to_pylist() == [
            "[NAME] requested records",
            None,
        ]
        assert len(calls) == 2

        writer.done_writing()
        with pytest.raises(StopIteration):
            reader.read_chunk()
    finally:
        client.close()
        server.shutdown()

    assert first_response.schema == schema
    assert second_response.schema == schema
    assert first_response.num_rows + second_response.num_rows == 4
    assert first_response.column("record_id").to_pylist() == [1, 2]
    assert second_response.column("record_id").to_pylist() == [3, 4]
    assert first_response.column("status").to_pylist() == ["open", "closed"]
    assert second_response.column("status").to_pylist() == ["open", "pending"]
    assert [len(call["texts"]) for call in calls] == [2, 1]
    assert all(call["operation"] == "deidentify" for call in calls)
    assert all(call["policy"] == "hipaa_safe_harbor" for call in calls)
    assert all(call["batch_size"] == 32 for call in calls)

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert all(name not in log_output for name in raw_names)


def test_descriptor_is_versioned_and_carries_policy() -> None:
    descriptor = make_deidentify_descriptor(
        "note",
        policy="hipaa_safe_harbor",
    )

    assert descriptor.descriptor_type == flight.DescriptorType.CMD
    assert descriptor.command == (
        b'{"policy":"hipaa_safe_harbor","text_column":"note","version":1}'
    )
    assert ARROW_FLIGHT_DESCRIPTOR_VERSION == 1


@pytest.mark.parametrize("batch_size", [0, -1])
def test_server_rejects_non_positive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        ArrowFlightDeidentificationServer(batch_size=batch_size)


def test_server_rejects_managed_process_batch_overrides() -> None:
    with pytest.raises(ValueError, match="cannot override managed option.*policy"):
        ArrowFlightDeidentificationServer(
            process_batch_options={"policy": "not-from-descriptor"}
        )
