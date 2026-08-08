from __future__ import annotations

from datetime import datetime, timezone

import pytest

pa = pytest.importorskip("pyarrow")

from openmed.traces.schemas.columnar import (
    ColumnarTraceAdapterError,
    ColumnarTraceSchemaAdapter,
    iter_redacted_record_batches,
    redact_record_batch,
)


def _trace_schema():
    return pa.schema(
        [
            pa.field("trace_id", pa.string(), nullable=False),
            pa.field(
                "payload",
                pa.struct(
                    [
                        pa.field("message", pa.large_string()),
                        pa.field(
                            "events",
                            pa.list_(
                                pa.struct(
                                    [
                                        pa.field("text", pa.string()),
                                        pa.field("kind", pa.string(), nullable=False),
                                    ]
                                )
                            ),
                        ),
                    ]
                ),
            ),
            pa.field("label", pa.int8(), nullable=False),
            pa.field("observed_at", pa.timestamp("us", tz="UTC")),
        ],
        metadata={b"fixture": b"synthetic"},
    )


def _trace_batch():
    table = pa.Table.from_pylist(
        [
            {
                "trace_id": "trace-a",
                "payload": {
                    "message": "SYNTHETIC_NOTE_A",
                    "events": [
                        {"text": "SYNTHETIC_EVENT_A", "kind": "tool"},
                        None,
                    ],
                },
                "label": 1,
                "observed_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            },
            {
                "trace_id": "trace-b",
                "payload": None,
                "label": 0,
                "observed_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
            },
        ],
        schema=_trace_schema(),
    )
    return table.to_batches(max_chunksize=2)[0]


def test_redact_record_batch_preserves_nested_schema_and_labels() -> None:
    batch = _trace_batch()
    seen: list[str] = []

    def redact(value: str) -> str:
        seen.append(value)
        return value.replace("SYNTHETIC_", "MASKED_")

    adapted = redact_record_batch(
        batch,
        text_columns=["payload.message", ("payload", "events", "text")],
        text_redactor=redact,
    )

    assert adapted.schema == batch.schema
    assert adapted.num_rows == batch.num_rows
    assert adapted.column("trace_id").equals(batch.column("trace_id"))
    assert adapted.column("label").equals(batch.column("label"))
    assert adapted.column("observed_at").equals(batch.column("observed_at"))
    assert adapted.column("payload").type == batch.column("payload").type
    assert adapted.to_pylist() == [
        {
            "trace_id": "trace-a",
            "payload": {
                "message": "MASKED_NOTE_A",
                "events": [
                    {"text": "MASKED_EVENT_A", "kind": "tool"},
                    None,
                ],
            },
            "label": 1,
            "observed_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        },
        {
            "trace_id": "trace-b",
            "payload": None,
            "label": 0,
            "observed_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        },
    ]
    assert seen == ["SYNTHETIC_NOTE_A", "SYNTHETIC_EVENT_A"]


def test_default_redactor_is_deterministic_and_only_touches_selected_text() -> None:
    batch = pa.record_batch(
        {
            "text": ["SYNTHETIC_NOTE_A", None, ""],
            "label": [1, 0, 1],
        }
    )

    first = redact_record_batch(batch, text_columns=["text"])
    second = redact_record_batch(batch, text_columns=["text"])

    assert first.equals(second)
    assert first.column("text").to_pylist() == ["[REDACTED]", None, ""]
    assert first.column("label").equals(batch.column("label"))


def test_iter_redacted_record_batches_is_lazy_and_bounded() -> None:
    batch = pa.record_batch(
        {
            "text": ["SYNTHETIC_A", "SYNTHETIC_B", "SYNTHETIC_C", "SYNTHETIC_D"],
            "label": [0, 1, 0, 1],
        }
    )
    consumed: list[str] = []

    def source():
        consumed.append("first")
        yield batch
        consumed.append("second")
        yield batch.slice(0, 1)

    iterator = iter_redacted_record_batches(
        source(),
        text_columns=["text"],
        text_redactor=lambda value: value.replace("SYNTHETIC_", "MASKED_"),
        batch_size=2,
    )

    assert consumed == []
    first = next(iterator)
    assert consumed == ["first"]
    assert first.num_rows == 2
    remaining = list(iterator)
    assert [part.num_rows for part in remaining] == [2, 1]
    assert all(part.num_rows <= 2 for part in [first, *remaining])
    assert consumed == ["first", "second"]


def test_redactor_errors_do_not_include_source_values() -> None:
    batch = pa.record_batch({"text": ["SYNTHETIC_SECRET_VALUE"]})

    def fail(_: str) -> str:
        raise RuntimeError("SYNTHETIC_SECRET_VALUE must not escape")

    with pytest.raises(ColumnarTraceAdapterError) as caught:
        redact_record_batch(batch, text_columns=["text"], text_redactor=fail)

    assert "SYNTHETIC_SECRET_VALUE" not in str(caught.value)


def test_schema_adapter_reuses_configuration() -> None:
    adapter = ColumnarTraceSchemaAdapter(
        ["payload.message"],
        text_redactor=lambda value: "MASKED_" + value,
        batch_size=1,
    )

    batches = list(adapter.adapt_batches([_trace_batch()]))

    assert [batch.num_rows for batch in batches] == [1, 1]
    assert [
        (
            payload["message"]
            if (payload := batch.column("payload").to_pylist()[0]) is not None
            else None
        )
        for batch in batches
    ] == ["MASKED_SYNTHETIC_NOTE_A", None]


def test_invalid_nested_path_is_reported_without_values() -> None:
    with pytest.raises(ColumnarTraceAdapterError, match="missing"):
        redact_record_batch(
            _trace_batch(),
            text_columns=["payload.missing"],
        )
