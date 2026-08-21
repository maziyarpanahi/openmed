"""Offline tests for the framework-neutral stream processor helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openmed.integrations.stream_processor import (
    StreamDeidentifyMapFunction,
    StreamSink,
    run_stream_job,
)


def _redact(text: str) -> str:
    return text.replace("Jane Roe", "[NAME]").replace("555-0101", "[PHONE]")


class BatchProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> Any:
        self.calls.append((list(texts), kwargs))
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    success=True,
                    result=SimpleNamespace(deidentified_text=_redact(text)),
                )
                for text in texts
            ]
        )


def test_map_batch_redacts_target_field_and_preserves_other_fields() -> None:
    probe = BatchProbe()
    mapper = StreamDeidentifyMapFunction(
        "note",
        policy="hipaa_safe_harbor",
        batch_size=2,
        process_batch_fn=probe,
        use_safety_sweep=False,
    )
    records = [
        {
            "event_id": "evt-1",
            "note": "Patient Jane Roe called 555-0101",
            "encounter": {"id": "enc-7"},
        },
        {"event_id": "evt-2", "note": "No identifiers", "priority": 2},
        {"event_id": "evt-3", "note": None, "priority": 1},
    ]

    output = mapper.map_batch(records)

    assert output == [
        {
            "event_id": "evt-1",
            "note": "Patient [NAME] called [PHONE]",
            "encounter": {"id": "enc-7"},
        },
        {"event_id": "evt-2", "note": "No identifiers", "priority": 2},
        {"event_id": "evt-3", "note": None, "priority": 1},
    ]
    assert records[0]["note"] == "Patient Jane Roe called 555-0101"
    assert [texts for texts, _ in probe.calls] == [
        ["Patient Jane Roe called 555-0101", "No identifiers"]
    ]
    kwargs = probe.calls[0][1]
    assert kwargs == {
        "operation": "deidentify",
        "model_name": "disease_detection_superclinical",
        "batch_size": 2,
        "continue_on_error": False,
        "method": "mask",
        "policy": "hipaa_safe_harbor",
        "use_safety_sweep": False,
    }


def test_out_of_order_duplicates_are_stateless_idempotent_and_keep_order() -> None:
    probe = BatchProbe()
    mapper = StreamDeidentifyMapFunction("note", batch_size=2, process_batch_fn=probe)
    records = [
        {"sequence": 9, "note": "Patient Jane Roe called"},
        {"sequence": 2, "note": "Patient Jane Roe called"},
        {"sequence": 7, "note": "Call 555-0101"},
    ]

    first = mapper.map_batch(records)
    second = mapper.map_batch(first)

    assert [record["sequence"] for record in first] == [9, 2, 7]
    assert first[0]["note"] == first[1]["note"] == "Patient [NAME] called"
    assert first[2]["note"] == "Call [PHONE]"
    assert second == first
    assert [len(texts) for texts, _ in probe.calls] == [2, 1, 2, 1]


def test_offline_job_opens_once_per_subtask_and_writes_micro_batches() -> None:
    probe = BatchProbe()

    class OpenProbe(StreamDeidentifyMapFunction):
        def __init__(self) -> None:
            super().__init__("note", batch_size=2, process_batch_fn=probe)
            self.open_calls = 0

        def open(self, runtime_context: Any | None = None) -> None:
            self.open_calls += 1
            super().open(runtime_context)

    mapper = OpenProbe()
    written: list[dict[str, Any]] = []
    source = [
        {"sequence": 3, "note": "Patient Jane Roe"},
        {"sequence": 1, "note": "Call 555-0101"},
        {"sequence": 2, "note": "No identifiers"},
    ]

    count = run_stream_job(source, mapper, StreamSink(written.append))

    assert count == 3
    assert mapper.open_calls == 1
    assert [record["sequence"] for record in written] == [3, 1, 2]
    assert [len(texts) for texts, _ in probe.calls] == [2, 1]


@pytest.mark.parametrize(
    ("record", "error", "message"),
    [
        ({"other": "value"}, KeyError, "missing text field"),
        ({"note": 42}, TypeError, "must be a string or None"),
    ],
)
def test_map_rejects_invalid_records(
    record: dict[str, Any], error: type[Exception], message: str
) -> None:
    mapper = StreamDeidentifyMapFunction("note", process_batch_fn=BatchProbe())

    with pytest.raises(error, match=message):
        mapper.map(record)


def test_batch_result_count_must_match_input_count() -> None:
    mapper = StreamDeidentifyMapFunction(
        "note",
        process_batch_fn=lambda texts, **kwargs: SimpleNamespace(items=[]),
    )

    with pytest.raises(ValueError, match="returned 0 results for 1 inputs"):
        mapper.map({"note": "Patient Jane Roe"})
