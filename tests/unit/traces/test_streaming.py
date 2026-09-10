from __future__ import annotations

import io
import json
from typing import Any

import pytest

from openmed.traces.streaming import (
    CancellationToken,
    TraceRecordTooLargeError,
    TraceRedactionError,
    TraceRedactor,
    redact_ndjson_stream,
    redact_trace_lines,
)

SYNTHETIC_NAME = "Synthetic Person"
SYNTHETIC_EMAIL = "trace-user-17@example.test"


def _fake_redactor(text: str) -> str:
    return text.replace(SYNTHETIC_NAME, "[NAME]").replace(SYNTHETIC_EMAIL, "[EMAIL]")


def _records(count: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "sequence": index,
            "message": f"Patient {SYNTHETIC_NAME} event {index}",
            "attributes": {"service": "synthetic-worker"},
        }
        for index in range(count)
    ]


def test_streams_in_order_and_flushes_at_record_bound() -> None:
    consumed: list[int] = []

    def source():
        for record in _records():
            consumed.append(record["sequence"])
            yield record

    progress: list[dict[str, int | bool]] = []
    runner = TraceRedactor(
        record_batch_size=2,
        byte_batch_size=10_000,
        text_redactor=_fake_redactor,
        on_progress=lambda snapshot: progress.append(snapshot.to_dict()),
    )
    iterator = runner.iter_records(source())

    first = next(iterator)
    assert consumed == [0, 1]
    output = [first, *iterator]

    assert [record["sequence"] for record in output] == [0, 1, 2, 3, 4]
    assert all(SYNTHETIC_NAME not in json.dumps(record) for record in output)
    assert all(
        record["attributes"] == {"service": "synthetic-worker"} for record in output
    )
    assert runner.report is not None
    assert runner.report.records_seen == 5
    assert runner.report.records_emitted == 5
    assert runner.report.batches_completed == 3
    assert len(progress) == 3
    assert all(
        set(snapshot)
        <= {
            "records_seen",
            "records_emitted",
            "bytes_seen",
            "bytes_emitted",
            "batches_completed",
            "redacted_fields",
            "cancelled",
        }
        for snapshot in progress
    )


def test_byte_bound_flushes_without_materializing_the_source() -> None:
    runner = TraceRedactor(
        record_batch_size=100,
        byte_batch_size=130,
        text_redactor=_fake_redactor,
    )

    output = list(runner.iter_records(_records(4)))

    assert len(output) == 4
    assert runner.report is not None
    assert runner.report.batches_completed == 4


def test_batch_emits_incrementally_before_processing_the_next_record() -> None:
    redactor_calls: list[str] = []

    def redactor(text: str) -> str:
        redactor_calls.append(text)
        if text == "second":
            raise ValueError("synthetic failure")
        return "[VALUE]"

    runner = TraceRedactor(
        record_batch_size=2,
        byte_batch_size=1_000,
        text_redactor=redactor,
    )
    iterator = runner.iter_records(
        [
            {"message": "first"},
            {"message": "second"},
        ]
    )

    assert next(iterator) == {"message": "[VALUE]"}
    assert redactor_calls == ["first"]
    with pytest.raises(TraceRedactionError):
        next(iterator)


def test_expanded_output_record_cannot_exceed_byte_bound() -> None:
    runner = TraceRedactor(
        record_batch_size=2,
        byte_batch_size=100,
        text_redactor=lambda _text: "X" * 10_000,
    )

    with pytest.raises(TraceRecordTooLargeError) as error:
        list(runner.iter_records([{"message": "small"}]))

    assert str(error.value) == "redacted trace record exceeds byte_batch_size"
    assert runner.report is not None
    assert runner.report.records_seen == 1
    assert runner.report.records_emitted == 0


@pytest.mark.parametrize(
    "record",
    [
        {"sequence": 10**100},
        {"payload": '"' * 60},
    ],
)
def test_byte_bound_accounts_for_numbers_and_json_escaping(
    record: dict[str, Any],
) -> None:
    runner = TraceRedactor(byte_batch_size=100)

    with pytest.raises(TraceRecordTooLargeError):
        list(runner.iter_records([record]))


def test_wildcard_paths_redact_lists_and_tuples() -> None:
    runner = TraceRedactor(
        text_fields=("events.*.message",),
        text_redactor=_fake_redactor,
    )
    record = {
        "events": (
            {"message": SYNTHETIC_NAME},
            {"message": SYNTHETIC_EMAIL},
        )
    }

    output = list(runner.iter_records([record]))

    assert output == [
        {
            "events": (
                {"message": "[NAME]"},
                {"message": "[EMAIL]"},
            )
        }
    ]


def test_dotted_and_nested_paths_are_all_redacted() -> None:
    runner = TraceRedactor(
        text_fields=("attributes.user.email",),
        text_redactor=_fake_redactor,
    )
    record = {
        "attributes.user.email": SYNTHETIC_EMAIL,
        "attributes": {
            "user.email": SYNTHETIC_EMAIL,
            "user": {"email": SYNTHETIC_EMAIL},
        },
    }

    output = list(runner.iter_records([record]))

    assert output == [
        {
            "attributes.user.email": "[EMAIL]",
            "attributes": {
                "user.email": "[EMAIL]",
                "user": {"email": "[EMAIL]"},
            },
        }
    ]
    assert runner.report is not None
    assert runner.report.redacted_fields == 3


def test_default_redactor_is_deterministic_across_batches() -> None:
    records = [
        {
            "sequence": 1,
            "message": f"Contact {SYNTHETIC_EMAIL}; Patient {SYNTHETIC_NAME}",
        },
        {
            "sequence": 2,
            "attributes": {
                "user.email": SYNTHETIC_EMAIL,
                "user.name": SYNTHETIC_NAME,
            },
        },
    ]
    runner = TraceRedactor(
        record_batch_size=1,
        method="replace",
        hmac_secret="offline-fixture-secret",
    )

    output = list(runner.iter_records(records))
    first_token = output[0]["message"].split(";")[0].split()[-1]
    second_token = output[1]["attributes"]["user.email"]
    first_name_token = output[0]["message"].split(";")[1].split()[-1]
    second_name_token = output[1]["attributes"]["user.name"]

    assert first_token == second_token
    assert first_name_token == second_name_token
    assert SYNTHETIC_EMAIL not in json.dumps(output)
    assert SYNTHETIC_NAME not in json.dumps(output)
    assert first_token.startswith("EMAIL_")
    assert runner.report is not None
    assert runner.report.redacted_fields == 3


def test_cancellation_finishes_the_current_batch_and_stops_before_next() -> None:
    consumed: list[int] = []

    def source():
        for record in _records(6):
            consumed.append(record["sequence"])
            yield record

    token = CancellationToken()
    runner = TraceRedactor(
        record_batch_size=2,
        text_redactor=_fake_redactor,
        cancellation=token,
        on_progress=lambda snapshot: snapshot.records_emitted >= 2,
    )

    output = list(runner.iter_records(source()))

    assert [record["sequence"] for record in output] == [0, 1]
    assert consumed == [0, 1]
    assert token.cancelled is False
    assert runner.report is not None
    assert runner.report.cancelled is True
    assert runner.report.records_seen == 2
    assert runner.report.records_emitted == 2

    token.cancel()
    second_runner = TraceRedactor(
        record_batch_size=2,
        text_redactor=_fake_redactor,
        cancellation=token,
    )
    assert list(second_runner.iter_records(_records())) == []
    assert second_runner.report is not None
    assert second_runner.report.cancelled is True


def test_cancellation_requested_mid_batch_emits_the_complete_batch() -> None:
    consumed: list[int] = []
    token = CancellationToken()

    def source():
        for record in _records(4):
            consumed.append(record["sequence"])
            yield record
            if record["sequence"] == 0:
                token.cancel()

    runner = TraceRedactor(
        record_batch_size=3,
        byte_batch_size=10_000,
        text_redactor=_fake_redactor,
        cancellation=token,
    )

    output = list(runner.iter_records(source()))

    assert [record["sequence"] for record in output] == [0, 1, 2]
    assert consumed == [0, 1, 2]
    assert runner.report is not None
    assert runner.report.records_seen == 3
    assert runner.report.records_emitted == 3
    assert runner.report.batches_completed == 1
    assert runner.report.cancelled is True


def test_callback_failures_and_oversized_records_do_not_echo_source_values() -> None:
    def failing_redactor(text: str) -> str:
        raise ValueError(f"synthetic failure for {text}")

    runner = TraceRedactor(
        text_redactor=failing_redactor,
        byte_batch_size=10_000,
    )
    with pytest.raises(TraceRedactionError) as error:
        list(runner.iter_records([{"message": SYNTHETIC_NAME}]))
    assert SYNTHETIC_NAME not in str(error.value)
    assert runner.report is not None
    assert SYNTHETIC_NAME not in repr(runner.report)

    with pytest.raises(TraceRecordTooLargeError) as oversized:
        list(
            TraceRedactor(
                byte_batch_size=8,
                text_redactor=_fake_redactor,
            ).iter_records([{"message": SYNTHETIC_NAME}])
        )
    assert SYNTHETIC_NAME not in str(oversized.value)


def test_deidentified_result_property_is_read_only_once() -> None:
    class StatefulResult:
        def __init__(self) -> None:
            self.reads = 0

        @property
        def deidentified_text(self) -> str:
            self.reads += 1
            return "[NAME]" if self.reads == 1 else SYNTHETIC_NAME

    result = StatefulResult()
    runner = TraceRedactor(text_redactor=lambda _text: result)

    output = list(runner.iter_records([{"message": SYNTHETIC_NAME}]))

    assert output == [{"message": "[NAME]"}]
    assert result.reads == 1
    assert SYNTHETIC_NAME not in json.dumps(output)


def test_ndjson_stream_returns_value_free_report() -> None:
    input_stream = io.StringIO(
        "\n".join(json.dumps(record) for record in _records(2)) + "\n"
    )
    output_stream = io.StringIO()

    report = redact_ndjson_stream(
        input_stream,
        output_stream,
        record_batch_size=2,
        text_redactor=_fake_redactor,
    )
    output = [json.loads(line) for line in output_stream.getvalue().splitlines()]

    assert [record["sequence"] for record in output] == [0, 1]
    assert SYNTHETIC_NAME not in output_stream.getvalue()
    assert report.records_emitted == 2
    assert set(report.to_dict()) == {
        "records_seen",
        "records_emitted",
        "bytes_seen",
        "bytes_emitted",
        "batches_completed",
        "redacted_fields",
        "cancelled",
    }


def test_ndjson_line_iterator_keeps_pseudonyms_across_batches() -> None:
    lines = [
        json.dumps({"message": SYNTHETIC_EMAIL}),
        json.dumps({"attributes": {"user.email": SYNTHETIC_EMAIL}}),
    ]

    output = [
        json.loads(line)
        for line in redact_trace_lines(
            lines,
            record_batch_size=1,
            method="replace",
            hmac_secret="line-fixture-secret",
        )
    ]

    assert output[0]["message"] == output[1]["attributes"]["user.email"]
    assert SYNTHETIC_EMAIL not in json.dumps(output)
