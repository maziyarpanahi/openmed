"""Synthetic unit tests for append-only agent run event sequences."""

from __future__ import annotations

import json

import pytest

from openmed.agent.event_sequence import (
    EVENT_SEQUENCE_SCHEMA_VERSION,
    MAX_EVENT_SEQUENCE_LENGTH,
    MAX_SEQUENCE_FINDINGS,
    MAX_SEQUENCE_NUMBER,
    SEQUENCE_REASON_CODES,
    EventReference,
    EventSequenceError,
    EventSequenceReport,
    SequenceFinding,
    validate_event_sequence,
)

RUN = "run_0123456789abcdef"
OTHER_RUN = "run_fedcba9876543210"

SENTINEL_PAYLOAD = "PATIENT JANE DOE MRN 8675309 Bearer abc.def /var/phi/note.txt"


def ref(number: int, *, run_id: str = RUN, event_id: str | None = None):
    return EventReference(
        run_id=run_id,
        event_id=event_id if event_id is not None else f"ev-{number:06d}",
        sequence_number=number,
    )


def codes(report: EventSequenceReport) -> list[str]:
    return list(report.reason_codes)


def test_contiguous_sequence_is_valid() -> None:
    report = validate_event_sequence(RUN, [ref(0), ref(1), ref(2)])
    assert report.is_valid
    assert report.findings == ()
    assert (report.event_count, report.first_sequence_number) == (3, 0)
    assert report.last_sequence_number == 2
    assert report.terminal_sequence_number is None
    assert report.schema_version == EVENT_SEQUENCE_SCHEMA_VERSION


def test_single_event_sequence_is_valid() -> None:
    report = validate_event_sequence(RUN, [ref(0)], terminal_sequence_number=0)
    assert report.is_valid
    assert (report.first_sequence_number, report.last_sequence_number) == (0, 0)


def test_empty_sequence_only_valid_when_allowed() -> None:
    allowed = validate_event_sequence(RUN, [], allow_empty=True)
    assert allowed.is_valid
    assert allowed.event_count == 0
    assert allowed.first_sequence_number is None
    assert allowed.last_sequence_number is None

    rejected = validate_event_sequence(RUN, [])
    assert codes(rejected) == ["empty_sequence"]
    assert rejected.findings[0].sequence_number is None
    assert rejected.findings[0].event_id is None


def test_generators_are_consumed_in_append_order() -> None:
    report = validate_event_sequence(RUN, (ref(index) for index in range(4)))
    assert report.is_valid
    assert report.event_count == 4


@pytest.mark.parametrize("start", [0, 1, 7])
def test_expected_start_is_configurable(start: int) -> None:
    report = validate_event_sequence(
        RUN, [ref(start), ref(start + 1)], expected_start=start
    )
    assert report.is_valid


def test_start_sequence_mismatch_is_reported() -> None:
    report = validate_event_sequence(RUN, [ref(3), ref(4)])
    assert codes(report) == ["start_sequence_mismatch"]
    assert report.findings[0].sequence_number == 3


def test_gaps_report_the_first_missing_number_per_gap() -> None:
    report = validate_event_sequence(RUN, [ref(0), ref(2), ref(5)])
    assert codes(report) == ["sequence_gap", "sequence_gap"]
    assert [finding.sequence_number for finding in report.findings] == [1, 3]
    assert all(finding.event_id is None for finding in report.findings)


def test_duplicate_sequence_number_is_reported_once_per_repeat() -> None:
    report = validate_event_sequence(
        RUN, [ref(0), ref(1, event_id="ev-a"), ref(1, event_id="ev-b")]
    )
    assert codes(report) == ["duplicate_sequence_number", "out_of_order"]
    assert {finding.event_id for finding in report.findings} == {"ev-b"}


def test_duplicate_event_id_is_reported_independently() -> None:
    report = validate_event_sequence(
        RUN, [ref(0, event_id="ev-a"), ref(1, event_id="ev-a")]
    )
    assert codes(report) == ["duplicate_event_id"]
    assert report.findings[0].sequence_number == 1


def test_reordered_input_is_reported() -> None:
    report = validate_event_sequence(RUN, [ref(0), ref(2), ref(1)])
    assert codes(report) == ["out_of_order"]
    assert report.findings[0].sequence_number == 1
    assert report.last_sequence_number == 1


def test_cross_run_reference_is_reported() -> None:
    report = validate_event_sequence(
        RUN, [ref(0), ref(1, run_id=OTHER_RUN, event_id="ev-x")]
    )
    assert codes(report) == ["cross_run_reference"]
    assert report.findings[0].event_id == "ev-x"


def test_post_terminal_events_and_missing_terminal_event() -> None:
    report = validate_event_sequence(
        RUN, [ref(0), ref(1), ref(2)], terminal_sequence_number=1
    )
    assert codes(report) == ["post_terminal_event"]
    assert report.findings[0].sequence_number == 2

    missing = validate_event_sequence(RUN, [ref(0), ref(1)], terminal_sequence_number=5)
    assert codes(missing) == ["terminal_event_missing"]
    assert missing.findings[0].sequence_number == 5


def test_findings_are_ordered_deterministically() -> None:
    report = validate_event_sequence(
        RUN,
        [
            ref(4, run_id=OTHER_RUN, event_id="ev-z"),
            ref(4, event_id="ev-y"),
            ref(1, event_id="ev-w"),
        ],
        terminal_sequence_number=3,
    )
    assert codes(report) == [
        "out_of_order",
        "start_sequence_mismatch",
        "sequence_gap",
        "terminal_event_missing",
        "cross_run_reference",
        "duplicate_sequence_number",
        "out_of_order",
        "post_terminal_event",
        "post_terminal_event",
    ]
    assert [finding.sequence_number for finding in report.findings] == [
        1,
        1,
        2,
        3,
        4,
        4,
        4,
        4,
        4,
    ]
    assert (
        report.to_json()
        == validate_event_sequence(
            RUN,
            [
                ref(4, run_id=OTHER_RUN, event_id="ev-z"),
                ref(4, event_id="ev-y"),
                ref(1, event_id="ev-w"),
            ],
            terminal_sequence_number=3,
        ).to_json()
    )


def test_findings_are_truncated_with_a_stable_marker() -> None:
    references = [ref(index * 2) for index in range(MAX_SEQUENCE_FINDINGS + 10)]
    report = validate_event_sequence(RUN, references)
    assert len(report.findings) == MAX_SEQUENCE_FINDINGS
    assert report.findings[-1].reason_code == "findings_truncated"
    assert report.findings[-1].sequence_number is None


def test_report_serialization_is_byte_stable_and_field_ordered() -> None:
    report = validate_event_sequence(RUN, [ref(0), ref(2)])
    assert list(report.to_dict()) == [
        "schema_version",
        "run_id",
        "event_count",
        "first_sequence_number",
        "last_sequence_number",
        "terminal_sequence_number",
        "findings",
    ]
    assert list(report.to_dict()["findings"][0]) == [
        "reason_code",
        "sequence_number",
        "event_id",
    ]
    assert json.loads(report.to_json()) == report.to_dict()
    assert report.to_json() == report.to_json()


def test_reference_serialization_is_metadata_only() -> None:
    assert ref(3).to_dict() == {
        "run_id": RUN,
        "event_id": "ev-000003",
        "sequence_number": 3,
    }


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, 2**70])
def test_boolean_and_non_integer_sequence_numbers_fail_closed(value) -> None:
    with pytest.raises(EventSequenceError) as excinfo:
        EventReference(run_id=RUN, event_id="ev-a", sequence_number=value)
    assert excinfo.value.code in {
        "invalid_sequence_number",
        "sequence_number_out_of_range",
    }
    assert excinfo.value.field_name == "sequence_number"


@pytest.mark.parametrize("value", [-1, MAX_SEQUENCE_NUMBER + 1])
def test_out_of_range_sequence_numbers_fail_closed(value: int) -> None:
    with pytest.raises(EventSequenceError, match="^sequence_number: "):
        EventReference(run_id=RUN, event_id="ev-a", sequence_number=value)


def test_max_sequence_number_is_accepted() -> None:
    reference = EventReference(RUN, "ev-a", MAX_SEQUENCE_NUMBER)
    assert reference.sequence_number == MAX_SEQUENCE_NUMBER


@pytest.mark.parametrize("field", ["run_id", "event_id"])
@pytest.mark.parametrize("value", ["", "-lead", "trail-", "a" * 129, 7, None, b"x"])
def test_invalid_identifiers_fail_closed(field: str, value) -> None:
    kwargs = {"run_id": RUN, "event_id": "ev-a", "sequence_number": 0}
    kwargs[field] = value
    with pytest.raises(EventSequenceError) as excinfo:
        EventReference(**kwargs)
    assert (excinfo.value.code, excinfo.value.field_name) == (
        "invalid_identifier",
        field,
    )


def test_unknown_reason_codes_fail_closed() -> None:
    with pytest.raises(EventSequenceError, match="^reason_code: unknown_reason_code$"):
        SequenceFinding("not_a_reason")
    assert "findings_truncated" in SEQUENCE_REASON_CODES


@pytest.mark.parametrize(
    "references", [ref(0), "abc", b"abc", 5, None, [ref(0), "abc"], [None]]
)
def test_invalid_reference_containers_fail_closed(references) -> None:
    with pytest.raises(EventSequenceError) as excinfo:
        validate_event_sequence(RUN, references)
    assert excinfo.value.code in {"invalid_sequence", "invalid_reference_type"}
    assert excinfo.value.field_name == "references"


def test_oversized_sequences_fail_closed() -> None:
    class _Endless:
        def __iter__(self):
            number = 0
            while True:
                yield ref(number)
                number += 1

    with pytest.raises(EventSequenceError, match="^references: too_many_events$"):
        validate_event_sequence(RUN, _Endless())


def test_length_limit_is_a_positive_bound() -> None:
    assert MAX_EVENT_SEQUENCE_LENGTH > 0


@pytest.mark.parametrize("value", [True, -1, 1.5, "0", 2**70])
def test_invalid_expected_start_fails_closed(value) -> None:
    with pytest.raises(EventSequenceError, match="^expected_start: "):
        validate_event_sequence(RUN, [ref(0)], expected_start=value)


@pytest.mark.parametrize("value", [True, -1, 1.5, "0"])
def test_invalid_terminal_sequence_fails_closed(value) -> None:
    with pytest.raises(EventSequenceError, match="^terminal_sequence_number: "):
        validate_event_sequence(RUN, [ref(0)], terminal_sequence_number=value)


def test_terminal_before_expected_start_fails_closed() -> None:
    with pytest.raises(
        EventSequenceError, match="^terminal_sequence_number: terminal_before_start$"
    ):
        validate_event_sequence(
            RUN, [ref(5)], expected_start=5, terminal_sequence_number=4
        )


def test_invalid_allow_empty_flag_fails_closed() -> None:
    with pytest.raises(EventSequenceError, match="^allow_empty: invalid_flag$"):
        validate_event_sequence(RUN, [ref(0)], allow_empty=1)


def test_invalid_run_id_fails_closed() -> None:
    with pytest.raises(EventSequenceError, match="^run_id: invalid_identifier$"):
        validate_event_sequence("bad id", [ref(0)])


def test_invalid_schema_version_fails_closed() -> None:
    with pytest.raises(EventSequenceError, match="^schema_version: "):
        EventSequenceReport(
            run_id=RUN,
            event_count=0,
            first_sequence_number=None,
            last_sequence_number=None,
            terminal_sequence_number=None,
            findings=(),
            schema_version="openmed.agent.event_sequence.v0",
        )


def test_no_payload_text_can_reach_findings_or_errors() -> None:
    report = validate_event_sequence(
        RUN, [ref(1, run_id=OTHER_RUN)], terminal_sequence_number=9
    )
    assert SENTINEL_PAYLOAD not in report.to_json()

    with pytest.raises(EventSequenceError) as excinfo:
        EventReference(run_id=RUN, event_id=SENTINEL_PAYLOAD, sequence_number=0)
    assert SENTINEL_PAYLOAD not in str(excinfo.value)
    assert SENTINEL_PAYLOAD not in repr(excinfo.value)


def test_event_sequence_contract_is_available_from_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.EventReference is EventReference
    assert agent.EventSequenceError is EventSequenceError
    assert agent.EventSequenceReport is EventSequenceReport
    assert agent.SequenceFinding is SequenceFinding
    assert agent.validate_event_sequence is validate_event_sequence
    exported = {
        "EVENT_SEQUENCE_SCHEMA_VERSION",
        "EventReference",
        "EventSequenceError",
        "EventSequenceReport",
        "MAX_EVENT_SEQUENCE_LENGTH",
        "MAX_SEQUENCE_FINDINGS",
        "MAX_SEQUENCE_NUMBER",
        "SEQUENCE_REASON_CODES",
        "SequenceFinding",
        "validate_event_sequence",
    }
    assert exported.issubset(set(agent.__all__))
