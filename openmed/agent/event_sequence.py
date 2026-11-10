"""Deterministic validation of append-only agent run event sequences.

Run evidence can only be replayed when every event carries a contiguous,
run-scoped sequence number. This module checks that contract from metadata
alone: opaque run and event identifiers plus integers. Event payloads,
prompts, tool arguments, and clinical text are never accepted, retained, or
echoed in findings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

EVENT_SEQUENCE_SCHEMA_VERSION = "openmed.agent.event_sequence.v1"
MAX_EVENT_SEQUENCE_LENGTH = 100_000
MAX_SEQUENCE_NUMBER = 1_000_000_000
MAX_SEQUENCE_FINDINGS = 1_000

SEQUENCE_REASON_CODES = frozenset(
    {
        "cross_run_reference",
        "duplicate_event_id",
        "duplicate_sequence_number",
        "empty_sequence",
        "findings_truncated",
        "out_of_order",
        "post_terminal_event",
        "sequence_gap",
        "start_sequence_mismatch",
        "terminal_event_missing",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_FINDING_FIELDS = ("reason_code", "sequence_number", "event_id")
_REPORT_FIELDS = (
    "schema_version",
    "run_id",
    "event_count",
    "first_sequence_number",
    "last_sequence_number",
    "terminal_sequence_number",
    "findings",
)


class EventSequenceError(ValueError):
    """Raised when event-sequence input fails closed structural validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional fixed public field associated with the failure.

    Messages and attributes carry controlled diagnostic metadata only; the
    rejected value is never retained or echoed.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class EventReference:
    """Opaque, run-scoped reference to one appended agent run event.

    Attributes:
        run_id: Bounded opaque identifier of the owning run.
        event_id: Bounded opaque identifier of the event itself.
        sequence_number: Non-negative append position within the run.
    """

    run_id: str
    event_id: str
    sequence_number: int

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        _validate_identifier(self.event_id, "event_id")
        _validate_sequence_number(self.sequence_number, "sequence_number")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary of metadata-only fields."""

        return {
            "run_id": self.run_id,
            "event_id": self.event_id,
            "sequence_number": self.sequence_number,
        }


@dataclass(frozen=True, slots=True)
class SequenceFinding:
    """One stable, value-free reason why a sequence is not replayable.

    Attributes:
        reason_code: Member of :data:`SEQUENCE_REASON_CODES`.
        sequence_number: Sequence number the finding refers to, when one
            applies.
        event_id: Opaque event identifier, when the finding names an event.
    """

    reason_code: str
    sequence_number: int | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.reason_code) is not str
            or self.reason_code not in SEQUENCE_REASON_CODES
        ):
            raise EventSequenceError("unknown_reason_code", "reason_code")
        if self.sequence_number is not None:
            _validate_sequence_number(self.sequence_number, "sequence_number")
        if self.event_id is not None:
            _validate_identifier(self.event_id, "event_id")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "reason_code": self.reason_code,
            "sequence_number": self.sequence_number,
            "event_id": self.event_id,
        }
        return {field: values[field] for field in _FINDING_FIELDS}


@dataclass(frozen=True, slots=True)
class EventSequenceReport:
    """Deterministic, metadata-only verdict for one run's event sequence."""

    run_id: str
    event_count: int
    first_sequence_number: int | None
    last_sequence_number: int | None
    terminal_sequence_number: int | None
    findings: tuple[SequenceFinding, ...]
    schema_version: str = EVENT_SEQUENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != EVENT_SEQUENCE_SCHEMA_VERSION
        ):
            raise EventSequenceError("invalid_schema_version", "schema_version")

    @property
    def is_valid(self) -> bool:
        """Return whether the sequence produced no findings."""

        return not self.findings

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return the finding reason codes in report order."""

        return tuple(finding.reason_code for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "event_count": self.event_count,
            "first_sequence_number": self.first_sequence_number,
            "last_sequence_number": self.last_sequence_number,
            "terminal_sequence_number": self.terminal_sequence_number,
            "findings": [finding.to_dict() for finding in self.findings],
        }
        return {field: values[field] for field in _REPORT_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def validate_event_sequence(
    run_id: str,
    references: Iterable[EventReference],
    *,
    expected_start: int = 0,
    terminal_sequence_number: int | None = None,
    allow_empty: bool = False,
) -> EventSequenceReport:
    """Validate a run-scoped sequence of opaque event references.

    The references are consumed in the order supplied, which is the order a
    reader would replay them in. Structural problems (wrong types, negative or
    out-of-range numbers, oversized input) fail closed with
    :class:`EventSequenceError`. Sequence problems are reported as findings so
    a caller sees every defect in one pass.

    Args:
        run_id: Opaque identifier every reference must belong to.
        references: Iterable of :class:`EventReference` values in append order.
        expected_start: Sequence number the first event must carry.
        terminal_sequence_number: Optional declared last sequence number. When
            supplied, later events and a missing terminal event are reported.
        allow_empty: Whether an empty sequence is acceptable.

    Returns:
        An :class:`EventSequenceReport` whose findings are ordered by sequence
        number, then reason code, then opaque event identifier.

    Raises:
        EventSequenceError: If the input is structurally invalid.
    """

    _validate_identifier(run_id, "run_id")
    _validate_sequence_number(expected_start, "expected_start")
    if terminal_sequence_number is not None:
        _validate_sequence_number(terminal_sequence_number, "terminal_sequence_number")
        if terminal_sequence_number < expected_start:
            raise EventSequenceError(
                "terminal_before_start", "terminal_sequence_number"
            )
    if type(allow_empty) is not bool:
        raise EventSequenceError("invalid_flag", "allow_empty")

    ordered = _materialize(references)
    findings: list[SequenceFinding] = []

    if not ordered:
        if not allow_empty:
            findings.append(SequenceFinding("empty_sequence"))
        return _build_report(
            run_id=run_id,
            ordered=ordered,
            terminal_sequence_number=terminal_sequence_number,
            findings=findings,
        )

    seen_numbers: set[int] = set()
    seen_event_ids: set[str] = set()
    previous: int | None = None
    for reference in ordered:
        number = reference.sequence_number
        if reference.run_id != run_id:
            findings.append(
                SequenceFinding("cross_run_reference", number, reference.event_id)
            )
        if number in seen_numbers:
            findings.append(
                SequenceFinding("duplicate_sequence_number", number, reference.event_id)
            )
        if reference.event_id in seen_event_ids:
            findings.append(
                SequenceFinding("duplicate_event_id", number, reference.event_id)
            )
        if previous is not None and number <= previous:
            findings.append(SequenceFinding("out_of_order", number, reference.event_id))
        if terminal_sequence_number is not None and number > terminal_sequence_number:
            findings.append(
                SequenceFinding("post_terminal_event", number, reference.event_id)
            )
        seen_numbers.add(number)
        seen_event_ids.add(reference.event_id)
        previous = number

    distinct = sorted(seen_numbers)
    if distinct[0] != expected_start:
        findings.append(SequenceFinding("start_sequence_mismatch", distinct[0]))
    for lower, upper in zip(distinct, distinct[1:]):
        if upper > lower + 1:
            findings.append(SequenceFinding("sequence_gap", lower + 1))
    if (
        terminal_sequence_number is not None
        and terminal_sequence_number not in seen_numbers
    ):
        findings.append(
            SequenceFinding("terminal_event_missing", terminal_sequence_number)
        )

    return _build_report(
        run_id=run_id,
        ordered=ordered,
        terminal_sequence_number=terminal_sequence_number,
        findings=findings,
    )


def _build_report(
    *,
    run_id: str,
    ordered: tuple[EventReference, ...],
    terminal_sequence_number: int | None,
    findings: list[SequenceFinding],
) -> EventSequenceReport:
    numbers = [reference.sequence_number for reference in ordered]
    return EventSequenceReport(
        run_id=run_id,
        event_count=len(ordered),
        first_sequence_number=numbers[0] if numbers else None,
        last_sequence_number=numbers[-1] if numbers else None,
        terminal_sequence_number=terminal_sequence_number,
        findings=_order_findings(findings),
    )


def _order_findings(findings: list[SequenceFinding]) -> tuple[SequenceFinding, ...]:
    ordered = sorted(
        findings,
        key=lambda finding: (
            -1 if finding.sequence_number is None else finding.sequence_number,
            finding.reason_code,
            finding.event_id or "",
        ),
    )
    if len(ordered) > MAX_SEQUENCE_FINDINGS:
        truncated = ordered[: MAX_SEQUENCE_FINDINGS - 1]
        truncated.append(SequenceFinding("findings_truncated"))
        return tuple(truncated)
    return tuple(ordered)


def _materialize(references: Any) -> tuple[EventReference, ...]:
    if isinstance(references, (str, bytes, bytearray, EventReference)):
        raise EventSequenceError("invalid_sequence", "references")
    try:
        iterator = iter(references)
    except TypeError:
        pass
    else:
        collected: list[EventReference] = []
        for item in iterator:
            if not isinstance(item, EventReference):
                raise EventSequenceError("invalid_reference_type", "references")
            if len(collected) == MAX_EVENT_SEQUENCE_LENGTH:
                raise EventSequenceError("too_many_events", "references")
            collected.append(item)
        return tuple(collected)
    raise EventSequenceError("invalid_sequence", "references")


def _validate_identifier(value: Any, field_name: str) -> None:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise EventSequenceError("invalid_identifier", field_name)


def _validate_sequence_number(value: Any, field_name: str) -> None:
    if type(value) is not int:
        raise EventSequenceError("invalid_sequence_number", field_name)
    if value < 0 or value > MAX_SEQUENCE_NUMBER:
        raise EventSequenceError("sequence_number_out_of_range", field_name)


__all__ = [
    "EVENT_SEQUENCE_SCHEMA_VERSION",
    "MAX_EVENT_SEQUENCE_LENGTH",
    "MAX_SEQUENCE_FINDINGS",
    "MAX_SEQUENCE_NUMBER",
    "SEQUENCE_REASON_CODES",
    "EventReference",
    "EventSequenceError",
    "EventSequenceReport",
    "SequenceFinding",
    "validate_event_sequence",
]
