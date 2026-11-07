"""Deterministic temporal-order checks for evidence-bound summary claims.

The validator consumes only event identifiers referenced by a claim and an
existing privacy-safe :class:`~openmed.clinical.timeline.Timeline`. It never
accepts or rewrites claim text. Results expose pair indexes and controlled
reason codes rather than event identifiers, source text, or clinical values.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

from .timeline import Timeline

SUMMARY_TEMPORAL_ORDER_SCHEMA_VERSION: Final = 1
SUMMARY_TEMPORAL_ORDER_ADVISORY: Final = (
    "Temporal-order validation is a deterministic review aid. Invalid or "
    "ambiguous ordering requires review and does not rewrite the clinical claim."
)

TemporalOrderStatus = Literal["validated", "review_required"]
TemporalOrderCode = Literal[
    "duplicate_reference",
    "insufficient_references",
    "order_inversion",
    "order_unresolved",
    "unknown_event",
]


@dataclass(frozen=True, slots=True)
class TemporalOrderFinding:
    """Value-free result for one claim-reference pair or collection check."""

    code: TemporalOrderCode
    first_reference_index: int | None = None
    second_reference_index: int | None = None

    def __post_init__(self) -> None:
        indexes = (self.first_reference_index, self.second_reference_index)
        if self.code == "insufficient_references":
            if indexes != (None, None):
                raise ValueError("insufficient-reference finding cannot name a pair")
            return
        if any(type(index) is not int or index < 0 for index in indexes):
            raise ValueError("temporal-order finding requires valid pair indexes")
        if self.first_reference_index >= self.second_reference_index:
            raise ValueError("temporal-order finding indexes must follow claim order")

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic value-free finding."""

        payload: dict[str, object] = {"code": self.code}
        if self.first_reference_index is not None:
            payload["reference_indexes"] = [
                self.first_reference_index,
                self.second_reference_index,
            ]
        return payload


@dataclass(frozen=True, slots=True)
class SummaryTemporalOrderResult:
    """Review routing for one claim's ordered event references."""

    reference_count: int
    findings: tuple[TemporalOrderFinding, ...]
    schema_version: int = SUMMARY_TEMPORAL_ORDER_SCHEMA_VERSION
    advisory: str = SUMMARY_TEMPORAL_ORDER_ADVISORY

    def __post_init__(self) -> None:
        if type(self.reference_count) is not int or self.reference_count < 0:
            raise ValueError("reference_count must be a non-negative integer")
        if self.schema_version != SUMMARY_TEMPORAL_ORDER_SCHEMA_VERSION:
            raise ValueError("unsupported temporal-order schema version")
        findings = tuple(self.findings)
        if any(not isinstance(item, TemporalOrderFinding) for item in findings):
            raise TypeError("findings must contain TemporalOrderFinding values")
        object.__setattr__(
            self,
            "findings",
            tuple(sorted(findings, key=_finding_key)),
        )

    @property
    def status(self) -> TemporalOrderStatus:
        """Return whether the claim order is supported or needs review."""

        return "review_required" if self.findings else "validated"

    @property
    def review_required(self) -> bool:
        """Return whether invalid or ambiguous ordering was found."""

        return bool(self.findings)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic result without event identifiers or text."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "review_required": self.review_required,
            "reference_count": self.reference_count,
            "findings": [finding.to_dict() for finding in self.findings],
            "advisory": self.advisory,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON suitable for a value-free audit artifact."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def validate_summary_temporal_order(
    claim_event_references: Iterable[str],
    evidence_timeline: Timeline,
) -> SummaryTemporalOrderResult:
    """Validate claim-reference order against established timeline relations.

    Every ordered pair in the claim is checked against the transitive closure
    of retained ``BEFORE`` and ``AFTER`` timeline evidence. A reverse path is an
    inversion. Missing events, duplicate references, and pairs with no proven
    direction are unresolved and therefore routed to review.

    Args:
        claim_event_references: Event identifiers in the order asserted by the
            claim. Claim text is deliberately not accepted.
        evidence_timeline: Existing privacy-safe clinical evidence timeline.

    Returns:
        A value-free validation result. The function never changes the claim.
    """

    if isinstance(claim_event_references, (str, bytes, bytearray)):
        raise TypeError("claim event references must be an iterable of identifiers")
    if not isinstance(evidence_timeline, Timeline):
        raise TypeError("evidence_timeline must be a Timeline")
    try:
        references = tuple(claim_event_references)
    except TypeError:
        raise TypeError(
            "claim event references must be an iterable of identifiers"
        ) from None
    if any(type(reference) is not str or not reference for reference in references):
        raise ValueError("claim event references must be non-empty strings")

    event_ids = tuple(event.event_id for event in evidence_timeline.events)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("evidence timeline contains duplicate event identifiers")
    known_events = set(event_ids)
    precedes = _precedence_closure(evidence_timeline, known_events)

    if len(references) < 2:
        return SummaryTemporalOrderResult(
            reference_count=len(references),
            findings=(TemporalOrderFinding(code="insufficient_references"),),
        )

    findings: list[TemporalOrderFinding] = []
    for first_index, first_event in enumerate(references[:-1]):
        for second_index in range(first_index + 1, len(references)):
            second_event = references[second_index]
            code = _pair_finding_code(
                first_event,
                second_event,
                known_events=known_events,
                precedes=precedes,
            )
            if code is not None:
                findings.append(
                    TemporalOrderFinding(
                        code=code,
                        first_reference_index=first_index,
                        second_reference_index=second_index,
                    )
                )
    return SummaryTemporalOrderResult(
        reference_count=len(references),
        findings=tuple(findings),
    )


def _pair_finding_code(
    first_event: str,
    second_event: str,
    *,
    known_events: set[str],
    precedes: set[tuple[str, str]],
) -> TemporalOrderCode | None:
    if first_event == second_event:
        return "duplicate_reference"
    if first_event not in known_events or second_event not in known_events:
        return "unknown_event"
    if (first_event, second_event) in precedes:
        return None
    if (second_event, first_event) in precedes:
        return "order_inversion"
    return "order_unresolved"


def _precedence_closure(
    timeline: Timeline,
    known_events: set[str],
) -> set[tuple[str, str]]:
    adjacency = {event_id: set() for event_id in known_events}
    for edge in timeline.kept_edges:
        if edge.relation_type == "BEFORE":
            before, after = edge.source.span_id, edge.target.span_id
        elif edge.relation_type == "AFTER":
            before, after = edge.target.span_id, edge.source.span_id
        else:
            continue
        if before in known_events and after in known_events:
            adjacency[before].add(after)

    closure: set[tuple[str, str]] = set()
    for origin in sorted(known_events):
        pending = list(sorted(adjacency[origin], reverse=True))
        visited: set[str] = set()
        while pending:
            target = pending.pop()
            if target in visited:
                continue
            visited.add(target)
            closure.add((origin, target))
            pending.extend(sorted(adjacency[target] - visited, reverse=True))
    return closure


def _finding_key(finding: TemporalOrderFinding) -> tuple[int, int, str]:
    return (
        finding.first_reference_index
        if finding.first_reference_index is not None
        else -1,
        finding.second_reference_index
        if finding.second_reference_index is not None
        else -1,
        finding.code,
    )


__all__ = [
    "SUMMARY_TEMPORAL_ORDER_ADVISORY",
    "SUMMARY_TEMPORAL_ORDER_SCHEMA_VERSION",
    "SummaryTemporalOrderResult",
    "TemporalOrderCode",
    "TemporalOrderFinding",
    "TemporalOrderStatus",
    "validate_summary_temporal_order",
]
