"""Evidence-bound medication-change statements for guarded summaries.

Only typed, assertion-aware relations with opaque evidence identifiers can
produce statements. Incomplete, non-confirmed, or conflicting records are
withheld and represented by value-free review codes. The renderer describes
individual changes and never infers a final medication regimen.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

from .relations.assertion_filter import (
    RELATION_ASSERTION_STATUSES,
    RELATION_CONFIRMED,
)

SUMMARY_MEDICATION_CHANGES_SCHEMA_VERSION: Final = 1
SUMMARY_MEDICATION_CHANGES_ADVISORY: Final = (
    "Medication-change statements are evidence-bound review aids, not a final "
    "regimen, prescription decision, or substitute for clinician review."
)

MedicationChangeType = Literal["started", "stopped", "dose_changed"]
MedicationChangeIssueCode = Literal[
    "assertion_not_confirmed",
    "conflicting_records",
    "incomplete_dose_change",
    "invalid_effective_time",
    "invalid_evidence_identifier",
    "missing_change_type",
    "missing_evidence",
    "missing_medication",
    "unsupported_assertion_status",
    "unsupported_change_type",
]

_CHANGE_TYPES = frozenset({"started", "stopped", "dose_changed"})
_OPAQUE_EVIDENCE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})?)?$"
)


@dataclass(frozen=True, slots=True)
class MedicationChangeRelation:
    """One structured medication-change relation supplied to the renderer.

    Optional fields allow an extractor to pass an incomplete record through to
    review routing instead of raising or filling in a value. ``evidence_ids``
    must use opaque ``sha256:<hex>`` identifiers before a statement can render.
    """

    medication: str | None
    change_type: str | None
    assertion_status: str | None
    evidence_ids: tuple[str, ...] = ()
    previous_dose: str | None = None
    new_dose: str | None = None
    effective_time: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.medication,
            self.change_type,
            self.assertion_status,
            self.previous_dose,
            self.new_dose,
            self.effective_time,
        ):
            if value is not None and type(value) is not str:
                raise TypeError("medication-change text fields must be strings")
        try:
            evidence_ids = tuple(self.evidence_ids)
        except TypeError:
            raise TypeError("evidence_ids must be an iterable of strings") from None
        if any(type(identifier) is not str for identifier in evidence_ids):
            raise TypeError("evidence_ids must contain strings")
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True, slots=True)
class MedicationChangeIssue:
    """Value-free review reason associated with input record indexes."""

    code: MedicationChangeIssueCode
    record_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        indexes = tuple(self.record_indexes)
        if not indexes or any(type(index) is not int or index < 0 for index in indexes):
            raise ValueError("review issue requires non-negative record indexes")
        object.__setattr__(self, "record_indexes", tuple(sorted(set(indexes))))

    def to_dict(self) -> dict[str, object]:
        """Return the controlled value-free review representation."""

        return {"code": self.code, "record_indexes": list(self.record_indexes)}


@dataclass(frozen=True, slots=True)
class MedicationChangeStatement:
    """One rendered change statement with explicit evidence identifiers."""

    text: str
    evidence_ids: tuple[str, ...]
    effective_time: str | None = None

    def __post_init__(self) -> None:
        if type(self.text) is not str or not self.text:
            raise ValueError("medication-change statement text must be non-empty")
        evidence_ids = tuple(sorted(set(self.evidence_ids)))
        if not evidence_ids or any(
            _OPAQUE_EVIDENCE_ID_RE.fullmatch(identifier) is None
            for identifier in evidence_ids
        ):
            raise ValueError("statement evidence identifiers must be opaque")
        object.__setattr__(self, "evidence_ids", evidence_ids)

    def to_dict(self) -> dict[str, object]:
        """Return the statement and its evidence binding."""

        payload: dict[str, object] = {
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
        }
        if self.effective_time is not None:
            payload["effective_time"] = self.effective_time
        return payload


@dataclass(frozen=True, slots=True)
class MedicationChangeSummary:
    """Rendered statements plus value-free review routing."""

    statements: tuple[MedicationChangeStatement, ...]
    issues: tuple[MedicationChangeIssue, ...]
    record_count: int
    schema_version: int = SUMMARY_MEDICATION_CHANGES_SCHEMA_VERSION
    advisory: str = SUMMARY_MEDICATION_CHANGES_ADVISORY

    def __post_init__(self) -> None:
        if type(self.record_count) is not int or self.record_count < 0:
            raise ValueError("record_count must be a non-negative integer")
        if self.schema_version != SUMMARY_MEDICATION_CHANGES_SCHEMA_VERSION:
            raise ValueError("unsupported medication-change schema version")
        statements = tuple(self.statements)
        issues = tuple(self.issues)
        if any(not isinstance(item, MedicationChangeStatement) for item in statements):
            raise TypeError("statements must contain MedicationChangeStatement values")
        if any(not isinstance(item, MedicationChangeIssue) for item in issues):
            raise TypeError("issues must contain MedicationChangeIssue values")
        object.__setattr__(
            self, "statements", tuple(sorted(statements, key=_statement_key))
        )
        object.__setattr__(self, "issues", tuple(sorted(issues, key=_issue_key)))

    @property
    def review_required(self) -> bool:
        """Return whether any relation was withheld for review."""

        return bool(self.issues)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready summary view."""

        return {
            "schema_version": self.schema_version,
            "record_count": self.record_count,
            "statement_count": len(self.statements),
            "review_required": self.review_required,
            "statements": [statement.to_dict() for statement in self.statements],
            "issues": [issue.to_dict() for issue in self.issues],
            "advisory": self.advisory,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def render_medication_change_summary(
    relations: Iterable[MedicationChangeRelation],
) -> MedicationChangeSummary:
    """Render evidence-bound change events without inferring a final regimen.

    Args:
        relations: Typed medication-change records. Each record explicitly
            carries an assertion status and opaque evidence identifiers.

    Returns:
        Confirmed, complete, non-conflicting change statements plus controlled
        review issues for every withheld record.
    """

    if isinstance(relations, (str, bytes, bytearray)):
        raise TypeError("relations must contain MedicationChangeRelation values")
    try:
        records = tuple(relations)
    except TypeError:
        raise TypeError(
            "relations must contain MedicationChangeRelation values"
        ) from None
    if any(not isinstance(record, MedicationChangeRelation) for record in records):
        raise TypeError("relations must contain MedicationChangeRelation values")

    issues: list[MedicationChangeIssue] = []
    eligible: dict[int, MedicationChangeRelation] = {}
    for index, record in enumerate(records):
        codes = _record_issue_codes(record)
        if codes:
            issues.extend(
                MedicationChangeIssue(code=code, record_indexes=(index,))
                for code in codes
            )
        else:
            eligible[index] = record

    conflicted_indexes = _conflicted_indexes(eligible)
    for indexes in conflicted_indexes:
        issues.append(
            MedicationChangeIssue(
                code="conflicting_records",
                record_indexes=indexes,
            )
        )
    withheld = {index for indexes in conflicted_indexes for index in indexes}

    grouped: dict[
        tuple[str, str, str | None, str | None, str | None],
        list[MedicationChangeRelation],
    ] = defaultdict(list)
    for index, record in eligible.items():
        if index not in withheld:
            grouped[_statement_signature(record)].append(record)

    statements = tuple(
        _render_statement(group)
        for _, group in sorted(grouped.items(), key=lambda item: item[0])
    )
    return MedicationChangeSummary(
        statements=statements,
        issues=tuple(issues),
        record_count=len(records),
    )


def _record_issue_codes(
    record: MedicationChangeRelation,
) -> tuple[MedicationChangeIssueCode, ...]:
    codes: list[MedicationChangeIssueCode] = []
    medication = _clean(record.medication)
    change_type = _clean(record.change_type)
    assertion_status = _clean(record.assertion_status)

    if medication is None:
        codes.append("missing_medication")
    if change_type is None:
        codes.append("missing_change_type")
    elif change_type not in _CHANGE_TYPES:
        codes.append("unsupported_change_type")
    if assertion_status not in RELATION_ASSERTION_STATUSES:
        codes.append("unsupported_assertion_status")
    elif assertion_status != RELATION_CONFIRMED:
        codes.append("assertion_not_confirmed")
    if not record.evidence_ids:
        codes.append("missing_evidence")
    elif any(
        _OPAQUE_EVIDENCE_ID_RE.fullmatch(identifier) is None
        for identifier in record.evidence_ids
    ):
        codes.append("invalid_evidence_identifier")
    if record.effective_time is not None and (
        _ISO_TIMESTAMP_RE.fullmatch(record.effective_time.strip()) is None
    ):
        codes.append("invalid_effective_time")
    if change_type == "dose_changed" and (
        _clean(record.previous_dose) is None
        or _clean(record.new_dose) is None
        or _clean(record.previous_dose) == _clean(record.new_dose)
    ):
        codes.append("incomplete_dose_change")
    return tuple(codes)


def _conflicted_indexes(
    eligible: dict[int, MedicationChangeRelation],
) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[str, str | None], list[int]] = defaultdict(list)
    for index, record in eligible.items():
        groups[(_normalized_medication(record), _clean(record.effective_time))].append(
            index
        )

    conflicts: list[tuple[int, ...]] = []
    for indexes in groups.values():
        signatures = {_change_signature(eligible[index]) for index in indexes}
        if len(signatures) > 1:
            conflicts.append(tuple(sorted(indexes)))
    return tuple(sorted(conflicts))


def _render_statement(
    records: list[MedicationChangeRelation],
) -> MedicationChangeStatement:
    record = records[0]
    medication = _clean(record.medication)
    change_type = _clean(record.change_type)
    if medication is None or change_type is None:  # Guarded by eligibility.
        raise RuntimeError("eligible medication-change record became incomplete")
    if change_type == "started":
        text = f"Started {medication}."
    elif change_type == "stopped":
        text = f"Stopped {medication}."
    else:
        text = (
            f"Changed {medication} dose from {_clean(record.previous_dose)} "
            f"to {_clean(record.new_dose)}."
        )
    return MedicationChangeStatement(
        text=text,
        evidence_ids=tuple(
            identifier for item in records for identifier in item.evidence_ids
        ),
        effective_time=_clean(record.effective_time),
    )


def _statement_signature(
    record: MedicationChangeRelation,
) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        _normalized_medication(record),
        _clean(record.change_type) or "",
        _clean(record.effective_time),
        _clean(record.previous_dose),
        _clean(record.new_dose),
    )


def _change_signature(
    record: MedicationChangeRelation,
) -> tuple[str | None, str | None, str | None]:
    return (
        _clean(record.change_type),
        _clean(record.previous_dose),
        _clean(record.new_dose),
    )


def _normalized_medication(record: MedicationChangeRelation) -> str:
    return (_clean(record.medication) or "").casefold()


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _statement_key(
    statement: MedicationChangeStatement,
) -> tuple[str, str, tuple[str, ...]]:
    return (
        statement.effective_time or "",
        statement.text.casefold(),
        statement.evidence_ids,
    )


def _issue_key(issue: MedicationChangeIssue) -> tuple[tuple[int, ...], str]:
    return issue.record_indexes, issue.code


__all__ = [
    "SUMMARY_MEDICATION_CHANGES_ADVISORY",
    "SUMMARY_MEDICATION_CHANGES_SCHEMA_VERSION",
    "MedicationChangeIssue",
    "MedicationChangeIssueCode",
    "MedicationChangeRelation",
    "MedicationChangeStatement",
    "MedicationChangeSummary",
    "MedicationChangeType",
    "render_medication_change_summary",
]
