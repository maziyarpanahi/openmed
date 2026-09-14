"""Fail-closed handling for clinical summaries without approved evidence.

This module is the boundary immediately before a local summary generator.  It
accepts an already-filtered evidence collection, removes malformed or
explicitly unapproved records, and refuses to invoke the generator when no
approved record remains.  Refusals contain a stable code and aggregate counts
only; source text, extracted values, identifiers, and upstream exception text
are never copied into a refusal, report, or exception.

The boundary is deliberately local-only.  It does not load a model, access a
network, or decide whether a clinical finding is true.  A non-empty result is
still an assistive output that requires qualified clinical review.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, TypeVar

__all__ = [
    "EMPTY_EVIDENCE_REFUSAL_CODE",
    "SUMMARY_EMPTY_EVIDENCE_DISCLAIMER",
    "SUMMARY_EMPTY_EVIDENCE_REFUSAL_CODE",
    "SUMMARY_EMPTY_EVIDENCE_SCHEMA_VERSION",
    "SummaryEmptyEvidenceError",
    "SummaryEmptyEvidenceReason",
    "SummaryEmptyEvidenceRefusal",
    "build_summary_empty_evidence_refusal",
    "guard_summary_generation",
    "require_summary_evidence",
    "short_circuit_summary_generation",
]


SUMMARY_EMPTY_EVIDENCE_SCHEMA_VERSION: Final[int] = 1
SUMMARY_EMPTY_EVIDENCE_DISCLAIMER: Final[str] = (
    "Clinical summary generation was refused because no approved evidence "
    "survived filtering. This is an assistive safety gate, not a clinical "
    "decision; qualified clinical review is required."
)


class SummaryEmptyEvidenceReason(str, Enum):
    """Stable reason returned when no approved evidence can ground a summary."""

    EMPTY_APPROVED_EVIDENCE = "empty_approved_evidence"
    # These aliases make the same serialized code discoverable under common
    # terminology without creating multiple refusal categories.
    NO_APPROVED_EVIDENCE = "empty_approved_evidence"
    INVALID_APPROVED_EVIDENCE = "empty_approved_evidence"


SUMMARY_EMPTY_EVIDENCE_REFUSAL_CODE: Final[str] = (
    SummaryEmptyEvidenceReason.EMPTY_APPROVED_EVIDENCE.value
)
EMPTY_EVIDENCE_REFUSAL_CODE: Final[str] = SUMMARY_EMPTY_EVIDENCE_REFUSAL_CODE

SUMMARY_REFUSAL_STATUS: Final[str] = "refused"
_MISSING = object()
_UNAPPROVED = object()
_INVALID = object()
_INVALID_CONTAINER = object()

_CONTAINER_FIELDS: Final[tuple[str, ...]] = (
    "approved_evidence",
    "evidence",
    "records",
    "items",
)
_VALIDITY_FIELDS: Final[tuple[str, ...]] = ("valid", "is_valid")
_PAYLOAD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "claim",
        "code",
        "content",
        "data",
        "display",
        "end",
        "entity",
        "evidence",
        "evidence_id",
        "fact",
        "id",
        "label",
        "metadata",
        "offset",
        "payload",
        "provenance",
        "record",
        "source_end",
        "source_id",
        "source_ref",
        "source_span",
        "source_start",
        "start",
        "text",
        "value",
    }
)
_CONTROL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "approved",
        "approval",
        "approval_status",
        "is_approved",
        "is_valid",
        "review_status",
        "state",
        "status",
        "valid",
    }
)
_RECORD_MARKERS: Final[frozenset[str]] = _PAYLOAD_FIELDS | _CONTROL_FIELDS
_RECOGNIZED_OBJECT_FIELDS: Final[tuple[str, ...]] = tuple(sorted(_RECORD_MARKERS))

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SummaryEmptyEvidenceRefusal:
    """A deterministic, value-free refusal returned before summary generation.

    ``input_evidence_count`` is the number of candidates supplied to the
    boundary.  ``excluded_evidence_count`` is the number that was malformed or
    explicitly unapproved.  A refusal is only valid when every input candidate
    was excluded, so ``approved_evidence_count`` is always zero.
    """

    input_evidence_count: int
    excluded_evidence_count: int
    approved_evidence_count: int = 0
    status: str = SUMMARY_REFUSAL_STATUS
    reason: SummaryEmptyEvidenceReason = (
        SummaryEmptyEvidenceReason.EMPTY_APPROVED_EVIDENCE
    )
    schema_version: int = SUMMARY_EMPTY_EVIDENCE_SCHEMA_VERSION
    requires_clinician_review: bool = True
    autonomous_decision: bool = False
    disclaimer: str = SUMMARY_EMPTY_EVIDENCE_DISCLAIMER

    def __post_init__(self) -> None:
        _validate_count(self.input_evidence_count, "input_evidence_count")
        _validate_count(self.excluded_evidence_count, "excluded_evidence_count")
        _validate_count(self.approved_evidence_count, "approved_evidence_count")
        if self.approved_evidence_count != 0:
            raise ValueError("refusal approved evidence count must be zero")
        if self.excluded_evidence_count != self.input_evidence_count:
            raise ValueError("refusal evidence counts are inconsistent")
        if self.status != SUMMARY_REFUSAL_STATUS:
            raise ValueError("summary evidence refusal status is invalid")
        if not isinstance(self.reason, SummaryEmptyEvidenceReason):
            raise ValueError("summary evidence refusal reason is invalid")
        if self.reason.value != SUMMARY_EMPTY_EVIDENCE_REFUSAL_CODE:
            raise ValueError("summary evidence refusal reason is invalid")
        if type(self.schema_version) is not int or (
            self.schema_version != SUMMARY_EMPTY_EVIDENCE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported summary evidence refusal schema")
        if self.requires_clinician_review is not True:
            raise ValueError("summary evidence refusal must require review")
        if self.autonomous_decision is not False:
            raise ValueError("summary evidence refusal cannot be autonomous")
        if self.disclaimer != SUMMARY_EMPTY_EVIDENCE_DISCLAIMER:
            raise ValueError("summary evidence refusal disclaimer is invalid")

    @property
    def refusal_code(self) -> str:
        """Return the stable machine-readable refusal code."""

        return self.reason.value

    @property
    def reason_code(self) -> str:
        """Return the refusal code under the reason-oriented naming form."""

        return self.refusal_code

    @property
    def code(self) -> str:
        """Return the refusal code under the concise naming form."""

        return self.refusal_code

    @property
    def input_count(self) -> int:
        """Return the number of candidates supplied to the gate."""

        return self.input_evidence_count

    @property
    def approved_count(self) -> int:
        """Return the number of evidence records eligible for generation."""

        return self.approved_evidence_count

    @property
    def excluded_count(self) -> int:
        """Return the number of excluded candidates."""

        return self.excluded_evidence_count

    @property
    def invalid_evidence_count(self) -> int:
        """Return the value-free count of candidates excluded by the gate."""

        return self.excluded_evidence_count

    @property
    def rejected_evidence_count(self) -> int:
        """Return the value-free count of candidates excluded by the gate."""

        return self.excluded_evidence_count

    @property
    def refused(self) -> bool:
        """Return whether generation was refused."""

        return True

    @property
    def requires_review(self) -> bool:
        """Return the mandatory human-review flag."""

        return self.requires_clinician_review

    def to_dict(self) -> dict[str, Any]:
        """Return counts and fixed guardrail metadata without source values."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "refusal_code": self.refusal_code,
            "input_evidence_count": self.input_evidence_count,
            "approved_evidence_count": self.approved_evidence_count,
            "excluded_evidence_count": self.excluded_evidence_count,
            "requires_clinician_review": self.requires_clinician_review,
            "autonomous_decision": self.autonomous_decision,
            "disclaimer": self.disclaimer,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON containing no evidence values."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )

    def require_evidence(self) -> None:
        """Raise a fixed-code exception for strict generation callers."""

        raise SummaryEmptyEvidenceError(self)


class SummaryEmptyEvidenceError(ValueError):
    """Fixed-code exception for callers that require evidence before generation.

    The exception message contains only the stable refusal code.  The complete
    value-free refusal report remains available through :attr:`refusal`.
    """

    def __init__(self, refusal: SummaryEmptyEvidenceRefusal) -> None:
        if not isinstance(refusal, SummaryEmptyEvidenceRefusal):
            raise TypeError("refusal must be a SummaryEmptyEvidenceRefusal")
        self.refusal = refusal
        super().__init__(f"summary_refused_{refusal.refusal_code}")

    @property
    def refusal_code(self) -> str:
        """Return the stable machine-readable refusal code."""

        return self.refusal.refusal_code

    @property
    def reason_code(self) -> str:
        """Return the stable refusal code under the reason naming form."""

        return self.refusal_code

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free refusal report carried by the exception."""

        return self.refusal.to_dict()


def build_summary_empty_evidence_refusal(
    approved_evidence: Iterable[Any] | Mapping[str, Any] | None = None,
) -> SummaryEmptyEvidenceRefusal | None:
    """Build a refusal when no approved evidence survives validation.

    Args:
        approved_evidence: An already-filtered evidence collection. Mappings
            can be a single evidence record or a container under ``evidence``,
            ``records``, ``items``, or ``approved_evidence``. Missing approval
            metadata means the caller is asserting that the collection is
            approved; explicit rejection or invalidity is still excluded.

    Returns:
        A value-free refusal when no valid approved candidate remains, or
        ``None`` when at least one candidate can be passed to a generator.
    """

    rows, selected, excluded_count = _partition_evidence(approved_evidence)
    if selected:
        return None
    return _refusal(len(rows), excluded_count)


def require_summary_evidence(
    approved_evidence: Iterable[Any] | Mapping[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Return approved candidates or raise a fixed-code refusal exception.

    The returned tuple is intended for a local generator.  When this function
    refuses, the exception contains only counts and a stable code.
    """

    rows, selected, excluded_count = _partition_evidence(approved_evidence)
    if not selected:
        _refusal(len(rows), excluded_count).require_evidence()
    return selected


def guard_summary_generation(
    approved_evidence: Iterable[Any] | Mapping[str, Any] | None,
    summary_generator: Callable[[tuple[Any, ...]], _T],
) -> _T | SummaryEmptyEvidenceRefusal:
    """Invoke a local summary generator only when evidence survives filtering.

    The generator is called exactly once with a tuple containing only valid,
    approved candidates. It is never called for an empty or wholly invalid
    collection. The generator's result is returned unchanged; an empty-input
    path returns :class:`SummaryEmptyEvidenceRefusal` instead.

    No model, filesystem, or network operation is performed by this boundary.
    A caller-supplied generator may perform its own work, so local-only use is
    an explicit property of the generator passed by the caller.
    """

    if not callable(summary_generator):
        raise TypeError("summary_generator must be callable")
    rows, selected, excluded_count = _partition_evidence(approved_evidence)
    if not selected:
        return _refusal(len(rows), excluded_count)
    return summary_generator(selected)


# Descriptive aliases keep the gate discoverable for callers using either the
# short-circuit or guard terminology without creating a second implementation.
short_circuit_summary_generation = guard_summary_generation


def _refusal(input_count: int, excluded_count: int) -> SummaryEmptyEvidenceRefusal:
    """Construct a refusal after the private evidence partition is complete."""

    return SummaryEmptyEvidenceRefusal(
        input_evidence_count=input_count,
        excluded_evidence_count=excluded_count,
    )


def _partition_evidence(
    evidence: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...], int]:
    rows = _materialize_evidence(evidence)
    selected: list[Any] = []
    excluded_count = 0
    for row in rows:
        if _classify_evidence(row) is True:
            selected.append(row)
        else:
            excluded_count += 1
    return rows, tuple(selected), excluded_count


def _materialize_evidence(
    evidence: object,
) -> tuple[Any, ...]:
    if evidence is None:
        return ()
    if isinstance(evidence, Mapping):
        if _looks_like_record(evidence):
            return (evidence,)
        nested = _first_field(evidence, _CONTAINER_FIELDS)
        if nested is not _MISSING:
            return _materialize_evidence(nested)
        return (evidence,)
    if isinstance(evidence, (str, bytes, bytearray)):
        return (evidence,)
    if not isinstance(evidence, Iterable):
        return (_INVALID_CONTAINER,)
    try:
        return tuple(evidence)
    except Exception:
        # A broken or non-iterable source is represented as one invalid item.
        # The sentinel never escapes into a report or exception.
        return (_INVALID_CONTAINER,)


def _classify_evidence(value: Any) -> object:
    if value is _INVALID_CONTAINER or value is None:
        return _INVALID
    if isinstance(value, (str, bytes, bytearray, bool, int, float, complex)):
        return _INVALID

    data = _mapping_view(value)
    if data is None or not _has_payload(data):
        return _INVALID

    for field_name in _VALIDITY_FIELDS:
        validity = _first_field(data, (field_name,))
        if validity is _MISSING:
            continue
        if type(validity) is not bool:
            return _INVALID
        if not validity:
            return _INVALID

    approval = _approval_state(data)
    if approval is _UNAPPROVED or approval is _INVALID:
        return approval
    return True


def _mapping_view(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    try:
        attributes = {
            name: getattr(value, name)
            for name in _RECOGNIZED_OBJECT_FIELDS
            if hasattr(value, name)
        }
    except Exception:
        return None
    if attributes:
        return attributes
    try:
        raw_attributes = vars(value)
    except Exception:
        return None
    if not isinstance(raw_attributes, Mapping) or not raw_attributes:
        return None
    return raw_attributes


def _has_payload(data: Mapping[str, Any]) -> bool:
    try:
        keys = tuple(data.keys())
    except Exception:
        return False
    if any(key not in _CONTROL_FIELDS for key in keys):
        return True
    return any(
        _first_field(data, (field_name,)) is not _MISSING
        for field_name in _PAYLOAD_FIELDS
    )


def _approval_state(data: Mapping[str, Any]) -> object:
    approved = _first_field(data, ("approved", "is_approved"))
    if approved is not _MISSING:
        if type(approved) is not bool:
            return _INVALID
        return True if approved else _UNAPPROVED

    approval = _first_field(data, ("approval",))
    if approval is not _MISSING:
        if isinstance(approval, Mapping):
            nested = _first_field(
                approval,
                ("approved", "is_approved", "status", "state"),
            )
            if nested is _MISSING:
                return _INVALID
            return _approval_value(nested)
        return _approval_value(approval)

    for field_name in ("approval_status", "review_status", "state", "status"):
        value = _first_field(data, (field_name,))
        if value is _MISSING:
            continue
        state = _approval_value(
            value,
            allow_unknown_status=field_name in {"state", "status"},
        )
        if state is not _MISSING:
            return state
    # The parameter is explicitly an approved-evidence boundary. Callers may
    # therefore omit redundant approval metadata, while explicit rejection is
    # still fail-closed above.
    return True


def _approval_value(value: Any, *, allow_unknown_status: bool = False) -> object:
    if type(value) is bool:
        return True if value else _UNAPPROVED
    if not isinstance(value, str):
        return _INVALID
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"approved", "accepted", "reviewed", "verified"}:
        return True
    if normalized in {
        "rejected",
        "unapproved",
        "pending",
        "needs_review",
        "review_required",
        "excluded",
    }:
        return _UNAPPROVED
    if normalized in {"invalid", "malformed", "unusable"}:
        return _INVALID
    if allow_unknown_status:
        return _MISSING
    return _INVALID


def _looks_like_record(value: Mapping[str, Any]) -> bool:
    try:
        return any(name in value for name in _RECORD_MARKERS)
    except Exception:
        return True


def _first_field(value: Mapping[str, Any], names: Iterable[str]) -> object:
    for name in names:
        try:
            if name in value:
                return value[name]
        except Exception:
            return _INVALID
    return _MISSING


def _validate_count(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
