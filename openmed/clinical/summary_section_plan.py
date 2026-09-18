"""Deterministic, section-preserving planning for local clinical summaries.

The planner is the metadata-only boundary between reviewed clinical evidence
and a local summary generator.  It groups approved evidence by an explicit,
stable source-section identifier and orders both groups and references
deterministically.  A generator can use the returned offsets to resolve source
content inside the operator-controlled process without placing that content in
the plan, its reports, or its exceptions.

This module performs no model loading and no network access.  Plans are
assistive review artifacts: they do not summarize a patient, make a clinical
decision, or qualify evidence as clinically true.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Literal, cast

SUMMARY_SECTION_PLAN_SCHEMA_VERSION: Final[int] = 1
SUMMARY_SECTION_PLAN_DISCLAIMER: Final = (
    "Section-preserving clinical summary plans are deterministic assistive "
    "review artifacts, not clinical decisions or autonomous summaries."
)
SUMMARY_PLAN_STATUS_READY: Final = "ready"
SUMMARY_PLAN_STATUS_REFUSED: Final = "refused"
MAX_SUMMARY_PLAN_EVIDENCE: Final[int] = 4096
MAX_SUMMARY_PLAN_ID_LENGTH: Final[int] = 128

_ID_RE = re.compile(
    rf"^[A-Za-z0-9](?:[-A-Za-z0-9._:]{{0,{MAX_SUMMARY_PLAN_ID_LENGTH - 1}}})$"
)
_MISSING = object()
_UNAPPROVED = object()
_MAX_SOURCE_OFFSET = 2**63 - 1

SummaryPlanStatus = Literal["ready", "refused"]


class SummaryPlanRefusalReason(str, Enum):
    """Finite reasons why a summary section plan cannot be generated."""

    MISSING_SECTION_ID = "missing_section_id"
    # These aliases keep the typed reason discoverable under common wording
    # without creating additional serialized reason values.
    MISSING_SECTION_IDENTIFIER = "missing_section_id"
    NO_STABLE_SECTION_ID = "missing_section_id"
    INVALID_SECTION_ID = "invalid_section_id"
    INVALID_EVIDENCE = "invalid_evidence"
    INVALID_EVIDENCE_ID = "invalid_evidence_id"
    INVALID_SOURCE_OFFSET = "invalid_source_offset"
    INVALID_APPROVAL = "invalid_approval"
    CONFLICTING_SECTION_METADATA = "conflicting_section_metadata"
    UNKNOWN_SECTION_ID = "unknown_section_id"
    EVIDENCE_LIMIT = "evidence_limit"


class SummarySectionPlanError(ValueError):
    """Safe exception for callers that require a ready plan.

    The exception message contains only a fixed refusal code and never echoes
    an evidence identifier, section identifier, source surface, or upstream
    parser message.
    """

    def __init__(
        self,
        reason: SummaryPlanRefusalReason,
        *,
        rejected_count: int = 1,
    ) -> None:
        self.reason = _coerce_refusal_reason(reason)
        self.rejected_count = _safe_count(rejected_count, "rejected_count")
        super().__init__(f"summary_section_plan_{self.reason.value}")


@dataclass(frozen=True, slots=True)
class SummaryPlanRefusal:
    """A typed, value-free refusal returned by the planner."""

    reason: SummaryPlanRefusalReason
    rejected_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _coerce_refusal_reason(self.reason))
        object.__setattr__(
            self,
            "rejected_count",
            _safe_count(self.rejected_count, "rejected_count"),
        )

    @property
    def reason_code(self) -> str:
        """Return the stable machine-readable refusal code."""

        return self.reason.value

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible refusal without input values."""

        return {
            "reason": self.reason.value,
            "rejected_count": self.rejected_count,
        }


@dataclass(frozen=True, slots=True)
class SummaryEvidence:
    """A value-free reference to one approved source evidence item.

    ``evidence_id`` and ``section_id`` are structural identifiers supplied by
    the caller.  Optional source offsets let a local generator resolve the
    reference within its controlled source document.  No source text, label,
    value, or model output is stored on this record.
    """

    evidence_id: str
    section_id: str
    source_start: int | None = None
    source_end: int | None = None
    approved: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _stable_identifier(
                self.evidence_id,
                SummaryPlanRefusalReason.INVALID_EVIDENCE_ID,
            ),
        )
        object.__setattr__(
            self,
            "section_id",
            _stable_identifier(
                self.section_id,
                SummaryPlanRefusalReason.INVALID_SECTION_ID,
            ),
        )
        start, end = _optional_source_offsets(self.source_start, self.source_end)
        if type(self.approved) is not bool:
            raise ValueError("summary evidence approval is invalid")
        object.__setattr__(self, "source_start", start)
        object.__setattr__(self, "source_end", end)

    @property
    def source_offset(self) -> tuple[int, int] | None:
        """Return the optional half-open source offset."""

        if self.source_start is None or self.source_end is None:
            return None
        return self.source_start, self.source_end

    @property
    def section(self) -> str:
        """Return the source section identifier under the concise alias."""

        return self.section_id

    @property
    def source_section_id(self) -> str:
        """Return the source section identifier under its explicit name."""

        return self.section_id

    @property
    def id(self) -> str:
        """Return the evidence identifier under the common object alias."""

        return self.evidence_id

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free JSON representation."""

        payload: dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "section_id": self.section_id,
            "approved": self.approved,
        }
        if self.source_offset is not None:
            payload["source_offset"] = {
                "start": self.source_start,
                "end": self.source_end,
            }
        return payload


# Descriptive aliases make the reference role clear to callers that already
# use "reference" terminology for source-backed records.
SummaryEvidenceReference = SummaryEvidence
SummarySectionEvidence = SummaryEvidence


@dataclass(frozen=True, slots=True)
class SummarySectionGroup:
    """One section-local batch of approved evidence references."""

    section_id: str
    evidence: tuple[SummaryEvidence, ...]
    section_start: int | None = None
    section_end: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "section_id",
            _stable_identifier(
                self.section_id,
                SummaryPlanRefusalReason.INVALID_SECTION_ID,
            ),
        )
        evidence = tuple(self.evidence)
        if any(type(item) is not SummaryEvidence for item in evidence):
            raise ValueError("summary section group evidence is invalid")
        if any(item.section_id != self.section_id for item in evidence):
            raise ValueError("summary section group evidence has a mismatched section")
        if any(not item.approved for item in evidence):
            raise ValueError("summary section group contains unapproved evidence")
        start, end = _optional_source_offsets(self.section_start, self.section_end)
        object.__setattr__(self, "evidence", tuple(sorted(evidence, key=_evidence_key)))
        object.__setattr__(self, "section_start", start)
        object.__setattr__(self, "section_end", end)

    @property
    def evidence_count(self) -> int:
        """Return the number of approved references in this section batch."""

        return len(self.evidence)

    @property
    def source_offset(self) -> tuple[int, int] | None:
        """Return the optional detected source-section offset."""

        if self.section_start is None or self.section_end is None:
            return None
        return self.section_start, self.section_end

    @property
    def references(self) -> tuple[SummaryEvidence, ...]:
        """Return evidence under the source-reference naming convention."""

        return self.evidence

    @property
    def evidence_references(self) -> tuple[SummaryEvidence, ...]:
        """Return the evidence references in this section-local batch."""

        return self.evidence

    @property
    def group_id(self) -> str:
        """Return the stable source section identifier."""

        return self.section_id

    @property
    def source_section_id(self) -> str:
        """Return the source section identifier under its explicit name."""

        return self.section_id

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, value-free section batch."""

        payload: dict[str, Any] = {
            "section_id": self.section_id,
            "evidence": [item.to_dict() for item in self.evidence],
        }
        if self.source_offset is not None:
            payload["source_offset"] = {
                "start": self.section_start,
                "end": self.section_end,
            }
        return payload


SummarySection = SummarySectionGroup
SectionEvidenceGroup = SummarySectionGroup


@dataclass(frozen=True, slots=True)
class SummarySectionPlan:
    """Versioned deterministic plan for section-preserving local generation."""

    sections: tuple[SummarySectionGroup, ...] = ()
    status: SummaryPlanStatus = SUMMARY_PLAN_STATUS_READY
    refusal: SummaryPlanRefusal | None = None
    input_evidence_count: int = 0
    approved_evidence_count: int = 0
    excluded_evidence_count: int = 0
    schema_version: int = SUMMARY_SECTION_PLAN_SCHEMA_VERSION
    requires_clinician_review: bool = True
    autonomous_decision: bool = False
    disclaimer: str = SUMMARY_SECTION_PLAN_DISCLAIMER

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != SUMMARY_SECTION_PLAN_SCHEMA_VERSION
        ):
            raise ValueError("unsupported summary section plan schema")
        if self.status not in {
            SUMMARY_PLAN_STATUS_READY,
            SUMMARY_PLAN_STATUS_REFUSED,
        }:
            raise ValueError("invalid summary section plan status")
        sections = tuple(self.sections)
        if any(type(section) is not SummarySectionGroup for section in sections):
            raise ValueError("summary section plan sections are invalid")
        if tuple(sorted(sections, key=_section_key)) != sections:
            raise ValueError("summary section plan sections are not deterministic")
        for field_name in (
            "input_evidence_count",
            "approved_evidence_count",
            "excluded_evidence_count",
        ):
            _safe_count(getattr(self, field_name), field_name, allow_zero=True)
        if type(self.requires_clinician_review) is not bool:
            raise ValueError("summary section plan review flag is invalid")
        if type(self.autonomous_decision) is not bool:
            raise ValueError("summary section plan autonomy flag is invalid")
        if self.requires_clinician_review is not True:
            raise ValueError("summary section plan must require clinician review")
        if self.autonomous_decision is not False:
            raise ValueError("summary section plan cannot be autonomous")
        if self.disclaimer != SUMMARY_SECTION_PLAN_DISCLAIMER:
            raise ValueError("summary section plan disclaimer is invalid")
        if self.status == SUMMARY_PLAN_STATUS_READY and self.refusal is not None:
            raise ValueError("ready summary section plan cannot contain a refusal")
        if self.status == SUMMARY_PLAN_STATUS_REFUSED:
            if self.refusal is None:
                raise ValueError("refused summary section plan requires a reason")
            if sections:
                raise ValueError("refused summary section plan cannot contain sections")
        if self.approved_evidence_count != sum(
            section.evidence_count for section in sections
        ):
            raise ValueError("summary section plan evidence counts are inconsistent")
        object.__setattr__(self, "sections", sections)

    @property
    def groups(self) -> tuple[SummarySectionGroup, ...]:
        """Return section batches under the planner terminology."""

        return self.sections

    @property
    def batches(self) -> tuple[SummarySectionGroup, ...]:
        """Return section-local generation batches."""

        return self.sections

    @property
    def section_groups(self) -> tuple[SummarySectionGroup, ...]:
        """Return the planned source-section groups."""

        return self.sections

    @property
    def evidence_count(self) -> int:
        """Return the count of selected approved evidence references."""

        return self.approved_evidence_count

    @property
    def ready(self) -> bool:
        """Return whether local generation may consume this plan."""

        return self.status == SUMMARY_PLAN_STATUS_READY

    @property
    def is_ready(self) -> bool:
        """Return :attr:`ready` under the predicate naming convention."""

        return self.ready

    @property
    def refused(self) -> bool:
        """Return whether the plan was refused before generation."""

        return self.status == SUMMARY_PLAN_STATUS_REFUSED

    @property
    def is_refused(self) -> bool:
        """Return :attr:`refused` under the predicate naming convention."""

        return self.refused

    @property
    def refusal_reason(self) -> SummaryPlanRefusalReason | None:
        """Return the typed refusal reason, when planning was refused."""

        return self.refusal.reason if self.refusal is not None else None

    @property
    def reason(self) -> SummaryPlanRefusalReason | None:
        """Return the refusal reason under the concise result alias."""

        return self.refusal_reason

    @property
    def requires_review(self) -> bool:
        """Return the mandatory human-review flag."""

        return self.requires_clinician_review

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic plan report without source values."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "input_evidence_count": self.input_evidence_count,
            "approved_evidence_count": self.approved_evidence_count,
            "excluded_evidence_count": self.excluded_evidence_count,
            "section_count": len(self.sections),
            "sections": [section.to_dict() for section in self.sections],
            "requires_clinician_review": self.requires_clinician_review,
            "autonomous_decision": self.autonomous_decision,
            "disclaimer": self.disclaimer,
        }
        if self.refusal is not None:
            payload["refusal"] = self.refusal.to_dict()
        return payload

    def to_json(self) -> str:
        """Return byte-stable JSON for review and regression artifacts."""

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

    def require_ready(self) -> "SummarySectionPlan":
        """Return this plan or raise a fixed-code refusal exception."""

        if self.refusal is not None:
            raise SummarySectionPlanError(
                self.refusal.reason,
                rejected_count=self.refusal.rejected_count,
            )
        return self


@dataclass(frozen=True, slots=True)
class _SectionMetadata:
    section_id: str
    start: int | None = None
    end: int | None = None


class _PlanInputError(Exception):
    """Internal fixed-code input error that never escapes with raw content."""

    def __init__(self, reason: SummaryPlanRefusalReason) -> None:
        self.reason = reason


def build_summary_section_plan(
    evidence: Iterable[Any] | Mapping[str, Any] | None = None,
    *,
    approved_evidence: Iterable[Any] | Mapping[str, Any] | None = None,
    sections: Iterable[Any] | Mapping[str, Any] | None = None,
) -> SummarySectionPlan:
    """Build a deterministic section-preserving plan from approved evidence.

    Args:
        evidence: An iterable of evidence mappings or :class:`SummaryEvidence`
            records. Each selected record must carry an explicit stable
            ``section_id`` (``source_section_id`` is also accepted). Optional
            ``text``, ``value``, or other source-bearing fields are ignored and
            never copied. Records with an explicit ``approved=False`` (or an
            equivalent rejected review status) are excluded. If approval is
            omitted, the caller is treated as having supplied an already
            approved evidence collection.
        approved_evidence: Explicit keyword alias for ``evidence``. It is
            useful when the upstream review gate already exposes a named
            approved collection. Supplying both aliases is refused.
        sections: Optional detected section metadata. It is used only to order
            groups by source offsets and to retain value-free section offsets;
            section definitions never replace a missing evidence ``section_id``.

    Returns:
        A ready :class:`SummarySectionPlan`, or a refused plan carrying a typed
        :class:`SummaryPlanRefusalReason`. Refusal is all-or-nothing so a local
        generator cannot accidentally mix section-scoped and unscoped facts.

    The function is local-only and deterministic. It does not load a model,
    call a network service, log source content, or persist evidence values.
    """

    if approved_evidence is not None:
        if evidence is not None:
            return _refused_plan(
                SummaryPlanRefusalReason.INVALID_EVIDENCE,
                input_count=0,
            )
        evidence = approved_evidence

    try:
        rows = _materialize_evidence(evidence)
    except _PlanInputError as error:
        return _refused_plan(error.reason, input_count=0)

    input_count = len(rows)
    if input_count > MAX_SUMMARY_PLAN_EVIDENCE:
        return _refused_plan(
            SummaryPlanRefusalReason.EVIDENCE_LIMIT,
            input_count=input_count,
            rejected_count=input_count - MAX_SUMMARY_PLAN_EVIDENCE,
        )

    section_metadata, section_error = _coerce_section_metadata(sections)
    if section_error is not None:
        return _refused_plan(
            section_error,
            input_count=input_count,
            rejected_count=1,
        )

    selected: list[SummaryEvidence] = []
    excluded_count = 0
    for row in rows:
        try:
            normalized = _coerce_evidence(row)
        except _PlanInputError as error:
            return _refused_plan(
                error.reason,
                input_count=input_count,
                excluded_count=excluded_count,
                rejected_count=1,
            )
        if normalized is None:
            excluded_count += 1
            continue
        selected.append(normalized)

    if section_metadata:
        for item in selected:
            if item.section_id not in section_metadata:
                return _refused_plan(
                    SummaryPlanRefusalReason.UNKNOWN_SECTION_ID,
                    input_count=input_count,
                    excluded_count=excluded_count,
                    rejected_count=1,
                )

    grouped: dict[str, list[SummaryEvidence]] = {}
    for item in selected:
        grouped.setdefault(item.section_id, []).append(item)

    groups: list[SummarySectionGroup] = []
    for section_id, members in grouped.items():
        metadata = section_metadata.get(section_id)
        groups.append(
            SummarySectionGroup(
                section_id=section_id,
                evidence=tuple(members),
                section_start=metadata.start if metadata else None,
                section_end=metadata.end if metadata else None,
            )
        )
    groups.sort(key=lambda group: _section_key(group, section_metadata))

    return SummarySectionPlan(
        sections=tuple(groups),
        input_evidence_count=input_count,
        approved_evidence_count=len(selected),
        excluded_evidence_count=excluded_count,
    )


def plan_summary_sections(
    evidence: Iterable[Any] | Mapping[str, Any] | None = None,
    *,
    approved_evidence: Iterable[Any] | Mapping[str, Any] | None = None,
    sections: Iterable[Any] | Mapping[str, Any] | None = None,
) -> SummarySectionPlan:
    """Alias for :func:`build_summary_section_plan`."""

    return build_summary_section_plan(
        evidence,
        approved_evidence=approved_evidence,
        sections=sections,
    )


def build_section_preserving_summary_plan(
    evidence: Iterable[Any] | Mapping[str, Any] | None = None,
    *,
    approved_evidence: Iterable[Any] | Mapping[str, Any] | None = None,
    sections: Iterable[Any] | Mapping[str, Any] | None = None,
) -> SummarySectionPlan:
    """Build a plan under the full section-preserving feature name."""

    return build_summary_section_plan(
        evidence,
        approved_evidence=approved_evidence,
        sections=sections,
    )


def require_summary_section_plan(
    evidence: Iterable[Any] | Mapping[str, Any] | None = None,
    *,
    approved_evidence: Iterable[Any] | Mapping[str, Any] | None = None,
    sections: Iterable[Any] | Mapping[str, Any] | None = None,
) -> SummarySectionPlan:
    """Build a plan and raise a fixed-code error when it is refused."""

    return build_summary_section_plan(
        evidence,
        approved_evidence=approved_evidence,
        sections=sections,
    ).require_ready()


def _refused_plan(
    reason: SummaryPlanRefusalReason,
    *,
    input_count: int,
    excluded_count: int = 0,
    rejected_count: int = 1,
) -> SummarySectionPlan:
    return SummarySectionPlan(
        status=SUMMARY_PLAN_STATUS_REFUSED,
        refusal=SummaryPlanRefusal(reason, rejected_count=rejected_count),
        input_evidence_count=_safe_count(
            input_count, "input_evidence_count", allow_zero=True
        ),
        # A refused plan has no generation groups, so selected evidence is not
        # exposed as an accepted count even when earlier rows were parseable.
        approved_evidence_count=0,
        excluded_evidence_count=_safe_count(
            excluded_count,
            "excluded_evidence_count",
            allow_zero=True,
        ),
    )


def _materialize_evidence(
    evidence: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[Any, ...]:
    if evidence is None:
        return ()
    if isinstance(evidence, Mapping):
        # A mapping containing an evidence field is one record. A plain
        # id-to-record mapping is accepted as a convenience for local callers.
        if _has_any_key(
            evidence,
            {
                "section_id",
                "source_section_id",
                "section_identifier",
                "evidence_id",
                "start",
                "source_start",
            },
        ):
            return (evidence,)
        for container_name in (
            "approved_evidence",
            "evidence",
            "records",
            "items",
        ):
            nested = _first_field(evidence, (container_name,))
            if nested is not _MISSING:
                return _materialize_evidence(
                    cast(Iterable[Any] | Mapping[str, Any] | None, nested)
                )
        try:
            return tuple(evidence.values())
        except Exception:
            raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE) from None
    if isinstance(evidence, (str, bytes, bytearray)):
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE)
    try:
        return tuple(evidence)
    except Exception:
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE) from None


def _coerce_evidence(raw: Any) -> SummaryEvidence | None:
    if isinstance(raw, SummaryEvidence):
        if not raw.approved:
            return None
        return raw
    data = _mapping_view(raw)
    if data is None:
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE)

    approval = _approval_state(data)
    if approval is _UNAPPROVED:
        return None
    if approval is None:
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_APPROVAL)

    section_id = _section_identifier(data)
    if section_id is _MISSING:
        raise _PlanInputError(SummaryPlanRefusalReason.MISSING_SECTION_ID)
    section_id = _validated_identifier(
        section_id,
        SummaryPlanRefusalReason.INVALID_SECTION_ID,
    )

    source_start, source_end = _source_offset_from_mapping(data)
    evidence_id = _first_field(
        data,
        ("evidence_id", "citation_id", "entity_id", "span_id", "id"),
    )
    if evidence_id is _MISSING:
        if source_start is None or source_end is None:
            raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE_ID)
        evidence_id = f"offset-{source_start}-{source_end}"
    evidence_id = _validated_identifier(
        evidence_id,
        SummaryPlanRefusalReason.INVALID_EVIDENCE_ID,
    )
    try:
        return SummaryEvidence(
            evidence_id=evidence_id,
            section_id=section_id,
            source_start=source_start,
            source_end=source_end,
            approved=True,
        )
    except _PlanInputError:
        raise
    except Exception:
        # Do not let a custom mapping/object or a validation message echo its
        # submitted value through the planner boundary.
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE) from None


def _mapping_view(raw: Any) -> Mapping[str, Any] | None:
    if isinstance(raw, Mapping):
        return raw
    known_names = (
        "evidence_id",
        "citation_id",
        "entity_id",
        "span_id",
        "id",
        "section_id",
        "source_section_id",
        "section_identifier",
        "section",
        "start",
        "end",
        "source_start",
        "source_end",
        "source_offset",
        "source_span",
        "offset",
        "span",
        "approved",
        "review_status",
        "approval_status",
        "status",
    )
    values: dict[str, Any] = {}
    try:
        for name in known_names:
            value = getattr(raw, name, _MISSING)
            if value is not _MISSING:
                values[name] = value
    except Exception:
        return None
    return values or None


def _section_identifier(data: Mapping[str, Any]) -> object:
    value = _first_field(
        data,
        ("section_id", "source_section_id", "section_identifier", "sectionId"),
    )
    if value is not _MISSING:
        return value

    nested = _first_field(data, ("section", "source_section"))
    if isinstance(nested, Mapping):
        nested_id = _first_field(
            nested,
            ("section_id", "source_section_id", "section_identifier", "id"),
        )
        if nested_id is not _MISSING:
            return nested_id
        return _MISSING
    # A plain section label is not a stable identifier. The planner never
    # derives an identifier from a label, source text, or source offsets.
    return _MISSING


def _approval_state(data: Mapping[str, Any]) -> bool | object | None:
    approved = _first_field(data, ("approved", "is_approved"))
    if approved is not _MISSING:
        if type(approved) is not bool:
            return None
        return approved if approved else _UNAPPROVED

    approval = _first_field(data, ("approval",))
    if isinstance(approval, Mapping):
        nested = _first_field(approval, ("approved", "is_approved"))
        if nested is not _MISSING:
            if type(nested) is not bool:
                return None
            return nested if nested else _UNAPPROVED
        approval = _first_field(approval, ("status", "state"))
    if approval is not _MISSING:
        return _approval_status_value(approval)

    status = _first_field(data, ("review_status", "approval_status", "status"))
    if status is not _MISSING:
        return _approval_status_value(status)
    # The function's input contract is an approved-evidence boundary. Missing
    # approval metadata therefore means "already approved", while explicit
    # rejection is always excluded.
    return True


def _approval_status_value(value: Any) -> bool | object | None:
    if not isinstance(value, str):
        return None
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
    return None


def _source_offset_from_mapping(
    data: Mapping[str, Any],
) -> tuple[int | None, int | None]:
    nested = _first_field(data, ("source_offset", "source_offsets", "source_span"))
    if nested is _MISSING:
        nested = _first_field(data, ("offset", "span"))
    if nested is not _MISSING:
        if isinstance(nested, Mapping):
            start = _first_field(nested, ("start", "source_start"))
            end = _first_field(nested, ("end", "source_end"))
        elif isinstance(nested, (tuple, list)) and len(nested) == 2:
            start, end = nested
        else:
            raise _PlanInputError(SummaryPlanRefusalReason.INVALID_SOURCE_OFFSET)
    else:
        start = _first_field(data, ("source_start", "start"))
        end = _first_field(data, ("source_end", "end"))

    if start is _MISSING and end is _MISSING:
        return None, None
    if start is _MISSING or end is _MISSING:
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_SOURCE_OFFSET)
    return _validated_offsets(start, end)


def _coerce_section_metadata(
    sections: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[dict[str, _SectionMetadata], SummaryPlanRefusalReason | None]:
    if sections is None:
        return {}, None
    try:
        rows = _materialize_sections(sections)
    except _PlanInputError as error:
        return {}, error.reason

    result: dict[str, _SectionMetadata] = {}
    for raw in rows:
        try:
            data = _mapping_view(raw)
            if data is None:
                return {}, SummaryPlanRefusalReason.INVALID_EVIDENCE
            section_id = _first_field(
                data,
                ("section_id", "source_section_id", "section_identifier", "id"),
            )
            if section_id is _MISSING:
                return {}, SummaryPlanRefusalReason.MISSING_SECTION_ID
            section_id = _validated_identifier(
                section_id,
                SummaryPlanRefusalReason.INVALID_SECTION_ID,
            )
            start, end = _source_offset_from_mapping(data)
        except _PlanInputError as error:
            return {}, error.reason
        metadata = _SectionMetadata(section_id, start, end)
        existing = result.get(section_id)
        if existing is not None and existing != metadata:
            return {}, SummaryPlanRefusalReason.CONFLICTING_SECTION_METADATA
        result[section_id] = metadata
    return result, None


def _materialize_sections(
    sections: Iterable[Any] | Mapping[str, Any],
) -> tuple[Any, ...]:
    if isinstance(sections, Mapping):
        nested = _first_field(sections, ("sections", "clinical_sections"))
        if nested is not _MISSING:
            return _materialize_sections(
                cast(Iterable[Any] | Mapping[str, Any], nested)
            )
        if _has_any_key(
            sections,
            {"section_id", "source_section_id", "section_identifier", "id"},
        ):
            return (sections,)
        try:
            return tuple(sections.values())
        except Exception:
            raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE) from None
    if isinstance(sections, (str, bytes, bytearray)):
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE)
    try:
        return tuple(sections)
    except Exception:
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE) from None


def _first_field(data: Mapping[str, Any], names: Iterable[str]) -> object:
    for name in names:
        try:
            if name in data:
                return data[name]
        except Exception:
            raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE) from None
    return _MISSING


def _has_any_key(data: Mapping[str, Any], names: set[str]) -> bool:
    try:
        return any(name in data for name in names)
    except Exception:
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_EVIDENCE) from None


def _validated_identifier(
    value: object,
    reason: SummaryPlanRefusalReason,
) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_SUMMARY_PLAN_ID_LENGTH
        or _ID_RE.fullmatch(value) is None
    ):
        raise _PlanInputError(reason)
    return value


def _stable_identifier(
    value: object,
    reason: SummaryPlanRefusalReason,
) -> str:
    try:
        return _validated_identifier(value, reason)
    except _PlanInputError as error:
        raise ValueError(f"invalid summary section plan {error.reason.value}") from None


def _validated_offsets(start: object, end: object) -> tuple[int, int]:
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or start > _MAX_SOURCE_OFFSET
        or end > _MAX_SOURCE_OFFSET
    ):
        raise _PlanInputError(SummaryPlanRefusalReason.INVALID_SOURCE_OFFSET)
    return start, end


def _optional_source_offsets(
    start: object,
    end: object,
) -> tuple[int | None, int | None]:
    if start is None and end is None:
        return None, None
    if start is None or end is None:
        raise ValueError("summary source offsets are invalid")
    try:
        return _validated_offsets(start, end)
    except _PlanInputError as error:
        raise ValueError(
            f"summary source offsets are invalid: {error.reason.value}"
        ) from None


def _evidence_key(item: SummaryEvidence) -> tuple[Any, ...]:
    return (
        item.source_start if item.source_start is not None else _MAX_SOURCE_OFFSET,
        item.source_end if item.source_end is not None else _MAX_SOURCE_OFFSET,
        item.evidence_id.casefold(),
        item.evidence_id,
    )


def _section_key(
    group: SummarySectionGroup,
    metadata: Mapping[str, _SectionMetadata] | None = None,
) -> tuple[Any, ...]:
    source = metadata.get(group.section_id) if metadata is not None else None
    start = (
        group.section_start
        if group.section_start is not None
        else source.start
        if source is not None and source.start is not None
        else _first_evidence_start(group)
    )
    end = (
        group.section_end
        if group.section_end is not None
        else source.end
        if source is not None and source.end is not None
        else _MAX_SOURCE_OFFSET
    )
    return (
        start if start is not None else _MAX_SOURCE_OFFSET,
        end,
        group.section_id.casefold(),
        group.section_id,
    )


def _first_evidence_start(group: SummarySectionGroup) -> int | None:
    starts = [
        item.source_start for item in group.evidence if item.source_start is not None
    ]
    return min(starts) if starts else None


def _coerce_refusal_reason(value: object) -> SummaryPlanRefusalReason:
    if isinstance(value, SummaryPlanRefusalReason):
        return value
    try:
        return SummaryPlanRefusalReason(value)
    except Exception:
        raise ValueError("invalid summary section plan refusal reason") from None


def _safe_count(value: object, field_name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{field_name} is invalid")
    return value


__all__ = [
    "MAX_SUMMARY_PLAN_EVIDENCE",
    "MAX_SUMMARY_PLAN_ID_LENGTH",
    "SUMMARY_PLAN_STATUS_READY",
    "SUMMARY_PLAN_STATUS_REFUSED",
    "SUMMARY_SECTION_PLAN_DISCLAIMER",
    "SUMMARY_SECTION_PLAN_SCHEMA_VERSION",
    "SectionEvidenceGroup",
    "SummaryEvidence",
    "SummaryEvidenceReference",
    "SummaryPlanRefusal",
    "SummaryPlanRefusalReason",
    "SummaryPlanStatus",
    "SummarySection",
    "SummarySectionEvidence",
    "SummarySectionGroup",
    "SummarySectionPlan",
    "SummarySectionPlanError",
    "build_section_preserving_summary_plan",
    "build_summary_section_plan",
    "plan_summary_sections",
    "require_summary_section_plan",
]
