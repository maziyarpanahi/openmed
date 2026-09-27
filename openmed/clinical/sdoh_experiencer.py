"""Deterministic experiencer classification and filtering for SDOH evidence.

Social-history notes often combine patient, household, and family statements in
one local context.  This module assigns each candidate SDOH span to a small,
controlled experiencer vocabulary and keeps only confirmed patient evidence in
the patient-level view.  Non-patient and unresolved evidence remains available
as offset-only review metadata.

Cue matching is local, deterministic, and offline.  The source text and any
candidate values are used transiently and are never copied into a returned
record, serialized report, or exception message.  The output is an assistive
review aid, not a clinical determination.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

from .context import canonical_section_name
from .sections import detect_sections

PATIENT: Final[str] = "patient"
HOUSEHOLD: Final[str] = "household"
FAMILY: Final[str] = "family"
UNKNOWN: Final[str] = "unknown"

# Compatibility names make the SDOH-specific vocabulary explicit without
# changing the narrower patient/family constants used by the general context
# layer.
PATIENT_EXPERIENCER: Final[str] = PATIENT
HOUSEHOLD_EXPERIENCER: Final[str] = HOUSEHOLD
FAMILY_EXPERIENCER: Final[str] = FAMILY
UNKNOWN_EXPERIENCER: Final[str] = UNKNOWN

EXPERIENCER_CLASSES: Final[tuple[str, ...]] = (
    PATIENT,
    HOUSEHOLD,
    FAMILY,
    UNKNOWN,
)
SDOH_EXPERIENCER_VALUES: Final[tuple[str, ...]] = EXPERIENCER_CLASSES
SDOH_EXPERIENCER_SCHEMA_VERSION: Final[int] = 1

SDOH_EXPERIENCER_ADVISORY = (
    "SDOH experiencer filtering is a deterministic, assistive review aid. "
    "Only evidence attributed to the patient is eligible for patient-level "
    "output; verify all classifications with qualified clinical review."
)
SDOH_EXPERIENCER_FILTER_ADVISORY = SDOH_EXPERIENCER_ADVISORY

SpanOffset = tuple[int, int]
Experiencer = Literal["patient", "household", "family", "unknown"]
ClassificationSource = Literal["cue", "section", "default", "provided"]

_MAX_CUE_DISTANCE = 128
_SENTENCE_BOUNDARIES = ".!?;\n"
_CONTRASTIVE_BOUNDARY_RE = re.compile(
    r"(?<!\w)(?:but|however|although|whereas|yet)(?!\w)",
    re.IGNORECASE,
)
_REPORTING_VERB_RE = re.compile(
    r"(?<!\w)(?:reports?|denies?|states?|says?|notes?|describes?)(?!\w)",
    re.IGNORECASE,
)
_AFTER_CUE_RELATION_RE = re.compile(
    r"(?<!\w)(?:by|of|from|for|in|with|among|within|reported\s+by|"
    r"according\s+to|belongs\s+to|associated\s+with|involving)(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _CuePattern:
    """One compiled local subject cue."""

    experiencer: str
    pattern: re.Pattern[str]


def _literal_pattern(cues: Sequence[str]) -> re.Pattern[str]:
    alternatives = []
    for cue in sorted(cues, key=lambda value: (-len(value), value.casefold())):
        normalized = " ".join(cue.split())
        escaped = r"\s+".join(re.escape(part) for part in normalized.split())
        prefix = r"(?<!\w)" if normalized and normalized[0].isalnum() else ""
        suffix = r"(?!\w)" if normalized and normalized[-1].isalnum() else ""
        alternatives.append(f"{prefix}(?:{escaped}){suffix}")
    return re.compile("|".join(alternatives), re.IGNORECASE)


_CUE_PATTERNS: tuple[_CuePattern, ...] = (
    _CuePattern(
        PATIENT,
        re.compile(
            r"(?<!\w)(?:patient(?:['’]s)?|"
            r"(?:pt|self|my|i|himself|herself|themself|themselves))(?!\w)",
            re.IGNORECASE,
        ),
    ),
    _CuePattern(
        HOUSEHOLD,
        _literal_pattern(
            (
                "household",
                "household member",
                "roommate",
                "housemate",
                "cohabitant",
                "co-resident",
                "coresident",
                "caregiver",
                "partner",
                "boyfriend",
                "girlfriend",
                "living with",
                "lives with",
                "live with",
                "cohabits with",
            )
        ),
    ),
    _CuePattern(
        FAMILY,
        _literal_pattern(
            (
                "family history",
                "family member",
                "family",
                "relative",
                "mother",
                "mom",
                "mum",
                "father",
                "dad",
                "parent",
                "parents",
                "sibling",
                "siblings",
                "brother",
                "sister",
                "grandmother",
                "grandfather",
                "grandparent",
                "grandparents",
                "aunt",
                "uncle",
                "cousin",
                "niece",
                "nephew",
                "son",
                "daughter",
                "child",
                "children",
                "wife",
                "husband",
                "spouse",
                "maternal",
                "paternal",
            )
        ),
    ),
    _CuePattern(
        UNKNOWN,
        _literal_pattern(
            (
                "unknown",
                "unclear",
                "uncertain",
                "unspecified",
                "undocumented",
                "not specified",
                "not known",
                "not documented",
                "unable to determine",
                "unable to identify",
                "cannot determine",
                "cannot identify",
                "no person specified",
                "no experiencer specified",
                "friend",
                "neighbor",
                "neighbour",
                "coworker",
                "colleague",
                "donor",
                "contact",
                "other",
            )
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class _CueHit:
    experiencer: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SDOHExperiencerEvidence:
    """Value-free experiencer metadata for one SDOH candidate.

    Args:
        source_offsets: Half-open offsets of the candidate in the source
            document.
        experiencer: One of ``patient``, ``household``, ``family`` or
            ``unknown``.
        cue_offsets: Half-open offsets of local cues used for classification.
            Cue text is intentionally not retained.
        conflicting_experiencers: Distinct classes observed in one unresolved
            local scope.  Conflicts always produce ``unknown`` and require
            review.
        source: Whether the classification came from a local cue, section
            prior, caller-provided metadata, or the default.
        review_required: Whether a human must review the classification.
        input_index: Stable position of the candidate in the caller's input.
    """

    source_offsets: SpanOffset
    experiencer: str
    cue_offsets: tuple[SpanOffset, ...] = ()
    conflicting_experiencers: tuple[str, ...] = ()
    source: ClassificationSource = "default"
    review_required: bool = False
    input_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_offsets",
            _validate_offset(self.source_offsets, "source offsets"),
        )
        if self.experiencer not in EXPERIENCER_CLASSES:
            raise ValueError("unsupported SDOH experiencer class")
        if self.source not in {"cue", "section", "default", "provided"}:
            raise ValueError("unsupported SDOH experiencer source")

        cue_offsets = tuple(
            _validate_offset(offset, "cue offsets") for offset in self.cue_offsets
        )
        object.__setattr__(self, "cue_offsets", tuple(sorted(set(cue_offsets))))

        conflicts = set(self.conflicting_experiencers)
        if any(value not in EXPERIENCER_CLASSES for value in conflicts):
            raise ValueError("unsupported conflicting SDOH experiencer class")
        if self.experiencer in conflicts:
            raise ValueError("conflicting classes must differ from the result")
        ordered_conflicts = tuple(
            value for value in EXPERIENCER_CLASSES if value in conflicts
        )
        object.__setattr__(self, "conflicting_experiencers", ordered_conflicts)

        if type(self.review_required) is not bool:
            raise TypeError("review_required must be a boolean")
        if self.experiencer == UNKNOWN and not self.review_required:
            raise ValueError("unknown SDOH experiencers require human review")
        if ordered_conflicts and (
            self.experiencer != UNKNOWN or not self.review_required
        ):
            raise ValueError("conflicting SDOH experiencers require human review")
        if self.input_index is not None:
            if (
                isinstance(self.input_index, bool)
                or not isinstance(self.input_index, int)
                or self.input_index < 0
            ):
                raise ValueError("input index must be a non-negative integer")

    @property
    def source_span(self) -> SpanOffset:
        """Return the candidate's source offsets."""

        return self.source_offsets

    @property
    def source_offset(self) -> SpanOffset:
        """Return the candidate's source offsets."""

        return self.source_offsets

    @property
    def classification(self) -> str:
        """Return the controlled experiencer class."""

        return self.experiencer

    @property
    def classification_source(self) -> str:
        """Return how the experiencer classification was decided."""

        return self.source

    @property
    def cue_offset(self) -> SpanOffset | None:
        """Return the sole cue offset when there is exactly one cue."""

        return self.cue_offsets[0] if len(self.cue_offsets) == 1 else None

    @property
    def patient_record_eligible(self) -> bool:
        """Return whether the candidate may enter patient-level output."""

        return self.experiencer == PATIENT and not self.review_required

    @property
    def included_in_patient_output(self) -> bool:
        """Alias for :attr:`patient_record_eligible`."""

        return self.patient_record_eligible

    @property
    def has_conflict(self) -> bool:
        """Return whether local cues conflicted."""

        return bool(self.conflicting_experiencers)

    @property
    def exclusion_reason(self) -> str | None:
        """Return the value-free patient-level exclusion reason."""

        if self.patient_record_eligible:
            return None
        if self.experiencer == UNKNOWN:
            return "unknown experiencer"
        return "non-patient experiencer"

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata without source or finding values."""

        return {
            "schema_version": SDOH_EXPERIENCER_SCHEMA_VERSION,
            "source_offsets": {
                "start": self.source_offsets[0],
                "end": self.source_offsets[1],
            },
            "experiencer": self.experiencer,
            "cue_offsets": [list(offset) for offset in self.cue_offsets],
            "conflicting_experiencers": list(self.conflicting_experiencers),
            "source": self.source,
            "patient_record_eligible": self.patient_record_eligible,
            "exclusion_reason": self.exclusion_reason,
            "review_required": self.review_required,
            "input_index": self.input_index,
        }

    def to_json(self) -> str:
        """Return compact deterministic JSON containing only safe metadata."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SDOHExperiencerEvidence":
        """Rebuild a value-free record from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise TypeError("SDOH experiencer evidence payload must be a mapping")
        if payload.get("schema_version") != SDOH_EXPERIENCER_SCHEMA_VERSION:
            raise ValueError("unsupported SDOH experiencer schema version")
        source_payload = payload.get("source_offsets")
        if not isinstance(source_payload, Mapping):
            raise TypeError("SDOH experiencer source offsets are required")

        cue_offsets = payload.get("cue_offsets", ())
        conflicts = payload.get("conflicting_experiencers", ())
        if not _is_sequence(cue_offsets) or not _is_sequence(conflicts):
            raise TypeError("SDOH experiencer offset or conflict fields are invalid")
        if any(not isinstance(value, str) for value in conflicts):
            raise TypeError("SDOH experiencer conflicts are invalid")

        experiencer = payload.get("experiencer")
        source = payload.get("source")
        review_required = payload.get("review_required")
        if not isinstance(experiencer, str) or not isinstance(source, str):
            raise TypeError("SDOH experiencer class or source is invalid")
        if type(review_required) is not bool:
            raise TypeError("SDOH experiencer review flag is invalid")

        input_index = payload.get("input_index")
        if input_index is not None and not isinstance(input_index, int):
            raise TypeError("SDOH experiencer input index is invalid")
        source_offsets = _validate_offset(
            (source_payload.get("start"), source_payload.get("end")),
            "SDOH experiencer source offsets",
        )
        return cls(
            source_offsets=source_offsets,
            experiencer=experiencer,
            cue_offsets=tuple(cue_offsets),
            conflicting_experiencers=tuple(conflicts),
            source=cast(ClassificationSource, source),
            review_required=review_required,
            input_index=input_index,
        )


# Descriptive aliases keep the type discoverable under both evidence and
# classification terminology used by SDOH callers.
SDOHExperiencerClassification = SDOHExperiencerEvidence
ExperiencerClassification = SDOHExperiencerEvidence


@dataclass(frozen=True, slots=True)
class SDOHExperiencerFilterResult:
    """Partitioned, value-free SDOH experiencer evidence.

    ``patient_evidence`` is the only patient-level view.  ``excluded_evidence``
    retains every other candidate with its source and cue offsets so a human
    reviewer can inspect the original document in a separately authorized
    workflow.  The class is iterable as ``(patient_evidence, excluded_evidence)``
    for callers that prefer tuple unpacking.
    """

    all_evidence: tuple[SDOHExperiencerEvidence, ...]
    patient_evidence: tuple[SDOHExperiencerEvidence, ...]
    excluded_evidence: tuple[SDOHExperiencerEvidence, ...]

    def __post_init__(self) -> None:
        all_evidence = tuple(self.all_evidence)
        patient_evidence = tuple(self.patient_evidence)
        excluded_evidence = tuple(self.excluded_evidence)
        if any(not isinstance(item, SDOHExperiencerEvidence) for item in all_evidence):
            raise TypeError("all SDOH evidence must use SDOHExperiencerEvidence")
        if any(
            not isinstance(item, SDOHExperiencerEvidence)
            for item in patient_evidence + excluded_evidence
        ):
            raise TypeError("filtered SDOH evidence must use SDOHExperiencerEvidence")
        if any(not item.patient_record_eligible for item in patient_evidence):
            raise ValueError("patient evidence must be patient-record eligible")
        if set(patient_evidence).intersection(excluded_evidence):
            raise ValueError("patient and excluded evidence must be disjoint")
        if patient_evidence + excluded_evidence != all_evidence:
            raise ValueError("filtered evidence must partition all evidence")
        object.__setattr__(self, "all_evidence", all_evidence)
        object.__setattr__(self, "patient_evidence", patient_evidence)
        object.__setattr__(self, "excluded_evidence", excluded_evidence)

    @property
    def patient(self) -> tuple[SDOHExperiencerEvidence, ...]:
        """Return the patient-level evidence."""

        return self.patient_evidence

    @property
    def included(self) -> tuple[SDOHExperiencerEvidence, ...]:
        """Return patient-level evidence using filter terminology."""

        return self.patient_evidence

    @property
    def patient_level(self) -> tuple[SDOHExperiencerEvidence, ...]:
        """Return the patient-level evidence."""

        return self.patient_evidence

    @property
    def excluded(self) -> tuple[SDOHExperiencerEvidence, ...]:
        """Return reviewable evidence excluded from patient-level output."""

        return self.excluded_evidence

    @property
    def review_queue(self) -> tuple[SDOHExperiencerEvidence, ...]:
        """Return evidence requiring human review in stable input order."""

        return tuple(item for item in self.all_evidence if item.review_required)

    def __iter__(self):
        yield self.patient_evidence
        yield self.excluded_evidence

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, value-free filter report."""

        return {
            "schema_version": SDOH_EXPERIENCER_SCHEMA_VERSION,
            "all_evidence": [item.to_dict() for item in self.all_evidence],
            "patient_evidence": [item.to_dict() for item in self.patient_evidence],
            "excluded_evidence": [item.to_dict() for item in self.excluded_evidence],
        }

    def to_json(self) -> str:
        """Return compact deterministic JSON containing only safe metadata."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SDOHExperiencerFilterResult":
        """Rebuild a value-free filter report from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise TypeError("SDOH experiencer filter payload must be a mapping")
        if payload.get("schema_version") != SDOH_EXPERIENCER_SCHEMA_VERSION:
            raise ValueError("unsupported SDOH experiencer schema version")
        fields = ("all_evidence", "patient_evidence", "excluded_evidence")
        values = tuple(
            _require_sequence(payload.get(field), "SDOH experiencer filter evidence")
            for field in fields
        )
        return cls(
            *(
                tuple(SDOHExperiencerEvidence.from_dict(item) for item in value)
                for value in values
            )
        )


def classify_sdoh_experiencers(
    text: str,
    evidence: Iterable[Any] | Any,
    *,
    sections: Iterable[Any] | Mapping[str, Any] | str | None = None,
    section_experiencer: str | None = None,
    default_experiencer: str = UNKNOWN,
) -> list[SDOHExperiencerEvidence]:
    """Classify candidate SDOH spans by their local experiencer.

    Args:
        text: Source document text. It is used transiently for local cue
            matching and is never copied into returned records.
        evidence: One candidate or an iterable of candidates exposing
            ``start``/``end``, ``span``, or ``source_offsets``. Existing
            ``SDOHFinding`` instances are accepted through their ``span``.
        sections: Optional section spans. A Social History section supplies a
            patient prior and a Family History section supplies a family prior.
        section_experiencer: Optional explicit prior used when no cue governs a
            candidate. It must be one of the controlled classes.
        default_experiencer: Fallback when no cue or section prior is present;
            the conservative default is ``unknown``.

    Returns:
        Value-free records sorted by source offsets and then input order.
        Unknown or conflicting classifications require human review.
    """

    if not isinstance(text, str):
        raise TypeError("SDOH experiencer source text must be a string")
    default = _normalize_experiencer(default_experiencer)
    section_default = (
        None
        if section_experiencer is None
        else _normalize_experiencer(section_experiencer)
    )
    detected_sections = detect_sections(text) if sections is None else sections
    section_ranges = _coerce_sections(detected_sections, len(text))
    items = _evidence_items(evidence)

    records: list[tuple[int, SDOHExperiencerEvidence]] = []
    for index, item in enumerate(items):
        offsets = _offset_from_item(item, len(text))
        records.append(
            (
                index,
                _classify_one(
                    text,
                    item,
                    offsets,
                    index=index,
                    section_ranges=section_ranges,
                    section_default=section_default,
                    default=default,
                ),
            )
        )
    records.sort(key=lambda item: (item[1].source_offsets, item[0]))
    return [record for _, record in records]


def classify_sdoh_experiencer(
    text: str,
    evidence: Any,
    **kwargs: Any,
) -> SDOHExperiencerEvidence:
    """Classify exactly one SDOH candidate and return its metadata."""

    records = classify_sdoh_experiencers(text, evidence, **kwargs)
    if len(records) != 1:
        raise ValueError("single-candidate classification requires exactly one item")
    return records[0]


def classify_sdoh_evidence(
    text: str,
    evidence: Iterable[Any] | Any,
    **kwargs: Any,
) -> list[SDOHExperiencerEvidence]:
    """Alias for :func:`classify_sdoh_experiencers`."""

    return classify_sdoh_experiencers(text, evidence, **kwargs)


def attach_sdoh_experiencer(
    text: str,
    evidence: Iterable[Any] | Any,
    **kwargs: Any,
) -> list[SDOHExperiencerEvidence]:
    """Attach value-free experiencer metadata to SDOH evidence spans."""

    return classify_sdoh_experiencers(text, evidence, **kwargs)


def filter_sdoh_findings(
    text: str,
    findings: Iterable[Any] | Any,
    **kwargs: Any,
) -> SDOHExperiencerFilterResult:
    """Classify and partition SDOH candidates into patient and excluded views.

    The returned patient and excluded collections contain only
    :class:`SDOHExperiencerEvidence` metadata.  Use each record's
    ``input_index`` to join the patient-level view with caller-owned findings
    inside the authorized process; do not persist the original values in this
    report.
    """

    all_evidence = tuple(classify_sdoh_experiencers(text, findings, **kwargs))
    patient_evidence = tuple(
        item for item in all_evidence if item.patient_record_eligible
    )
    excluded_evidence = tuple(
        item for item in all_evidence if not item.patient_record_eligible
    )
    return SDOHExperiencerFilterResult(
        all_evidence=all_evidence,
        patient_evidence=patient_evidence,
        excluded_evidence=excluded_evidence,
    )


def filter_sdoh_experiencers(
    text: str,
    findings: Iterable[Any] | Any,
    **kwargs: Any,
) -> SDOHExperiencerFilterResult:
    """Alias for :func:`filter_sdoh_findings`."""

    return filter_sdoh_findings(text, findings, **kwargs)


def filter_patient_sdoh(
    text: str,
    findings: Iterable[Any] | Any,
    **kwargs: Any,
) -> SDOHExperiencerFilterResult:
    """Return the patient-level and reviewable SDOH experiencer partition."""

    return filter_sdoh_findings(text, findings, **kwargs)


def filter_sdoh_evidence(
    evidence: Iterable[SDOHExperiencerEvidence] | Any,
) -> SDOHExperiencerFilterResult:
    """Partition already-classified, value-free SDOH evidence."""

    items = _classification_items(evidence)
    patient_evidence = tuple(item for item in items if item.patient_record_eligible)
    excluded_evidence = tuple(
        item for item in items if not item.patient_record_eligible
    )
    return SDOHExperiencerFilterResult(
        all_evidence=items,
        patient_evidence=patient_evidence,
        excluded_evidence=excluded_evidence,
    )


def _classify_one(
    text: str,
    item: Any,
    offsets: SpanOffset,
    *,
    index: int,
    section_ranges: Sequence[tuple[int, int, str | None]],
    section_default: str | None,
    default: str,
) -> SDOHExperiencerEvidence:
    start, end = offsets
    hits = _cue_hits(text, start, end)
    observed = {hit.experiencer for hit in hits}
    provided = _provided_experiencer(item)

    if provided is not None:
        observed.add(provided)
        if len(observed) > 1:
            result = UNKNOWN
            conflicts = tuple(
                value for value in EXPERIENCER_CLASSES if value in observed
            )
            source = cast(ClassificationSource, "cue" if hits else "provided")
            cue_offsets = tuple((hit.start, hit.end) for hit in hits)
            return SDOHExperiencerEvidence(
                source_offsets=offsets,
                experiencer=result,
                cue_offsets=cue_offsets,
                conflicting_experiencers=conflicts,
                source=source,
                review_required=True,
                input_index=index,
            )
        return SDOHExperiencerEvidence(
            source_offsets=offsets,
            experiencer=provided,
            cue_offsets=tuple(
                (hit.start, hit.end) for hit in hits if hit.experiencer == provided
            ),
            source="provided",
            review_required=provided == UNKNOWN,
            input_index=index,
        )

    if len(observed) > 1:
        conflicts = tuple(value for value in EXPERIENCER_CLASSES if value in observed)
        return SDOHExperiencerEvidence(
            source_offsets=offsets,
            experiencer=UNKNOWN,
            cue_offsets=tuple((hit.start, hit.end) for hit in hits),
            conflicting_experiencers=conflicts,
            source="cue",
            review_required=True,
            input_index=index,
        )

    if hits:
        experiencer = next(iter(observed))
        return SDOHExperiencerEvidence(
            source_offsets=offsets,
            experiencer=experiencer,
            cue_offsets=tuple(
                (hit.start, hit.end) for hit in hits if hit.experiencer == experiencer
            ),
            source="cue",
            review_required=experiencer == UNKNOWN,
            input_index=index,
        )

    section = _section_for_item(item, offsets, section_ranges)
    prior = section_default
    if prior is None:
        prior = _section_prior(section)
    source = cast(ClassificationSource, "section" if prior is not None else "default")
    experiencer = prior or default
    return SDOHExperiencerEvidence(
        source_offsets=offsets,
        experiencer=experiencer,
        source=source,
        review_required=experiencer == UNKNOWN,
        input_index=index,
    )


def _cue_hits(text: str, start: int, end: int) -> tuple[_CueHit, ...]:
    scope_start, scope_end = _scope_bounds(text, start, end)
    hits: list[_CueHit] = []
    for cue_pattern in _CUE_PATTERNS:
        for match in cue_pattern.pattern.finditer(text, scope_start, scope_end):
            cue_start, cue_end = match.span()
            if not _cue_reaches_target(
                text,
                cue_pattern.experiencer,
                cue_start,
                cue_end,
                start,
                end,
            ):
                continue
            hit = _CueHit(cue_pattern.experiencer, cue_start, cue_end)
            if _is_nested_patient_owner(text, hit, scope_end, end):
                continue
            hits.append(hit)

    unique = {(hit.experiencer, hit.start, hit.end): hit for hit in hits}
    return tuple(
        sorted(
            unique.values(),
            key=lambda hit: (
                hit.start,
                hit.end,
                EXPERIENCER_CLASSES.index(hit.experiencer),
            ),
        )
    )


def _is_nested_patient_owner(
    text: str,
    hit: _CueHit,
    scope_end: int,
    target_end: int,
) -> bool:
    """Ignore a patient reporter/owner when a later subject governs the span."""

    if hit.experiencer != PATIENT:
        return False
    tail_end = min(scope_end, hit.end + _MAX_CUE_DISTANCE)
    tail = text[hit.end : tail_end]
    next_subject = _next_nonpatient_subject(tail)
    if next_subject is None:
        return False
    next_subject_start = hit.end + next_subject.start()
    if next_subject_start >= target_end:
        return False
    prefix = tail[: next_subject.start()]
    matched_text = text[hit.start : hit.end]
    if matched_text.endswith(("'s", "’s")) or re.search(r"['’]s\s*$", prefix):
        return True
    return _REPORTING_VERB_RE.search(prefix) is not None


def _next_nonpatient_subject(text: str) -> re.Match[str] | None:
    patterns = [
        cue.pattern for cue in _CUE_PATTERNS if cue.experiencer in {HOUSEHOLD, FAMILY}
    ]
    matches: list[re.Match[str]] = []
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            matches.append(match)
    return min(matches, key=lambda match: (match.start(), match.end()), default=None)


def _near_target(
    cue_start: int, cue_end: int, target_start: int, target_end: int
) -> bool:
    if cue_end <= target_start:
        return target_start - cue_end <= _MAX_CUE_DISTANCE
    if target_end <= cue_start:
        return cue_start - target_end <= _MAX_CUE_DISTANCE
    return True


def _cue_reaches_target(
    text: str,
    experiencer: str,
    cue_start: int,
    cue_end: int,
    target_start: int,
    target_end: int,
) -> bool:
    if not _near_target(cue_start, cue_end, target_start, target_end):
        return False
    if cue_end <= target_start or cue_start < target_end:
        return True
    if experiencer == UNKNOWN:
        return True
    between = text[target_end:cue_start]
    return _AFTER_CUE_RELATION_RE.search(between) is not None


def _scope_bounds(text: str, start: int, end: int) -> SpanOffset:
    left = (
        max(
            (text.rfind(boundary, 0, start) for boundary in _SENTENCE_BOUNDARIES),
            default=-1,
        )
        + 1
    )
    right_candidates = [text.find(boundary, end) for boundary in _SENTENCE_BOUNDARIES]
    right_candidates = [candidate for candidate in right_candidates if candidate >= 0]
    right = min(right_candidates, default=len(text))

    for boundary in _CONTRASTIVE_BOUNDARY_RE.finditer(text, left, right):
        if boundary.end() <= start:
            left = boundary.end()
        elif boundary.start() >= end:
            right = boundary.start()
            break
    return left, right


def _provided_experiencer(item: Any) -> str | None:
    for container in _containers(item):
        value = _field(
            container,
            "sdoh_experiencer",
            "experiencer",
            "experiencer_class",
            "subject_experiencer",
        )
        if value is not None:
            return _normalize_experiencer(value)
    return None


def _containers(item: Any) -> tuple[Any, ...]:
    values = [item]
    for name in ("metadata", "clinical_context", "context"):
        nested = _field(item, name)
        if nested is not None:
            values.append(nested)
    return tuple(values)


def _section_for_item(
    item: Any,
    offsets: SpanOffset,
    section_ranges: Sequence[tuple[int, int, str | None]],
) -> str | None:
    for container in _containers(item):
        value = _field(container, "section", "section_label", "section_name")
        if isinstance(value, str) and value.strip():
            return _normalize_section(value)

    start, end = offsets
    containing = [
        section
        for section in section_ranges
        if section[0] <= start and end <= section[1]
    ]
    if not containing:
        return None
    return min(containing, key=lambda section: (section[1] - section[0], section[0]))[2]


def _section_prior(section: str | None) -> str | None:
    if section == "social_history":
        return PATIENT
    if section == "family_history":
        return FAMILY
    if section in {HOUSEHOLD, "households"}:
        return HOUSEHOLD
    return None


def _coerce_sections(
    sections: Iterable[Any] | Mapping[str, Any] | str | None,
    text_length: int,
) -> tuple[tuple[int, int, str | None], ...]:
    if sections is None:
        return ()
    if isinstance(sections, str):
        return ((0, text_length, _normalize_section(sections)),)

    if isinstance(sections, Mapping):
        if _field(sections, "start", "start_char") is not None:
            items: Iterable[Any] = (sections,)
        else:
            items = tuple(
                {"label": label, **value}
                if isinstance(value, Mapping)
                else {"label": label, "span": value}
                for label, value in sections.items()
            )
    else:
        try:
            items = tuple(sections)
        except TypeError as exc:
            raise TypeError("SDOH section collection must be iterable") from exc

    result: list[tuple[int, int, str | None]] = []
    for item in items:
        start, end = _section_offsets(item)
        if end > text_length:
            raise ValueError("SDOH section offsets must be within source text")
        label = _field(item, "label", "section", "name", "section_label")
        result.append((start, end, _normalize_section(label)))
    return tuple(sorted(result, key=lambda value: (value[0], value[1], value[2] or "")))


def _section_offsets(item: Any) -> SpanOffset:
    candidate = _field(item, "span", "source_offsets", "offsets") or item
    if isinstance(candidate, Mapping):
        start = candidate.get("start", candidate.get("start_char"))
        end = candidate.get("end", candidate.get("end_char"))
    elif _is_offset_sequence(candidate):
        start, end = candidate
    else:
        start = _field(item, "start", "start_char")
        end = _field(item, "end", "end_char")
    return _validate_offset((start, end), "SDOH section offsets")


def _normalize_section(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    canonical = canonical_section_name(value)
    if canonical is not None:
        return canonical
    normalized = "_".join(value.casefold().split())
    return normalized or None


def _evidence_items(evidence: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(evidence, Mapping) or _has_offset_field(evidence):
        return (evidence,)
    if _is_offset_sequence(evidence):
        return (evidence,)
    if isinstance(evidence, str | bytes | bytearray):
        raise TypeError("SDOH experiencer evidence must contain source offsets")
    try:
        return tuple(evidence)
    except TypeError as exc:
        raise TypeError("SDOH experiencer evidence must be iterable") from exc


def _classification_items(
    evidence: Iterable[SDOHExperiencerEvidence] | Any,
) -> tuple[SDOHExperiencerEvidence, ...]:
    if isinstance(evidence, SDOHExperiencerEvidence):
        items = (evidence,)
    else:
        try:
            items = tuple(evidence)
        except TypeError as exc:
            raise TypeError("classified SDOH evidence must be iterable") from exc
    if any(not isinstance(item, SDOHExperiencerEvidence) for item in items):
        raise TypeError("classified SDOH evidence must use SDOHExperiencerEvidence")
    return items


def _has_offset_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in value for key in ("span", "source_offsets", "source_offset", "start")
        )
    return any(
        getattr(value, key, None) is not None
        for key in ("span", "source_offsets", "source_offset", "start")
    )


def _offset_from_item(item: Any, text_length: int) -> SpanOffset:
    candidate = _field(item, "source_offsets", "source_offset", "span", "offsets")
    if candidate is None:
        candidate = item

    if isinstance(candidate, Mapping):
        start = candidate.get("start", candidate.get("source_start"))
        end = candidate.get("end", candidate.get("source_end"))
    elif _is_offset_sequence(candidate):
        start, end = candidate
    else:
        start = _field(item, "source_start", "start", "start_char", "begin")
        end = _field(item, "source_end", "end", "end_char", "stop")
    return _validate_offset((start, end), "SDOH evidence offsets", text_length)


def _field(item: Any, *names: str) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item and item[name] is not None:
                return item[name]
        return None
    for name in names:
        value = getattr(item, name, None)
        if value is not None:
            return value
    return None


def _normalize_experiencer(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("SDOH experiencer class must be a string")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "self": PATIENT,
        "subject": PATIENT,
        "current_patient": PATIENT,
        "household_member": HOUSEHOLD,
        "roommate": HOUSEHOLD,
        "housemate": HOUSEHOLD,
        "relative": FAMILY,
        "family_member": FAMILY,
        "family_history": FAMILY,
        "non_patient": UNKNOWN,
        "nonpatient": UNKNOWN,
        "other": UNKNOWN,
        "unspecified": UNKNOWN,
        "undetermined": UNKNOWN,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in EXPERIENCER_CLASSES:
        raise ValueError("unsupported SDOH experiencer class")
    return normalized


def _is_offset_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _require_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if not _is_sequence(value):
        raise TypeError(f"{field_name} fields are invalid")
    return value


def _validate_offset(
    value: Any,
    field_name: str,
    text_length: int | None = None,
) -> SpanOffset:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 2
    ):
        raise TypeError(f"{field_name} must be a two-item offset sequence")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError(f"{field_name} must contain integer offsets")
    if start < 0 or end <= start:
        raise ValueError(f"{field_name} must form a non-empty half-open span")
    if text_length is not None and end > text_length:
        raise ValueError(f"{field_name} must be within the source text")
    return start, end


__all__ = [
    "EXPERIENCER_CLASSES",
    "Experiencer",
    "ExperiencerClassification",
    "FAMILY",
    "FAMILY_EXPERIENCER",
    "HOUSEHOLD",
    "HOUSEHOLD_EXPERIENCER",
    "PATIENT",
    "PATIENT_EXPERIENCER",
    "SDOH_EXPERIENCER_ADVISORY",
    "SDOH_EXPERIENCER_FILTER_ADVISORY",
    "SDOH_EXPERIENCER_SCHEMA_VERSION",
    "SDOH_EXPERIENCER_VALUES",
    "SDOHExperiencerClassification",
    "SDOHExperiencerEvidence",
    "SDOHExperiencerFilterResult",
    "UNKNOWN",
    "UNKNOWN_EXPERIENCER",
    "attach_sdoh_experiencer",
    "classify_sdoh_evidence",
    "classify_sdoh_experiencer",
    "classify_sdoh_experiencers",
    "filter_patient_sdoh",
    "filter_sdoh_evidence",
    "filter_sdoh_experiencers",
    "filter_sdoh_findings",
]
