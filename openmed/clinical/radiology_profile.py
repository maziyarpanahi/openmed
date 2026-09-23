"""Rules-only radiology finding profile with assertion-aware provenance.

The profile composes the existing local radiology finding and report parsers.
It adds section labels, scoped assertion state, and evidence offsets without
copying surrounding report text into the result.  It does not infer a
diagnosis, assign a severity, or load a terminology service.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypedDict

from .context import ModifierHit, resolve_span_context, scan_context_cues
from .radiology_finding import (
    Laterality,
    ProvenanceSpan,
    RadLexMappingSource,
    extract_radiology_findings,
)
from .radiology_report import FINDINGS, IMPRESSION, parse_radiology_report

RadiologyProfileSection = Literal["findings", "impression"]
AssertionState = Literal["affirmed", "negated", "unknown"]

RADIOLOGY_FINDING_PROFILE_VERSION = 1
RADIOLOGY_FINDING_PROFILE_SECTIONS: tuple[RadiologyProfileSection, ...] = (
    FINDINGS,
    IMPRESSION,
)
RADIOLOGY_FINDING_PROFILE_ADVISORY = (
    "Radiology finding profiles are deterministic review aids, not diagnostic "
    "decisions. They preserve source offsets instead of surrounding report text, "
    "and never infer a finding, severity, or treatment recommendation."
)
# The shorter name is convenient for callers that already use the radiology
# profile vocabulary from the note router.
RADIOLOGY_PROFILE_ADVISORY = RADIOLOGY_FINDING_PROFILE_ADVISORY


class RadiologyProfileEvidence(TypedDict):
    """Offset-only evidence links for one profile record."""

    finding: ProvenanceSpan
    laterality: ProvenanceSpan | None
    size_value: ProvenanceSpan | None
    size_unit: ProvenanceSpan | None
    location: ProvenanceSpan | None
    assertion: ProvenanceSpan | None
    section: ProvenanceSpan


class RadiologyProfileFinding(TypedDict):
    """Structured finding fields emitted by the radiology profile.

    ``None`` values are retained for missing optional attributes.  The
    ``unknown_fields`` list makes those omissions explicit instead of silently
    dropping them.  ``evidence`` contains only half-open offsets into the
    caller's source text; the source itself is never copied into this record.
    """

    finding: str
    laterality: Laterality
    size_value: float | None
    size_unit: str | None
    location: str | None
    assertion: AssertionState
    section: RadiologyProfileSection
    radlex_code: str | None
    unknown_fields: list[str]
    evidence: RadiologyProfileEvidence


@dataclass(frozen=True)
class RadiologyFindingProfile:
    """Immutable, local-first configuration for radiology finding extraction."""

    name: str = "radiology_finding"
    version: int = RADIOLOGY_FINDING_PROFILE_VERSION
    sections: tuple[RadiologyProfileSection, ...] = RADIOLOGY_FINDING_PROFILE_SECTIONS
    advisory: str = RADIOLOGY_FINDING_PROFILE_ADVISORY

    def extract(
        self,
        text: str,
        *,
        radlex_mapping: RadLexMappingSource | None = None,
        max_attribute_distance: int = 80,
    ) -> list[RadiologyProfileFinding]:
        """Extract profile records from ``text`` without network access.

        Only ``findings`` and ``impression`` sections are interpreted as
        finding-bearing prose.  Recommendation sections remain available to a
        caller's report parser but cannot create finding records here.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        section_names = _profile_sections(self.sections)
        report = parse_radiology_report(text)
        records: list[RadiologyProfileFinding] = []
        for section in section_names:
            span = report["section_spans"].get(section)
            if span is None:
                continue
            section_start, section_end = span
            section_text = text[section_start:section_end]
            raw_findings = extract_radiology_findings(
                section_text,
                radlex_mapping=radlex_mapping,
                max_attribute_distance=max_attribute_distance,
            )
            records.extend(
                _profile_finding(
                    text,
                    section,
                    span,
                    raw_finding,
                )
                for raw_finding in raw_findings
            )
        return records

    def __call__(
        self,
        text: str,
        *,
        radlex_mapping: RadLexMappingSource | None = None,
        max_attribute_distance: int = 80,
    ) -> list[RadiologyProfileFinding]:
        """Call :meth:`extract` as a concise profile pipeline operation."""
        return self.extract(
            text,
            radlex_mapping=radlex_mapping,
            max_attribute_distance=max_attribute_distance,
        )


RADIOLOGY_FINDING_PROFILE = RadiologyFindingProfile()


def extract_radiology_profile(
    text: str,
    *,
    radlex_mapping: RadLexMappingSource | None = None,
    max_attribute_distance: int = 80,
) -> list[RadiologyProfileFinding]:
    """Extract assertion-aware radiology findings in source order.

    The result is deterministic for a fixed source string and caller-supplied
    mapping.  ``assertion`` is ``"negated"`` when a scoped negation cue reaches
    the finding, ``"unknown"`` for an uncertain or hypothetical finding, and
    ``"affirmed"`` otherwise.  Missing laterality, size, and location are
    represented by explicit unknown entries in ``unknown_fields``.
    """
    return RADIOLOGY_FINDING_PROFILE.extract(
        text,
        radlex_mapping=radlex_mapping,
        max_attribute_distance=max_attribute_distance,
    )


def extract_radiology_finding_profile(
    text: str,
    *,
    radlex_mapping: RadLexMappingSource | None = None,
    max_attribute_distance: int = 80,
) -> list[RadiologyProfileFinding]:
    """Backward-friendly alias for :func:`extract_radiology_profile`."""
    return extract_radiology_profile(
        text,
        radlex_mapping=radlex_mapping,
        max_attribute_distance=max_attribute_distance,
    )


def parse_radiology_finding_profile(
    text: str,
    *,
    radlex_mapping: RadLexMappingSource | None = None,
    max_attribute_distance: int = 80,
) -> list[RadiologyProfileFinding]:
    """Parse alias matching the report parser's ``parse_*`` naming style."""
    return extract_radiology_profile(
        text,
        radlex_mapping=radlex_mapping,
        max_attribute_distance=max_attribute_distance,
    )


def _profile_sections(
    sections: Iterable[RadiologyProfileSection],
) -> tuple[RadiologyProfileSection, ...]:
    """Return supported section names once, preserving profile order."""
    supported = {FINDINGS, IMPRESSION}
    result: list[RadiologyProfileSection] = []
    for section in sections:
        if section in supported and section not in result:
            result.append(section)
    return tuple(result)


def _shift_span(
    span: ProvenanceSpan | None,
    offset: int,
) -> ProvenanceSpan | None:
    """Translate a local parser span into the original report offsets."""
    if span is None:
        return None
    return {
        "start": offset + span["start"],
        "end": offset + span["end"],
    }


def _nearest_hit(
    hits: Iterable[ModifierHit],
    *,
    target_start: int,
    target_end: int,
) -> ModifierHit | None:
    """Select a scoped cue deterministically for assertion provenance."""
    candidates = tuple(hits)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda hit: (
            max(target_start - hit.end, hit.start - target_end, 0),
            abs((hit.start + hit.end) - (target_start + target_end)),
            hit.start,
            hit.end,
        ),
    )


def _assertion(
    source: str,
    finding_start: int,
    finding_end: int,
) -> tuple[AssertionState, ProvenanceSpan | None]:
    """Resolve one finding's assertion and return the cue offset if present."""
    target = {
        "text": source[finding_start:finding_end],
        "start": finding_start,
        "end": finding_end,
        "context": source,
    }
    scoped = scan_context_cues(source, [target])
    hits = scoped[target]
    context = resolve_span_context(target, hits)
    negation_hits = (hit for hit in hits if hit.category == "negation")
    uncertainty_hits = (
        hit for hit in hits if hit.category in {"uncertainty", "hypothetical"}
    )

    if context.negation == "negated":
        hit = _nearest_hit(
            negation_hits,
            target_start=finding_start,
            target_end=finding_end,
        )
        state: AssertionState = "negated"
    elif context.certainty == "uncertain" or context.temporality == "hypothetical":
        hit = _nearest_hit(
            uncertainty_hits,
            target_start=finding_start,
            target_end=finding_end,
        )
        state = "unknown"
    else:
        hit = None
        state = "affirmed"

    evidence = None if hit is None else {"start": hit.start, "end": hit.end}
    return state, evidence


def _profile_finding(
    source: str,
    section: RadiologyProfileSection,
    section_span: tuple[int, int],
    raw_finding: dict[str, object],
) -> RadiologyProfileFinding:
    """Convert one low-level finding into the profile contract."""
    raw_spans = raw_finding["provenance_spans"]
    if not isinstance(raw_spans, dict):  # pragma: no cover - typed helper guard
        raise TypeError("radiology finding provenance must be a mapping")

    section_start, section_end = section_span

    def shifted(name: str) -> ProvenanceSpan | None:
        value = raw_spans.get(name)
        return _shift_span(value if isinstance(value, dict) else None, section_start)

    finding_span = shifted("finding")
    if finding_span is None:  # pragma: no cover - guaranteed by source parser
        raise ValueError("radiology finding provenance is missing its finding span")

    assertion, assertion_span = _assertion(
        source,
        finding_span["start"],
        finding_span["end"],
    )
    laterality = raw_finding["laterality"]
    size_value = raw_finding["size_value"]
    size_unit = raw_finding["size_unit"]
    location = raw_finding["location"]
    unknown_fields: list[str] = []
    if laterality == "unknown":
        unknown_fields.append("laterality")
    if size_value is None or size_unit is None:
        unknown_fields.append("size")
    if location is None:
        unknown_fields.append("location")
    if assertion == "unknown":
        unknown_fields.append("assertion")

    evidence = RadiologyProfileEvidence(
        finding=finding_span,
        laterality=shifted("laterality"),
        size_value=shifted("size_value"),
        size_unit=shifted("size_unit"),
        location=shifted("location"),
        assertion=assertion_span,
        section={"start": section_start, "end": section_end},
    )
    return RadiologyProfileFinding(
        finding=raw_finding["finding"],
        laterality=laterality,
        size_value=size_value,
        size_unit=size_unit,
        location=location,
        assertion=assertion,
        section=section,
        radlex_code=raw_finding["radlex_code"],
        unknown_fields=unknown_fields,
        evidence=evidence,
    )


__all__ = [
    "AssertionState",
    "RADIOLOGY_FINDING_PROFILE",
    "RADIOLOGY_FINDING_PROFILE_ADVISORY",
    "RADIOLOGY_FINDING_PROFILE_SECTIONS",
    "RADIOLOGY_FINDING_PROFILE_VERSION",
    "RADIOLOGY_PROFILE_ADVISORY",
    "RadiologyFindingProfile",
    "RadiologyProfileEvidence",
    "RadiologyProfileFinding",
    "RadiologyProfileSection",
    "extract_radiology_finding_profile",
    "extract_radiology_profile",
    "parse_radiology_finding_profile",
]
