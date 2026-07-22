"""Deterministic extraction of explicitly stated radiology findings.

This small, rule-based parser captures common finding terms and binds nearby
laterality, size, and anatomical location with a span-proximity graph. It does
not infer diagnoses, negate findings, normalize measurements, or ship a
terminology. Callers may provide a deliberately scoped RadLex mapping or local
JSON mapping file when a code is needed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypedDict

RADIOLOGY_FINDING_ADVISORY = (
    "Radiology findings are deterministic extraction aids intended for "
    "radiologist review. They are not diagnostic conclusions. The parser only "
    "captures explicitly written text and does not include a RadLex ontology."
)

Laterality = Literal["left", "right", "bilateral", "unknown"]
RadLexMappingSource = Mapping[str, str] | str | PathLike[str]

#: Documented, case-insensitive surface lexicon used for laterality binding.
#: The immutable mapping is intentionally small and contains no terminology
#: data. Longer surfaces are preferred when the matcher is constructed.
RADIOLOGY_LATERALITY_LEXICON: Mapping[str, Laterality] = MappingProxyType(
    {
        "bilaterally": "bilateral",
        "bilateral": "bilateral",
        "left-sided": "left",
        "left sided": "left",
        "right-sided": "right",
        "right sided": "right",
        "left": "left",
        "right": "right",
        "both": "bilateral",
        "b/l": "bilateral",
        "lt": "left",
        "rt": "right",
    }
)


class ProvenanceSpan(TypedDict):
    """A half-open character range into the source report."""

    start: int
    end: int


class RadiologyFinding(TypedDict):
    """One explicitly written finding and its bound report attributes."""

    finding: str
    laterality: Laterality
    size_value: float | None
    size_unit: str | None
    location: str | None
    radlex_code: str | None
    provenance_spans: dict[str, ProvenanceSpan]


_LATERALITY_RE = re.compile(
    r"(?<![\w/])(?:"
    + "|".join(
        re.escape(term)
        for term in sorted(RADIOLOGY_LATERALITY_LEXICON, key=len, reverse=True)
    )
    + r")(?![\w/])",
    re.IGNORECASE,
)
_SIZE_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mm|millimeters?|cm|centimeters?)\b",
    re.IGNORECASE,
)

# Finding terms are deliberately a small, transparent lexicon rather than an
# embedded ontology.  Plurals map to the singular term returned to callers.
_FINDING_TERMS: dict[str, str] = {
    "atelectasis": "atelectasis",
    "calcification": "calcification",
    "calcifications": "calcification",
    "consolidation": "consolidation",
    "cyst": "cyst",
    "cysts": "cyst",
    "effusion": "effusion",
    "fracture": "fracture",
    "hematoma": "hematoma",
    "hernia": "hernia",
    "infiltrate": "infiltrate",
    "lesion": "lesion",
    "lesions": "lesion",
    "mass": "mass",
    "masses": "mass",
    "nodule": "nodule",
    "nodules": "nodule",
    "opacity": "opacity",
    "opacities": "opacity",
    "pneumothorax": "pneumothorax",
    "scarring": "scarring",
    "stenosis": "stenosis",
}
_FINDING_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_FINDING_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Location phrases are kept in the report's wording apart from case folding.
# They intentionally omit laterality, which is represented separately.
_LOCATION_RE = re.compile(
    r"\b(?:"
    r"(?:upper|middle|lower)\s+(?:lobes?|quadrants?)"
    r"|apex|apical(?:\s+segment)?|base|basal(?:\s+segment)?"
    r"|central|peripheral|hilar|perihilar|subpleural|retroareolar"
    r"|mediastinum|lung|lungs|breast|breasts|axilla|axillary\s+tail"
    r"|[A-Za-z]+\s+(?:lobe|quadrant|pole|segment)"
    r")\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"(?<!\d)[.!?](?!\d)|[;\n]")
_BINDING_CONNECTOR_RE = re.compile(r"(?i),|\b(?:and|but|plus|with)\b")
_DEFAULT_MAX_ATTRIBUTE_DISTANCE = 80


def _span_gap(first: re.Match[str], second: re.Match[str]) -> int:
    """Return the character gap between two non-overlapping mention spans."""
    return max(first.start() - second.end(), second.start() - first.end(), 0)


def _nearest(
    matches: list[re.Match[str]],
    finding: re.Match[str],
    *,
    max_distance: int,
) -> re.Match[str] | None:
    """Return the closest graph neighbor, with source order breaking ties."""
    candidates = [
        match for match in matches if _span_gap(match, finding) <= max_distance
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda match: (
            _span_gap(match, finding),
            abs((match.start() + match.end()) - (finding.start() + finding.end())),
            match.start(),
        ),
    )


def _normalized_unit(unit: str) -> str:
    return (
        "mm"
        if unit.casefold().startswith("mm") or unit.casefold().startswith("mill")
        else "cm"
    )


def _validated_radlex_mapping(mapping: object, *, source: str) -> dict[str, str]:
    """Validate and case-fold a caller-owned RadLex mapping."""
    if not isinstance(mapping, Mapping):
        raise ValueError(
            f"{source} must contain a JSON object mapping findings to codes"
        )

    normalized: dict[str, str] = {}
    for finding, code in mapping.items():
        if not isinstance(finding, str) or not finding.strip():
            raise ValueError(f"{source} finding keys must be non-empty strings")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"{source} RadLex codes must be non-empty strings")
        key = finding.casefold()
        if key in normalized and normalized[key] != code:
            raise ValueError(f"{source} contains conflicting keys for {finding!r}")
        normalized[key] = code
    return normalized


def _load_radlex_mapping(
    source: RadLexMappingSource | None,
) -> dict[str, str]:
    """Load only a caller-supplied mapping or local JSON mapping file."""
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return _validated_radlex_mapping(source, source="radlex_mapping")
    if not isinstance(source, (str, PathLike)):
        raise TypeError("radlex_mapping must be a mapping or local JSON file path")

    path = Path(source)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return _validated_radlex_mapping(payload, source=str(path))


def _clause_matches(text: str) -> list[tuple[int, int]]:
    """Return non-empty sentence-like clauses, preserving source offsets."""
    clauses: list[tuple[int, int]] = []
    start = 0
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(text):
        if text[start : boundary.start()].strip():
            clauses.append((start, boundary.start()))
        start = boundary.end()
    if text[start:].strip():
        clauses.append((start, len(text)))
    return clauses


def _binding_windows(
    clause: str,
    finding_matches: list[re.Match[str]],
) -> list[tuple[int, int]]:
    """Partition a clause into graph neighborhoods around each finding.

    Coordinating punctuation and words between adjacent findings prune graph
    edges so attributes from ``right ... nodule and left ... mass`` cannot
    cross-bind. With no connector, the midpoint is the deterministic boundary.
    """
    if not finding_matches:
        return []

    boundaries = [0]
    for left, right in zip(finding_matches, finding_matches[1:]):
        between = clause[left.end() : right.start()]
        connectors = list(_BINDING_CONNECTOR_RE.finditer(between))
        if connectors:
            connector = connectors[-1]
            boundary = left.end() + (connector.start() + connector.end()) // 2
        else:
            boundary = (left.end() + right.start()) // 2
        boundaries.append(boundary)
    boundaries.append(len(clause))
    return list(zip(boundaries, boundaries[1:]))


def _matches_in_window(
    matches: list[re.Match[str]],
    window: tuple[int, int],
) -> list[re.Match[str]]:
    """Return mentions whose midpoint belongs to one binding window."""
    start, end = window
    return [
        match for match in matches if start <= (match.start() + match.end()) / 2 < end
    ]


def extract_radiology_findings(
    text: str,
    *,
    radlex_mapping: RadLexMappingSource | None = None,
    max_attribute_distance: int = _DEFAULT_MAX_ATTRIBUTE_DISTANCE,
) -> list[RadiologyFinding]:
    """Extract explicitly stated findings in deterministic report order.

    Each sentence-like clause becomes a span-proximity graph. Finding mentions
    connect to attribute mentions in their connector-delimited neighborhood
    and within ``max_attribute_distance`` characters. The shortest edge wins;
    source position is the deterministic tie-breaker. ``provenance_spans``
    always contains ``finding`` and additionally contains the written
    ``laterality``, ``size_value``, ``size_unit``, and ``location`` fields when
    present.

    Args:
        text: Raw radiology report text.
        radlex_mapping: Optional caller-owned mapping from finding text to a
            RadLex code, or a path to a local UTF-8 JSON object with that shape.
            No ontology is bundled, downloaded, or otherwise accessed.
        max_attribute_distance: Maximum character gap for a finding-to-
            attribute graph edge. Must be non-negative.

    Returns:
        Findings in their source order. Empty or whitespace-only input returns
        an empty list.
    """
    if max_attribute_distance < 0:
        raise ValueError("max_attribute_distance must be non-negative")

    source = text or ""
    normalized_radlex_mapping = _load_radlex_mapping(radlex_mapping)
    findings: list[RadiologyFinding] = []
    for clause_start, clause_end in _clause_matches(source):
        clause = source[clause_start:clause_end]
        finding_matches = list(_FINDING_RE.finditer(clause))
        lateralities = list(_LATERALITY_RE.finditer(clause))
        sizes = list(_SIZE_RE.finditer(clause))
        locations = list(_LOCATION_RE.finditer(clause))
        windows = _binding_windows(clause, finding_matches)
        for match, window in zip(finding_matches, windows):
            laterality_match = _nearest(
                _matches_in_window(lateralities, window),
                match,
                max_distance=max_attribute_distance,
            )
            size_match = _nearest(
                _matches_in_window(sizes, window),
                match,
                max_distance=max_attribute_distance,
            )
            location_match = _nearest(
                _matches_in_window(locations, window),
                match,
                max_distance=max_attribute_distance,
            )
            finding = _FINDING_TERMS[match.group().casefold()]
            spans: dict[str, ProvenanceSpan] = {
                "finding": {
                    "start": clause_start + match.start(),
                    "end": clause_start + match.end(),
                }
            }
            laterality: Laterality = "unknown"
            if laterality_match is not None:
                laterality = RADIOLOGY_LATERALITY_LEXICON[
                    laterality_match.group().casefold()
                ]
                spans["laterality"] = {
                    "start": clause_start + laterality_match.start(),
                    "end": clause_start + laterality_match.end(),
                }
            size_value: float | None = None
            size_unit: str | None = None
            if size_match is not None:
                size_value = float(size_match.group("value"))
                size_unit = _normalized_unit(size_match.group("unit"))
                spans["size_value"] = {
                    "start": clause_start + size_match.start("value"),
                    "end": clause_start + size_match.end("value"),
                }
                spans["size_unit"] = {
                    "start": clause_start + size_match.start("unit"),
                    "end": clause_start + size_match.end("unit"),
                }
            location: str | None = None
            if location_match is not None:
                location = location_match.group().casefold()
                spans["location"] = {
                    "start": clause_start + location_match.start(),
                    "end": clause_start + location_match.end(),
                }
            findings.append(
                RadiologyFinding(
                    finding=finding,
                    laterality=laterality,
                    size_value=size_value,
                    size_unit=size_unit,
                    location=location,
                    radlex_code=normalized_radlex_mapping.get(finding.casefold()),
                    provenance_spans=spans,
                )
            )
    return findings


__all__ = [
    "RADIOLOGY_FINDING_ADVISORY",
    "RADIOLOGY_LATERALITY_LEXICON",
    "Laterality",
    "ProvenanceSpan",
    "RadLexMappingSource",
    "RadiologyFinding",
    "extract_radiology_findings",
]
