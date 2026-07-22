"""Deterministic extraction of explicitly stated radiology findings.

This small, rule-based parser captures common finding terms and binds nearby
laterality, size, and anatomical location within the same sentence-like clause.
It does not infer diagnoses, negate findings, normalize measurements, or ship a
terminology.  Callers may provide a deliberately scoped RadLex mapping when a
code is needed.
"""

from __future__ import annotations

import re
from typing import Literal, Mapping, TypedDict

RADIOLOGY_FINDING_ADVISORY = (
    "Radiology findings are deterministic extraction aids intended for "
    "radiologist review. They are not diagnostic conclusions. The parser only "
    "captures explicitly written text and does not include a RadLex ontology."
)

Laterality = Literal["left", "right", "bilateral", "unknown"]


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


_LATERALITY_TERMS: dict[str, Laterality] = {
    "bilaterally": "bilateral",
    "bilateral": "bilateral",
    "left-sided": "left",
    "right-sided": "right",
    "left": "left",
    "right": "right",
    "both": "bilateral",
    "lt": "left",
    "rt": "right",
}
_LATERALITY_RE = re.compile(
    r"\b(?:" + "|".join(map(re.escape, _LATERALITY_TERMS)) + r")\b",
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


def _nearest(
    matches: list[re.Match[str]], finding: re.Match[str]
) -> re.Match[str] | None:
    """Return the closest match to a finding, with source order breaking ties."""
    if not matches:
        return None
    return min(
        matches,
        key=lambda match: (
            max(finding.start() - match.end(), match.start() - finding.end(), 0),
            abs((match.start() + match.end()) - (finding.start() + finding.end())),
            match.start(),
        ),
    )


def _normalized_unit(unit: str) -> str:
    return "mm" if unit.casefold().startswith("mm") or unit.casefold().startswith("mill") else "cm"


def _radlex_code(finding: str, mapping: Mapping[str, str] | None) -> str | None:
    """Look up one finding without modifying or supplementing caller mapping."""
    if mapping is None:
        return None
    if finding in mapping:
        return mapping[finding]
    folded = finding.casefold()
    for key, value in mapping.items():
        if key.casefold() == folded:
            return value
    return None


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


def extract_radiology_findings(
    text: str,
    *,
    radlex_mapping: Mapping[str, str] | None = None,
) -> list[RadiologyFinding]:
    """Extract explicitly stated findings in deterministic report order.

    Attributes are bound only to a finding in the same sentence-like clause.
    When several attributes occur in that clause, the closest one is used;
    source position is the deterministic tie-breaker. ``provenance_spans``
    always contains ``finding`` and additionally contains the written
    ``laterality``, ``size``, and ``location`` attributes when present.

    Args:
        text: Raw radiology report text.
        radlex_mapping: Optional caller-owned mapping from finding text to a
            RadLex code. No ontology is bundled or accessed by this parser.

    Returns:
        Findings in their source order. Empty or whitespace-only input returns
        an empty list.
    """
    source = text or ""
    findings: list[RadiologyFinding] = []
    for clause_start, clause_end in _clause_matches(source):
        clause = source[clause_start:clause_end]
        lateralities = list(_LATERALITY_RE.finditer(clause))
        sizes = list(_SIZE_RE.finditer(clause))
        locations = list(_LOCATION_RE.finditer(clause))
        for match in _FINDING_RE.finditer(clause):
            laterality_match = _nearest(lateralities, match)
            size_match = _nearest(sizes, match)
            location_match = _nearest(locations, match)
            finding = _FINDING_TERMS[match.group().casefold()]
            spans: dict[str, ProvenanceSpan] = {
                "finding": {"start": clause_start + match.start(), "end": clause_start + match.end()}
            }
            laterality: Laterality = "unknown"
            if laterality_match is not None:
                laterality = _LATERALITY_TERMS[laterality_match.group().casefold()]
                spans["laterality"] = {
                    "start": clause_start + laterality_match.start(),
                    "end": clause_start + laterality_match.end(),
                }
            size_value: float | None = None
            size_unit: str | None = None
            if size_match is not None:
                size_value = float(size_match.group("value"))
                size_unit = _normalized_unit(size_match.group("unit"))
                spans["size"] = {
                    "start": clause_start + size_match.start(),
                    "end": clause_start + size_match.end(),
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
                    radlex_code=_radlex_code(finding, radlex_mapping),
                    provenance_spans=spans,
                )
            )
    return findings


__all__ = [
    "RADIOLOGY_FINDING_ADVISORY",
    "Laterality",
    "ProvenanceSpan",
    "RadiologyFinding",
    "extract_radiology_findings",
]
