"""Small, local LOINC vocabulary for clinical note sections.

The map is intentionally limited to section concepts used by the local
detector.  It is a code crosswalk, not a bundled terminology release or a
replacement for a caller-managed terminology service.
"""

from __future__ import annotations

from types import MappingProxyType

LOINC_SYSTEM = "http://loinc.org"

# These are the stable section concepts used by the clinical section
# lexicons.  Keep values as strings so they can be serialized without a
# terminology dependency and passed through downstream ``sections=`` APIs.
SECTION_LOINC_MAP = MappingProxyType(
    {
        "allergies": "48765-2",
        "assessment": "51848-0",
        "assessment_and_plan": "51847-2",
        "chief_complaint": "8661-1",
        "family_history": "10157-6",
        "findings": "30954-2",
        "history": "10164-2",
        "history_of_present_illness": "10164-2",
        "impression": "19005-8",
        "medications": "10160-0",
        "past_medical_history": "11348-0",
        "plan": "18776-5",
        "problem_list": "11450-4",
        "review_of_systems": "10187-3",
        "social_history": "29762-2",
    }
)

# Public compatibility aliases use the vocabulary wording already present in
# the section detector API.
SECTION_LOINC_CODES = SECTION_LOINC_MAP
LOINC_SECTION_CODES = SECTION_LOINC_MAP


def section_loinc_code(label: str) -> str | None:
    """Return the local LOINC code for a canonical section label."""

    return SECTION_LOINC_MAP.get(label)


def section_codings(label: str) -> list[dict[str, str]]:
    """Return JSON-ready LOINC codings for *label*.

    An unknown or preamble section has no standard LOINC section concept and
    therefore returns an empty list.
    """

    code = section_loinc_code(label)
    if code is None:
        return []
    return [{"system": LOINC_SYSTEM, "code": code}]


def section_codes(label: str) -> list[str]:
    """Return the serializable LOINC code list for a section label."""

    code = section_loinc_code(label)
    return [] if code is None else [code]


__all__ = [
    "LOINC_SECTION_CODES",
    "LOINC_SYSTEM",
    "SECTION_LOINC_CODES",
    "SECTION_LOINC_MAP",
    "section_codes",
    "section_codings",
    "section_loinc_code",
]
