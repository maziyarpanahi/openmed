"""FHIR R4 representation of validated SNOMED CT expressions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ...grounding.ecl import ECLValidator
from ...grounding.postcoordination import SnomedExpression
from ..codeable_concept_simple import codeable_concept

__all__ = [
    "POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL",
    "postcoordinated_codeable_concept",
    "stamp_postcoordination_provenance",
]

POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/snomed-postcoordination-provenance"
)
_SNOMED_URI = "http://snomed.info/sct"


def stamp_postcoordination_provenance(
    coding: Mapping[str, Any],
    *,
    edition_uri: str,
) -> dict[str, Any]:
    """Mark a SNOMED ``Coding`` as validated and composed, not looked up."""

    if coding.get("system") != _SNOMED_URI:
        raise ValueError("post-coordination provenance requires a SNOMED Coding")
    if not isinstance(edition_uri, str) or not edition_uri.strip():
        raise ValueError("post-coordinated Coding requires a caller edition URI")
    result = deepcopy(dict(coding))
    result["version"] = edition_uri.strip()
    provenance = {
        "url": POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL,
        "extension": [
            {"url": "origin", "valueCode": "composed"},
            {"url": "eclValidated", "valueBoolean": True},
        ],
    }
    extensions = result.get("extension")
    if extensions is None:
        result["extension"] = [provenance]
        return result
    if not isinstance(extensions, list):
        raise TypeError("Coding.extension must be a list when present")
    result["extension"] = [
        extension
        for extension in extensions
        if not (
            isinstance(extension, Mapping)
            and extension.get("url") == POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL
        )
    ]
    result["extension"].append(provenance)
    return result


def postcoordinated_codeable_concept(
    expression: SnomedExpression,
    *,
    validator: ECLValidator,
    text: str | None = None,
    display: str | None = None,
) -> dict[str, Any]:
    """Validate and build a CodeableConcept containing a composed expression."""

    if not isinstance(expression, SnomedExpression):
        raise TypeError("expression must be a SnomedExpression")
    if not isinstance(validator, ECLValidator):
        raise TypeError("validator must be an edition-backed ECLValidator")
    validator.require_valid(expression)
    coding: dict[str, Any] = {
        "system": _SNOMED_URI,
        "code": expression.to_scg(),
        "userSelected": False,
    }
    if display is not None:
        coding["display"] = display
    coding = stamp_postcoordination_provenance(
        coding,
        edition_uri=validator.edition_uri,
    )
    return codeable_concept([coding], text=text)
