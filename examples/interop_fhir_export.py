#!/usr/bin/env python3
"""Export synthetic grounded clinical spans as a FHIR R4 Bundle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from openmed.clinical.exporters import to_fhir
from openmed.clinical.grounding import Candidate, GroundedSpan

_FHIR_RESOURCE_TYPES = frozenset(
    {"Condition", "MedicationStatement", "Observation", "Procedure"}
)


def _span(
    text: str,
    start: int,
    label: str,
    system: str,
    code: str,
    *,
    display: str,
    metadata: Mapping[str, Any] | None = None,
) -> GroundedSpan:
    """Build one deterministic synthetic grounding result."""
    return GroundedSpan(
        text=text,
        start=start,
        end=start + len(text),
        canonical_label=label,
        candidates=(
            Candidate(
                system=system,
                code=code,
                display=display,
                score=1.0,
                source="synthetic-example",
                matched_alias=display,
                match_kind="exact",
            ),
        ),
        metadata=metadata or {},
    )


def example_spans() -> tuple[GroundedSpan, ...]:
    """Return fabricated spans with representative clinical codings."""
    return (
        _span(
            "type 2 diabetes",
            0,
            "CONDITION",
            "ICD10CM",
            "E11.9",
            display="Type 2 diabetes mellitus without complications",
        ),
        _span(
            "metformin 500 mg tablet",
            24,
            "MEDICATION",
            "RXNORM",
            "860975",
            display="Metformin hydrochloride 500 MG oral tablet",
        ),
        _span(
            "glucose 130 mg/dL",
            55,
            "LAB_TEST",
            "LOINC",
            "2345-7",
            display="Glucose [Mass/volume] in Serum or Plasma",
            metadata={"value": 130, "unit": "mg/dL"},
        ),
    )


def validate_example_bundle_shape(bundle: Mapping[str, Any]) -> None:
    """Fail fast when the example did not produce its expected R4 shape.

    This is a dependency-free smoke check for the example's transaction Bundle,
    not a replacement for full FHIR profile validation.
    """
    if bundle.get("resourceType") != "Bundle":
        raise ValueError("expected a FHIR Bundle")
    if bundle.get("type") != "transaction":
        raise ValueError("expected a FHIR transaction Bundle")

    entries = bundle.get("entry")
    if not isinstance(entries, list) or not entries:
        raise ValueError("expected a non-empty Bundle.entry list")

    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Bundle.entry[{index}] must be an object")
        if not str(entry.get("fullUrl", "")).startswith("urn:uuid:"):
            raise ValueError(f"Bundle.entry[{index}].fullUrl must be a UUID URN")

        resource = entry.get("resource")
        if not isinstance(resource, Mapping):
            raise ValueError(f"Bundle.entry[{index}].resource must be an object")
        resource_type = resource.get("resourceType")
        if resource_type not in _FHIR_RESOURCE_TYPES:
            raise ValueError(
                f"Bundle.entry[{index}] has unsupported resourceType {resource_type!r}"
            )

        request = entry.get("request")
        if not isinstance(request, Mapping):
            raise ValueError(f"Bundle.entry[{index}].request must be an object")
        if request.get("method") != "POST" or request.get("url") != resource_type:
            raise ValueError(f"Bundle.entry[{index}].request is inconsistent")


def build_bundle() -> dict[str, Any]:
    """Export and smoke-check the fabricated spans."""
    bundle = to_fhir(
        example_spans(),
        document_id="synthetic-interop-example",
        subject_reference="Patient/synthetic-patient",
    )
    if bundle is None:
        raise RuntimeError("FHIR export unexpectedly returned no Bundle")
    validate_example_bundle_shape(bundle)
    return bundle


def main() -> dict[str, Any]:
    """Print the synthetic FHIR Bundle as JSON and return it."""
    bundle = build_bundle()
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return bundle


if __name__ == "__main__":
    main()
