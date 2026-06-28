#!/usr/bin/env python3
"""First-five-minutes redact -> extract -> FHIR walkthrough.

This example uses one embedded synthetic clinical note, redacts structured PII
with ``deidentify()``, extracts simple clinical entities, and emits a FHIR R4
Bundle with the existing exporter.
"""

from __future__ import annotations

import json
import logging
from typing import Any

_MODEL_LOGGER = logging.getLogger("openmed.core.models")
_MODEL_LOGGER_LEVEL = _MODEL_LOGGER.level
_MODEL_LOGGER.setLevel(logging.ERROR)
try:
    from openmed import DeidentificationResult, TextProcessor, deidentify
    from openmed.clinical.exporters.fhir import to_bundle
finally:
    _MODEL_LOGGER.setLevel(_MODEL_LOGGER_LEVEL)

SYNTHETIC_NOTE = (
    "Synthetic intake note for patient DEMO-001. DOB: 1975-04-03. "
    "Contact phone: 212-555-0198. Email: demo.patient@example.test. "
    "Assessment: type 2 diabetes with home glucose log review. "
    "Medication: metformin 500 mg twice daily. "
    "Vitals: BP: 128/76; HR: 72; Temperature: 98.4 F."
)


class _NoDownloadTokenClassificationPipeline:
    """Token-classification stand-in that avoids first-run model downloads."""

    tokenizer = None

    def __call__(self, inputs: Any, **_: Any) -> list[Any]:
        if isinstance(inputs, list):
            return [[] for _ in inputs]
        return []


class _NoDownloadLoader:
    """Loader compatible with ``deidentify(..., loader=...)`` for this demo."""

    config = None

    def create_pipeline(self, *_: Any, **__: Any) -> Any:
        return _NoDownloadTokenClassificationPipeline()

    def get_max_sequence_length(self, *_: Any, **__: Any) -> None:
        return None


def redact_note(note: str = SYNTHETIC_NOTE) -> DeidentificationResult:
    """Redact the synthetic note without downloading model artifacts."""
    return deidentify(
        note,
        method="mask",
        confidence_threshold=0.5,
        loader=_NoDownloadLoader(),
        use_safety_sweep=True,
    )


def extract_clinical_entities(redacted_text: str) -> dict[str, list[str]]:
    """Extract deterministic clinical mentions from the redacted note."""
    raw_entities = TextProcessor().extract_medical_entities(redacted_text)
    return {
        entity_type: sorted(values)
        for entity_type, values in raw_entities.items()
        if values
    }


def build_fhir_resources(entities: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Map extracted clinical mentions to minimal FHIR resources."""
    resources: list[dict[str, Any]] = [
        {"resourceType": "Patient", "id": "synthetic-patient"},
        {
            "resourceType": "Encounter",
            "id": "synthetic-encounter",
            "status": "finished",
            "subject": {"reference": "Patient/synthetic-patient"},
        },
    ]

    observation_index = 1
    for value in entities.get("vital_signs", []):
        resources.append(
            {
                "resourceType": "Observation",
                "id": f"vital-{observation_index}",
                "status": "final",
                "subject": {"reference": "Patient/synthetic-patient"},
                "encounter": {"reference": "Encounter/synthetic-encounter"},
                "code": {"text": "Synthetic vital sign mention"},
                "valueString": value,
            }
        )
        observation_index += 1

    for index, value in enumerate(entities.get("dosages", []), start=1):
        resources.append(
            {
                "resourceType": "MedicationStatement",
                "id": f"medication-{index}",
                "status": "active",
                "subject": {"reference": "Patient/synthetic-patient"},
                "context": {"reference": "Encounter/synthetic-encounter"},
                "medicationCodeableConcept": {"text": "metformin"},
                "dosage": [{"text": value}],
            }
        )

    return resources


def build_fhir_bundle(entities: dict[str, list[str]]) -> dict[str, Any]:
    """Build a deterministic transaction Bundle from extracted entities."""
    resources = build_fhir_resources(entities)
    return to_bundle(resources, doc_id="first-five-minutes-synthetic")


def main() -> dict[str, Any]:
    """Run the synthetic redact -> extract -> FHIR walkthrough."""
    deidentified = redact_note()
    entities = extract_clinical_entities(deidentified.deidentified_text)
    bundle = build_fhir_bundle(entities)

    print("=== Redacted text ===")
    print(deidentified.deidentified_text)
    print("\n=== Extracted clinical entities ===")
    print(json.dumps(entities, indent=2, sort_keys=True))
    print("\n=== FHIR bundle ===")
    print(json.dumps(bundle, indent=2, sort_keys=True))

    return bundle


if __name__ == "__main__":
    main()
