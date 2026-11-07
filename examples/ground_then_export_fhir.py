#!/usr/bin/env python3
"""Run a fully offline de-identify -> NER -> ground -> FHIR pipeline.

The example keeps the terminology inputs synthetic and in memory so it can be
run on a clean checkout without downloading a model or vocabulary. Clinical
mentions are found by a deterministic local NER fixture, then grounded through
the public ``ground`` facade and exported as one Observation per concept.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from openmed import DeidentificationResult, deidentify, ground
from openmed.clinical.exporters.codeable_concept_simple import (
    codeable_concept_from_grounded_concept,
)
from openmed.clinical.exporters.fhir import to_bundle
from openmed.clinical.grounding import (
    GroundedSpan,
    VocabConcept,
    VocabularyIndex,
    canonical_system,
    system_uri,
)

GROUNDING_SYSTEMS: tuple[str, ...] = ("rxnorm", "loinc", "icd10cm")

# These are the only pre-de-identification PHI fixture values. They are never
# passed to grounding or written to the Bundle.
SYNTHETIC_PHI: tuple[str, ...] = (
    "1975-04-03",
    "212-555-0198",
    "grounding.patient@example.test",
)

SYNTHETIC_NOTE = (
    "Synthetic intake for a de-identification demonstration. "
    "DOB: 1975-04-03. Phone: 212-555-0198. "
    "Email: grounding.patient@example.test. "
    "Medication review: aspirin; acetaminophen; amoxicillin; atorvastatin; "
    "lisinopril; metformin; omeprazole; albuterol; levothyroxine; "
    "insulin glargine. "
    "Laboratory review: hemoglobin A1c; serum creatinine; sodium; potassium; "
    "total cholesterol; tsh; white blood cell count; platelet count; glucose; "
    "blood pressure. "
    "Problem list: type 2 diabetes mellitus without complications; essential "
    "hypertension; hyperlipidemia; asthma; pneumonia; chronic kidney disease "
    "stage 2; hypothyroidism; obesity; major depressive disorder; "
    "gastroesophageal reflux disease without esophagitis."
)


@dataclass(frozen=True)
class _ConceptFixture:
    """One synthetic NER surface and its local terminology record."""

    surface: str
    system: str
    code: str
    label: str


_CONCEPT_FIXTURES: tuple[_ConceptFixture, ...] = (
    _ConceptFixture("aspirin", "rxnorm", "1191", "MEDICATION"),
    _ConceptFixture("acetaminophen", "rxnorm", "161", "MEDICATION"),
    _ConceptFixture("amoxicillin", "rxnorm", "723", "MEDICATION"),
    _ConceptFixture("atorvastatin", "rxnorm", "83367", "MEDICATION"),
    _ConceptFixture("lisinopril", "rxnorm", "314077", "MEDICATION"),
    _ConceptFixture("metformin", "rxnorm", "860975", "MEDICATION"),
    _ConceptFixture("omeprazole", "rxnorm", "861526", "MEDICATION"),
    _ConceptFixture("albuterol", "rxnorm", "745679", "MEDICATION"),
    _ConceptFixture("levothyroxine", "rxnorm", "966247", "MEDICATION"),
    _ConceptFixture("insulin glargine", "rxnorm", "847232", "MEDICATION"),
    _ConceptFixture("hemoglobin A1c", "loinc", "4548-4", "LAB_TEST"),
    _ConceptFixture("serum creatinine", "loinc", "2160-0", "LAB_TEST"),
    _ConceptFixture("sodium", "loinc", "2951-2", "LAB_TEST"),
    _ConceptFixture("potassium", "loinc", "2823-3", "LAB_TEST"),
    _ConceptFixture("total cholesterol", "loinc", "2093-3", "LAB_TEST"),
    _ConceptFixture("tsh", "loinc", "3016-3", "LAB_TEST"),
    _ConceptFixture("white blood cell count", "loinc", "6690-2", "LAB_TEST"),
    _ConceptFixture("platelet count", "loinc", "777-3", "LAB_TEST"),
    _ConceptFixture("glucose", "loinc", "2345-7", "LAB_TEST"),
    _ConceptFixture("blood pressure", "loinc", "85354-9", "LAB_TEST"),
    _ConceptFixture(
        "type 2 diabetes mellitus without complications",
        "icd10cm",
        "E11.9",
        "CONDITION",
    ),
    _ConceptFixture("essential hypertension", "icd10cm", "I10", "CONDITION"),
    _ConceptFixture("hyperlipidemia", "icd10cm", "E78.5", "CONDITION"),
    _ConceptFixture("asthma", "icd10cm", "J45.909", "CONDITION"),
    _ConceptFixture("pneumonia", "icd10cm", "J18.9", "CONDITION"),
    _ConceptFixture(
        "chronic kidney disease stage 2",
        "icd10cm",
        "N18.2",
        "CONDITION",
    ),
    _ConceptFixture("hypothyroidism", "icd10cm", "E03.9", "CONDITION"),
    _ConceptFixture("obesity", "icd10cm", "E66.9", "CONDITION"),
    _ConceptFixture(
        "major depressive disorder",
        "icd10cm",
        "F32.9",
        "CONDITION",
    ),
    _ConceptFixture(
        "gastroesophageal reflux disease without esophagitis",
        "icd10cm",
        "K21.9",
        "CONDITION",
    ),
)


class _SyntheticVocabularyLoader:
    """Minimal local loader implementing the grounding loader protocol."""

    local_only = True

    def __init__(self, indexes: dict[str, VocabularyIndex]) -> None:
        self._indexes = indexes

    def get_index(self, system: str) -> VocabularyIndex:
        """Return the in-memory synthetic index for ``system``."""
        return self._indexes[canonical_system(system)]

    def snapshot_provenance(
        self,
        systems: Sequence[str],
    ) -> dict[str, dict[str, str]]:
        """Return stable, PHI-free metadata for each local index."""
        result: dict[str, dict[str, str]] = {}
        for raw_system in systems:
            normalized = canonical_system(raw_system)
            index = self._indexes[normalized]
            digest = index.content_hash
            result[normalized] = {
                "system": normalized,
                "system_uri": system_uri(normalized) or "",
                "version": f"synthetic-{normalized}-2026-09",
                "sha256": digest,
                "content_hash": digest,
                "artifact": "embedded-synthetic-fixture",
            }
        return result


class _NoDownloadTokenClassificationPipeline:
    """Token-classification stand-in that avoids first-run model downloads."""

    tokenizer = None

    def __call__(self, inputs: Any, **_: Any) -> list[Any]:
        if isinstance(inputs, list):
            return [[] for _ in inputs]
        return []


class _NoDownloadLoader:
    """Loader compatible with ``deidentify`` for this offline example."""

    config = None

    def create_pipeline(self, *_: Any, **__: Any) -> Any:
        return _NoDownloadTokenClassificationPipeline()

    def get_max_sequence_length(self, *_: Any, **__: Any) -> None:
        return None


def build_local_vocabulary_loader() -> _SyntheticVocabularyLoader:
    """Build the three small synthetic vocabulary snapshots used by the demo."""
    grouped: dict[str, list[VocabConcept]] = {
        system: [] for system in GROUNDING_SYSTEMS
    }
    for fixture in _CONCEPT_FIXTURES:
        grouped[fixture.system].append(
            VocabConcept(
                system=fixture.system,
                code=fixture.code,
                preferred_term=fixture.surface,
            )
        )
    indexes = {
        system: VocabularyIndex(system, concepts)
        for system, concepts in grouped.items()
    }
    return _SyntheticVocabularyLoader(indexes)


def redact_note(note: str = SYNTHETIC_NOTE) -> DeidentificationResult:
    """De-identify the fixture before any NER or grounding work begins."""
    return deidentify(
        note,
        method="mask",
        confidence_threshold=0.5,
        loader=_NoDownloadLoader(),
        use_safety_sweep=True,
    )


def extract_clinical_entities(redacted_text: str) -> list[dict[str, Any]]:
    """Run deterministic local NER over de-identified text with exact offsets.

    A production application can replace this fixture matcher with a local NER
    model. Keeping the example matcher deterministic makes the complete example
    runnable without model downloads while exercising the same span hand-off.
    """
    entities: list[dict[str, Any]] = []
    for fixture in _CONCEPT_FIXTURES:
        pattern = re.compile(re.escape(fixture.surface), re.IGNORECASE)
        for match in pattern.finditer(redacted_text):
            entities.append(
                {
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "label": fixture.label,
                    "system": fixture.system,
                }
            )
    entities.sort(key=lambda item: (item["start"], item["end"], item["system"]))
    if len(entities) != len(_CONCEPT_FIXTURES):
        raise ValueError(
            "the synthetic NER fixture must find every expected clinical surface"
        )
    return entities


def ground_entities(
    entities: Sequence[dict[str, Any]],
    *,
    loader: _SyntheticVocabularyLoader | None = None,
) -> list[GroundedSpan]:
    """Ground de-identified NER spans against local synthetic snapshots."""
    grounded = ground(
        entities,
        systems=GROUNDING_SYSTEMS,
        loader=loader or build_local_vocabulary_loader(),
        offline=True,
    )
    selected: list[GroundedSpan] = []
    for entity, span in zip(entities, grounded):
        expected_system = canonical_system(entity["system"])
        candidates = tuple(
            candidate
            for candidate in span.candidates
            if canonical_system(candidate.system) == expected_system
        )
        if not candidates:
            raise ValueError(
                f"no {expected_system} candidate was found for the synthetic NER span"
            )
        selected.append(replace(span, candidates=candidates, alternatives=()))
    return selected


def build_fhir_resources(
    grounded_spans: Sequence[GroundedSpan],
) -> list[dict[str, Any]]:
    """Map each grounded span to a coded Observation resource."""
    resources: list[dict[str, Any]] = [
        {"resourceType": "Patient", "id": "synthetic-patient"},
        {
            "resourceType": "Encounter",
            "id": "synthetic-encounter",
            "status": "finished",
            "subject": {"reference": "Patient/synthetic-patient"},
        },
    ]
    for index, grounded_span in enumerate(grounded_spans, start=1):
        resources.append(
            {
                "resourceType": "Observation",
                "id": f"grounded-{index:02d}",
                "status": "final",
                "subject": {"reference": "Patient/synthetic-patient"},
                "encounter": {"reference": "Encounter/synthetic-encounter"},
                "code": codeable_concept_from_grounded_concept(grounded_span),
            }
        )
    return resources


def build_fhir_bundle(
    grounded_spans: Sequence[GroundedSpan],
) -> dict[str, Any]:
    """Assemble deterministic FHIR R4 resources into a transaction Bundle."""
    return to_bundle(
        build_fhir_resources(grounded_spans),
        doc_id="ground-then-export-synthetic",
    )


def run_pipeline() -> dict[str, Any]:
    """Run the leakage-first pipeline and return only PHI-safe artifacts."""
    deidentified = redact_note()
    entities = extract_clinical_entities(deidentified.deidentified_text)
    grounded_spans = ground_entities(entities)
    resources = build_fhir_resources(grounded_spans)
    return {
        "deidentified_text": deidentified.deidentified_text,
        "entities": entities,
        "grounded_spans": grounded_spans,
        "resources": resources,
        "bundle": to_bundle(
            resources,
            doc_id="ground-then-export-synthetic",
        ),
    }


def main() -> dict[str, Any]:
    """Run the demo and print the de-identified hand-offs and FHIR Bundle."""
    pipeline = run_pipeline()
    print("=== De-identified text ===")
    print(pipeline["deidentified_text"])
    print("\n=== NER spans ===")
    print(json.dumps(pipeline["entities"], indent=2, sort_keys=True))
    print("\n=== Grounded concepts ===")
    print(len(pipeline["grounded_spans"]))
    print("\n=== FHIR Bundle ===")
    print(json.dumps(pipeline["bundle"], indent=2, sort_keys=True))
    return pipeline["bundle"]


if __name__ == "__main__":
    main()
