---
name: extract-clinical-entities-to-fhir
description: "Extract clinical entities from synthetic or already de-identified text with OpenMed and map them into deterministic FHIR R4 resources and a Bundle. Use when an agent must turn local clinical NER output into Conditions, MedicationStatements, Observations, or other FHIR resources without inventing terminology codes."
---

# Extract clinical entities to FHIR

Separate extraction from clinical coding. OpenMed finds spans and supplies the
mechanical FHIR builders; the application decides which resource type and
status are clinically appropriate.

## Procedure

1. Keep the source synthetic, or de-identify it inside the trusted boundary
   before extraction.
2. Run `openmed.analyze_text` with the task-appropriate clinical model.
3. Filter predictions by label and confidence; preserve offsets in a
   PHI-safe audit record.
4. Map each accepted span to the correct FHIR resource type.
5. Add terminology codes only from a user-approved mapping or terminology
   service. Never invent a code.
6. Assemble resources with `to_bundle` and validate against the target profile.

## Runnable synthetic example

Install the model runtime first with `python -m pip install "openmed[hf]"`.

```python
import json

from openmed import analyze_text
from openmed.clinical.exporters.fhir import to_bundle

note = "Assessment: type 2 diabetes mellitus is stable on metformin."
result = analyze_text(
    note,
    model_name="disease_detection_superclinical",
    confidence_threshold=0.5,
)

resources = [{"resourceType": "Patient", "id": "synthetic-patient"}]
for index, entity in enumerate(result.entities, start=1):
    if entity.label.upper() not in {"CONDITION", "DIAGNOSIS", "DISEASE"}:
        continue
    resources.append(
        {
            "resourceType": "Condition",
            "id": f"condition-{index}",
            "clinicalStatus": {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/"
                            "condition-clinical"
                        ),
                        "code": "active",
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/"
                            "condition-ver-status"
                        ),
                        "code": "confirmed",
                    }
                ]
            },
            # A text-only CodeableConcept is preferable to an invented code.
            "code": {"text": entity.text},
            "subject": {"reference": "Patient/synthetic-patient"},
        }
    )

if len(resources) == 1:
    raise RuntimeError("No condition spans met the label and confidence rules")

bundle = to_bundle(resources, doc_id="synthetic-note-001")
print(json.dumps(bundle, indent=2))
```

## Safety checks

- Do not put raw identifiers, source text, or reversible mappings in logs,
  `OperationOutcome.diagnostics`, or trace metadata.
- Keep a patient identity service separate from extracted clinical facts.
- Preserve negation, temporality, and experiencer context before asserting a
  resource as active or confirmed.
- Use a text-only `CodeableConcept` when no approved code is available.
- Validate the Bundle against the receiver's FHIR and profile requirements.
- Do not bundle restricted terminologies; use the user's licensed service.

## Repository example

Read and run
[the redaction-to-FHIR walkthrough](../../examples/first_five_minutes_redact_extract_fhir.py)
for an offline-friendly pipeline with deterministic extraction.
