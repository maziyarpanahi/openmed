# FHIR Interop Helpers

OpenMed exposes small FHIR R4 helpers for producing resources that downstream
FHIR servers and clients already understand. These helpers are local and
mechanical: they shape data you provide, but they do not call external
validators or network services.

## OperationOutcome

Use `to_operation_outcome()` when an exporter, operation wrapper, or validation
pass needs to report errors, warnings, or informational notes in a FHIR-native
shape.

```python
from openmed.clinical.exporters.fhir import (
    OperationOutcomeIssue,
    to_operation_outcome,
)

outcome = to_operation_outcome(
    [
        OperationOutcomeIssue(
            severity="error",
            code="required",
            diagnostics="Patient.name is required.",
            expression="Patient.name",
        ),
        {
            "severity": "warning",
            "code": "code-invalid",
            "diagnostics": "Unknown LOINC code.",
            "expression": "Observation.code.coding[0]",
        },
    ]
)
```

The returned resource is an R4 `OperationOutcome`:

```python
{
    "resourceType": "OperationOutcome",
    "issue": [
        {
            "severity": "error",
            "code": "required",
            "diagnostics": "Patient.name is required.",
            "expression": ["Patient.name"],
        },
        {
            "severity": "warning",
            "code": "code-invalid",
            "diagnostics": "Unknown LOINC code.",
            "expression": ["Observation.code.coding[0]"],
        },
    ],
}
```

Each issue must use FHIR R4 issue-severity values: `fatal`, `error`, `warning`,
or `information`. Issue codes must come from the R4 issue-type value set, such
as `invalid`, `structure`, `required`, `value`, `invariant`, `processing`,
`business-rule`, `exception`, or `informational`.

When there are no findings, `to_operation_outcome([])` returns a valid all-ok
resource with one informational issue:

```python
{
    "resourceType": "OperationOutcome",
    "issue": [
        {
            "severity": "information",
            "code": "informational",
            "diagnostics": "No issues detected.",
        }
    ],
}
```

For compatibility with older or ad-hoc result objects, `from_validation_result()`
accepts duck-typed shapes such as:

```python
from openmed.clinical.exporters.fhir import from_validation_result

result = {
    "errors": [
        {
            "message": "Malformed Patient resource.",
            "path": "Patient",
        }
    ],
    "warnings": ["Bundle.entry[0] has an unsupported profile."],
}

outcome = from_validation_result(result)
```

`from_validation_result()` is an adapter only. It does not implement structural
validation, US Core conformance checks, or the FHIR `$de-identify` operation.
Those producers should emit issue-like objects and pass them through this shared
builder.

## Privacy Boundary

FHIR `OperationOutcome.issue.diagnostics` is human-readable and may be logged by
servers, clients, gateways, or observability tools. Do not put raw PHI or direct
identifiers in diagnostics. Prefer `expression` paths, offsets, hashes,
provenance identifiers, and risk scores when reporting where a problem occurred.

OpenMed emits R4 `issue.expression` for element paths. It accepts legacy
`location` as input for adapter compatibility, but it never emits
`issue.location` because that field is deprecated in FHIR R4.

## Passive SDOH Observations

`to_sdoh_observations()` projects value-free SDOH evidence into local R4
`Observation` and `Provenance` resources. It accepts aligned `SDOHEvidence`,
`SDOHExperiencerEvidence` and `SDOHTemporalEvidence` contracts rather than raw
findings or source text. It performs no extraction, server write, credential
resolution, approval verification or storage. Applications retain responsibility
for authentic review, approved terminology and disclosure policy.

The category mapping pins [SDOH Clinical Care 2.3.0](https://hl7.org/fhir/us/sdoh-clinicalcare/STU2.3/ValueSet-SDOHCC-ValueSetSDOHCategory.html)
and [US Core Category 7.0.0](https://hl7.org/fhir/us/core/STU7/CodeSystem-us-core-category.html),
alongside R4 `social-history`. These are explicit interoperability baselines,
not claims to use the latest IG. Only HL7 category codes are bundled; clinical
observation and answer codes come from trusted caller configuration. No
SNOMED CT, UMLS, corpus data, restricted terminology expansion or weights are
shipped or fetched.

This synthetic record is pending review and therefore remains preliminary.
The example prints only a count and controlled status:

```python
# Runnable: synthetic passive SDOH projection with explicit caller bindings.
from datetime import datetime, timezone

from openmed.clinical.exporters.fhir import (
    SDOHFHIRCode,
    SDOHFHIRRecord,
    SDOHObservationBinding,
    to_sdoh_observations,
)
from openmed.clinical.sdoh_evidence import SDOHEvidence
from openmed.clinical.sdoh_experiencer import SDOHExperiencerEvidence
from openmed.clinical.sdoh_sensitive_use import SDOHPurpose
from openmed.clinical.sdoh_temporal import SDOHTemporalEvidence

offsets = (10, 20)
record = SDOHFHIRRecord(
    evidence=SDOHEvidence(
        "self_report",
        "present",
        "social_history",
        offsets,
        "needs_review",
        "food_insecurity",
    ),
    experiencer=SDOHExperiencerEvidence(offsets, "patient", source="provided"),
    temporal=SDOHTemporalEvidence(offsets, "current"),
)
system = "https://synthetic.example/CodeSystem/sdoh"
binding = SDOHObservationBinding(
    SDOHFHIRCode(system, "synthetic-assessment"),
    SDOHFHIRCode(system, "synthetic-reviewed-answer"),
)
result = to_sdoh_observations(
    [record],
    terminology={0: binding},
    subject_reference="urn:uuid:00000000-0000-4000-8000-000000000001",
    source_reference="urn:uuid:00000000-0000-4000-8000-000000000002",
    software_reference="urn:uuid:00000000-0000-4000-8000-000000000003",
    purpose=SDOHPurpose.CLINICAL_REVIEW,
    clock=lambda: datetime(2026, 1, 2, tzinfo=timezone.utc),
)
print(
    {
        "exported_count": len(result.observations),
        "status": result.observations[0]["status"],
    }
)
```

The terminology map uses each input record's integer index. This allows two
records with the same determinant to carry different explicitly reviewed
answers. The exporter never infers an answer from a determinant, assertion or
source surface. Missing bindings produce `terminology_unmapped` exclusions.
A binding without an answer retains a preliminary Observation with
`dataAbsentReason=unknown` and an `answer_unmapped` partial loss. Syntax checks
cannot establish a code's clinical meaning, licensing or absence of identifiers;
bindings must come from approved configuration, never source text.

| Determinant | Pinned domain category |
| --- | --- |
| `food_insecurity` | `food-insecurity` |
| `employment`, `employment_status` | `employment-status` |
| `housing`, `housing_insecurity` | Explicit `housing-instability`, `homelessness` or `inadequate-housing` |
| `financial_strain` | `financial-insecurity` |
| `transportation` | `transportation-insecurity` |
| `education` | `educational-attainment` |
| `insurance` | `health-insurance-coverage-status` |
| `social_support` | `social-connection` |
| `utilities` | `utility-insecurity` |

Housing requires `SDOHObservationBinding(domain_category=...)`: generic housing
evidence cannot establish which of its three distinct domains applies. Omission
is a `domain_ambiguous` exclusion; a category from another determinant is a
`domain_binding_mismatch` exclusion. Other unsupported determinants produce
`domain_unmapped` rather than a guessed category or patient observation.

Only affirmative patient evidence with review completed, a current resolved
temporal qualifier and an explicit answer can be final. Unreviewed or pending
records remain preliminary. Historical, future, unknown or temporally unresolved
records remain preliminary even after assertion review. Negated needs,
non-patient/unresolved experiencers, uncertain assertions, rejected review and
unconfirmed evidence are explicit exclusions. `losses` contains only input
indices, controlled codes and an `excluded` flag. Partial answer losses can
coexist with an exported preliminary record. Identical duplicates are reported
as exclusions rather than silently disappearing.

The three references must be distinct canonical lowercase UUIDv4 URNs for the
Patient, DocumentReference and Device. UUID syntax does not prove opacity,
provenance or unlinkability. The application owns protected mappings and their
resolution. `Provenance.target` links the Observation, and its source entity
carries the source reference, half-open offsets and controlled evidence,
review, assertion, section, experiencer and temporal labels. Offsets must align
across the three input contracts and fit the FHIR unsigned integer range.
No source surfaces, names, direct identifiers, display text, narrative,
diagnostics, credential values or source URLs are copied into resources/errors.

Every exported Observation and Provenance carries the same restrictive
`meta.security` labels: sensitive SDOH, required human review, allowed purposes
and all five prohibited automated uses from the existing sensitive-use
contract. An explicit `SDOHPurpose` must be allowed by the record's policy;
weakened review/prohibited-use labels produce `sensitive_use_refused` exclusions.
Labels preserve policy metadata; receiving systems must enforce it. They do
not confer authority, verify review or permit automated eligibility,
underwriting, employment, care denial or diagnosis decisions.

Temporal classes do not contain calendar anchors. Optional `effective_start`
and `effective_end` use only caller-approved calendar values: exact ISO dates
or aware datetimes with seconds and at most six fractional digits. Dates keep
day precision; datetimes normalize to UTC. A matching-precision end creates an
`effectivePeriod` and cannot precede its start. Unknown/conflicting temporal
classes reject supplied calendar values. No effective time uses the wall clock.
The caller must apply required date shifting/disclosure controls before passing
clinical dates. The injected clock supplies only technical `Provenance.recorded`,
is called once per exported batch and is skipped when all records are excluded.

At most 512 typed records and 512 indexed bindings are accepted. Terminology
mappings are inspected once into a bounded local snapshot; projection never
reinvokes provider getters. Exported qualifier codes must be exact built-in
strings, and malformed UTC offset minutes are refused before normalization. Errors use
fixed codes and suppress raw clock/calendar exceptions. Identical inputs and
recorded instants produce identical resources; retain that instant for replay.
Observation IDs bind the emitted metadata; Provenance IDs also bind recorded
time. These digests establish neither review authenticity nor clinical validity.

Each Observation passes the existing local base-R4 structural checker. Tests
also exercise `check_bundle()` against an authored synthetic profile with a
negative missing-category control. This is partial structural engineering
evidence. The exporter does not declare `meta.profile` or claim full conformance
to the [SDOHCC Observation Assessment profile](https://hl7.org/fhir/us/sdoh-clinicalcare/STU2.3/StructureDefinition-SDOHCC-ObservationAssessment.html).
For deployment, validate a caller-supplied local IG snapshot and all unsupported
constraints with an appropriate complete validator before declaring conformance.

## Bundles

Use `to_fhir()` when the inputs are grounded clinical spans. The facade routes
supported canonical labels, assembles their resources through `to_bundle()`,
and leaves unknown labels out of the Bundle with counts in a PHI-free sidecar:

```python
from openmed.clinical.exporters.fhir import to_fhir

bundle = to_fhir(grounded_spans, doc_id="note-123")
print(bundle.summary.exported_by_label)
print(bundle.summary.unmapped_by_label)
```

The sidecar is available as `bundle.summary` but is not a key in the FHIR
mapping, so `json.dumps(bundle)` contains only the R4 Bundle. The facade does
not synthesize a Patient resource. It emits `Condition`, `Observation`,
`MedicationStatement`, and `Procedure` for their supported canonical labels;
labels whose standalone exporters have not shipped are counted as unmapped.

Use `to_bundle()` to assemble standalone FHIR resources into a deterministic R4
`Bundle`.

```python
from openmed.clinical.exporters.fhir import to_bundle

bundle = to_bundle(
    [
        {
            "resourceType": "Observation",
            "id": "obs1",
            "status": "final",
            "code": {"text": "Glucose"},
        },
        {
            "resourceType": "DiagnosticReport",
            "id": "report1",
            "status": "final",
            "result": [{"reference": "Observation/obs1"}],
        },
    ],
    doc_id="note-123",
)
```

The helper assigns stable `urn:uuid` `fullUrl` values and rewrites internal
references that point to resources present in the bundle. It does not synthesize
missing resources and does not validate external FHIR profiles.

## Ground, then export

`examples/ground_then_export_fhir.py` is a runnable, fully offline composition
of the privacy and interoperability stages. It de-identifies a synthetic note
first, runs a deterministic local NER fixture over the de-identified text,
grounds 30 mentions across RxNorm, LOINC, and ICD-10-CM, maps each selected
candidate to a FHIR `CodeableConcept`, and assembles the resources with
`to_bundle()`:

```bash
python3 -m examples.ground_then_export_fhir
```

The example uses in-memory synthetic vocabulary indexes and a no-download PII
loader. Each emitted `Coding` carries its canonical vocabulary URI, snapshot
version, linker, score, and source offsets through the grounding provenance
extension. The adapter accepts the checked-in `GroundedSpan` result shape and
the one-system grounded-concept attributes used by newer grounding callers.
Grounding remains assist-only and requires human verification; it is not an
autonomous clinical coding, diagnosis, treatment, or billing decision.

When a caller knows the expected source vocabulary, the local conformance helper
can assert its URI as well as the CodeableConcept shape:

```python
from openmed.clinical.exporters import check_codeable_concept

findings = check_codeable_concept(
    concept,
    expected_system="http://loinc.org",
)
assert findings == []
```

For an opt-in, offline check of profiles declared in `meta.profile`, including
post-de-identification comparison, see
[WHO SMART Guidelines Profile Checks](./fhir-smart-guidelines.md).

## Base R4 Structural Validation

Use `validate_resource()` or `validate_bundle()` before handing an OpenMed
export to a FHIR server. Both functions run entirely offline against a bundled,
minimal table of base FHIR R4 (4.0.1) cardinalities, datatypes, and small fixed
required bindings:

```python
from openmed.clinical.exporters.fhir import validate_bundle, validate_resource

resource_result = validate_resource(
    {
        "resourceType": "Observation",
        "status": "final",
        "code": {"text": "synthetic measurement"},
    }
)
assert resource_result.is_valid

bundle_result = validate_bundle(
    {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "synthetic measurement"},
                }
            }
        ],
    }
)
assert bundle_result.errors[0].location == "Bundle.entry[0].resource.status"
```

`ValidationResult.errors` and `.warnings` contain immutable
`ValidationFinding` objects with `severity`, `location`, `message`, and a FHIR
issue `code`. Messages describe structure only and never quote resource values.
Results also expose `.issues`, so `from_validation_result(result)` can render a
standard R4 `OperationOutcome`.

The bundled subset covers the resources OpenMed emits: `Condition`,
`Observation`, `MedicationRequest`, `MedicationStatement`, `Procedure`,
`DiagnosticReport`, `AllergyIntolerance`, `Immunization`, and `Encounter`. A
different resource type produces a `not-supported` warning rather than a false
conformance claim. The constraint table contains only OpenMed's compact
derivation of CC0-licensed base R4 structure and fixed code-system metadata; it
does not include clinical terminology content, proprietary profiles, or
implementation-guide packages.

## US Core STU9 Conformance

Use `check_us_core()` for the bundled US Core 9.0.0 subset covering exported
`Condition`, laboratory `Observation`, `MedicationRequest`, and
`AllergyIntolerance` resources. It always runs base R4 validation first, reports
missing must-support elements as warnings, and reports required cardinality or
locally enumerable binding violations as errors:

```python
from openmed.clinical.exporters.fhir import check_us_core

result = check_us_core(
    {
        "resourceType": "Condition",
        "category": [
            {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/condition-category"
                        ),
                        "code": "problem-list-item",
                    }
                ]
            }
        ],
        "code": {"text": "synthetic condition"},
        "subject": {"reference": "Patient/synthetic"},
    }
)
assert result.is_valid
```

Pass a supported canonical URL or StructureDefinition id as `profile` to select
the encounter-diagnosis `Condition` profile or override the resource default.
The checker resolves supported `meta.profile` declarations as well. Its compact
constraint table is OpenMed-authored from the
[US Core STU9 definitions](https://hl7.org/fhir/us/core/STU9/) and does not
bundle clinical terminology expansions. Use the full HL7 validator or the
receiving server for complete invariants and terminology validation.

This base validator is intentionally distinct from `check_bundle()`. The latter
loads caller-supplied `StructureDefinition` and `ValueSet` resources to check
declared implementation-guide profiles. Neither checker contacts a terminology
server or replaces the complete HL7 validator for invariants, extensions, and
full profile conformance.
