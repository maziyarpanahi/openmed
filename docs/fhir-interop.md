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

## Governed write AuditEvents

`to_governed_write_audit_event()` projects already-classified create/update
attempts from a `GovernedWriteAuditAttempt`. The projection is local: it returns
an R4 AuditEvent for the caller to retain or submit through a separately
reviewed governed path. It reads no clinical payload, resolves no credentials,
contacts no server and installs no ledger storage or write adapter.

| Controller outcome | Create / update action | R4 audit outcome | Controlled subtype |
| --- | --- | --- | --- |
| Success | `C` / `U` | `0` | `success` |
| Server rejection | `C` / `U` | `4` | `server_rejected` |
| Commit unknown | `C` / `U` | `8` | `commit_unknown` |
| Policy denial | `C` / `U` | `4` | `policy_denied` |
| Admission refusal | `C` / `U` | `4` | `admission_refused` |

For `commit_unknown`, outcome `8` records the controller's failure to confirm
the attempt. The subtype retains target-effect uncertainty. It is not proof
that a clinical write failed or is absent: reconcile durable effects before
any retry, using the [workflow recovery contract](agent/workflow-recovery.md).
Admission and policy refusals can be recorded before review or materialization
without inventing a reviewer, receipt or target reference.

This runnable synthetic refusal has no approval or clinical input. Its fixed
clock makes repeated projection deterministic and prints only audit codes:

```python
# Runnable: synthetic local audit projection; no request or clinical payload.
from datetime import datetime, timezone

from openmed.clinical.exporters.fhir import (
    GovernedWriteAuditAttempt,
    GovernedWriteOutcome,
    to_governed_write_audit_event,
)

attempt = GovernedWriteAuditAttempt(
    action="create",
    outcome=GovernedWriteOutcome.POLICY_DENIED,
    software_agent_ref="urn:uuid:00000000-0000-4000-8000-000000000001",
)
instant = datetime(2026, 1, 2, tzinfo=timezone.utc)
event = to_governed_write_audit_event(attempt, clock=lambda: instant)
assert event == to_governed_write_audit_event(attempt, clock=lambda: instant)
print({"action": event["action"], "outcome": event["outcome"]})
```

Inputs are closed action/outcome enums, opaque references, an optional controlled
reviewer category and a receipt digest. Reviewer categories are `reviewer`,
`clinical_reviewer`, `privacy_reviewer`, `operations_reviewer` and
`security_reviewer`. Trusted code explicitly maps its verified policy role to
one category; that classification does not replace the policy role or identify
a person. Role and digest must be supplied together. Success, server rejection
and unknown commit require that pair and at least one target/proposal reference.
The projection does not verify signatures, receipt freshness, permissions,
reviewer identity or actual server effects. The application must derive those
assertions from its governed controller and protected evidence.

Software/target references accept canonical lowercase UUIDv4 URNs only. Direct
clinical IDs, private URLs, free text, query strings, fragments and unsupported
UUID shapes are refused. Target references are sorted and deduplicated, with a
128-input limit. UUID syntax is not proof of opaque provenance, unlinkability
or reference resolution: the caller owns the protected mapping, disclosure
policy and resolution context. Treat even opaque audit metadata as controlled
information. A receipt digest must have `sha256:` followed by 64 lowercase hex
characters; it links to local evidence and creates no approval.

The software Device appears as the initiating agent and source observer. A
supplied reviewer category appears in a separate agent without a person
reference or name. Resources contain controlled coding, technical recorded
time, opaque references and optional receipt digests. There are no source
payloads, narrative, diagnoses, diagnostics, credentials, endpoints, reviewer
names or raw exceptions. Error messages contain only fixed projection codes.

The injected clock is called once and must return an aware datetime. Recorded
time is normalized to UTC with microsecond precision. Identical metadata and
clock values produce identical resources and IDs; retain the original recorded
instant when reproducing an event. These IDs bind the projection's metadata,
not authority or durable commit proof. Each call returns a fresh resource.

The [FHIR R4 AuditEvent structure](https://hl7.org/fhir/R4/auditevent.html),
[action codes](https://hl7.org/fhir/R4/valueset-audit-event-action.html) and
[outcome codes](https://hl7.org/fhir/R4/codesystem-audit-event-outcome.html)
define the interoperability shape. Local structural and synthetic composition
checks are engineering evidence, not target-server or clinical validation.

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
