# Relation evidence binding

Higher-risk relation aids must be independently reviewable. Before a relation
enters a summary or review workflow, bind it to:

- a directed head and tail source span;
- at least one source span for the linking evidence;
- an explicit assertion state; and
- an identifier for the source document.

`openmed.clinical.relations.evidence_binding` provides this boundary as a
deterministic, local-only adapter. It does not infer an assertion state or
invent evidence when a producer omits either field.

## Bind a candidate

```python
from openmed.clinical.relations.evidence_binding import (
    bind_relation_evidence,
)

candidate = {
    "relation_type": "medication_change",
    "document_id": "synthetic-relation-document",
    "head": {"start": 0, "end": 8, "label": "MEDICATION"},
    "tail": {"start": 9, "end": 17, "label": "PROBLEM"},
    "evidence_spans": [{"start": 18, "end": 25, "label": "CONTEXT_CUE"}],
    "assertion_state": "affirmed",
    "score": 0.91,
}

bound = bind_relation_evidence(candidate)
payload = bound.to_dict()
```

The returned `GuardedRelation` contains only relation codes, offsets, controlled
assertion metadata, confidence, and opaque document/span identifiers. Raw
source text and arbitrary candidate metadata are not copied. A raw document
identifier is domain-separated and hashed before it is stored; an existing
`sha256:` or `hmac-sha256:` identifier is preserved.

Existing relation objects are accepted too. For example, a
`DocumentLevelRelation` can supply its `head`, `tail`, `score`, and
`evidence_sentence_offsets`; its assertion state must still be supplied
explicitly when it is not already present.

## Gate workflow input

Use `require_guarded_relations()` at the boundary of a summary or review
workflow. It rejects raw or incomplete records and returns a deterministic
ordering:

```python
from openmed.clinical.relations.evidence_binding import (
    require_guarded_relations,
)

review_records = require_guarded_relations((bound,), workflow="review")
summary_records = require_guarded_relations((bound,), workflow="summary")
```

Both workflow paths retain `requires_clinician_review=True` and
`autonomous_decision=False`. `EvidenceBindingError` messages are fixed field
or contract categories and never echo submitted identifiers, labels, or source
values.

The module uses only the Python standard library plus OpenMed's existing label
normalizer, performs no mandatory network call, and is an assistive provenance
guard rather than a diagnosis, treatment decision, compliance certification, or
clinical-device guarantee.

## Validation limits

Validation rebuilds typed spans and relation records before workflow entry. Unknown assertion-axis values and conflicting mapping aliases fail closed; they cannot become affirmed defaults. Collections are limited to 4096 entries, nested span/assertion structures to 32 levels, and individual strings to 1048576 characters. Identifiers are pseudonymous, not proof of anonymization. Relation codes are caller-declared schema metadata and must not contain patient data.

## Export reviewed relations to FHIR R4

`openmed.clinical.exporters.fhir.export_reviewed_relations()` projects accepted
guarded candidates and already exported endpoint resources into an offline
`collection` Bundle. It emits links, evidence offsets and digests without
copying source text. It performs no write, model load, terminology download or
server request.

Bind each candidate to relative `ResourceType/id` references and explicit
`patient`, `family` or `other` attribution using `ReviewedFHIRRelation`.
`relation_fhir_review_fingerprint()` returns the `sha256:` fingerprint for the
existing [review state machine](review-transitions.md). The fingerprint
binds the guarded evidence, endpoint content and identities, active Patient,
attribution, optional family record and permitted terminology systems. Store
that fingerprint in the existing review workflow's transitions. Export requires
a valid history ending in approval from `in_review`, with every transition
bound to that fingerprint. Changes require review again. The exporter does not
create approvals or authenticate reviewers; caller-supplied history is local
workflow evidence and grants no FHIR write authority.

```python
from openmed.clinical.exporters.fhir import (
    ReviewedFHIRRelation,
    export_reviewed_relations,
    relation_fhir_review_fingerprint,
)

# These inputs come from the caller's extraction, fact export and review UI.
candidate = ReviewedFHIRRelation(
    relation=bound,
    head_reference="Condition/condition-1",
    tail_reference="MedicationStatement/medication-1",
    experiencer="patient",
)
fingerprint = relation_fhir_review_fingerprint(
    candidate, endpoint_resources, "Patient/patient-1"
)
# Save fingerprint with the existing review decision. Only after approval:
reviewed_candidate = ReviewedFHIRRelation(
    relation=candidate.relation,
    head_reference=candidate.head_reference,
    tail_reference=candidate.tail_reference,
    experiencer=candidate.experiencer,
    review_transitions=approved_review_transitions,
)
result = export_reviewed_relations(
    [reviewed_candidate], endpoint_resources,
    patient_reference="Patient/patient-1",
)
```

The snippet describes application integration; the endpoint resources and
approved transitions are caller inputs. It does not silently approve a
candidate. Each endpoint reference must identify a supplied resource belonging
to the active Patient. References in the resulting Bundle resolve to its own
entries and pass the existing [R4 reference-type checks](../fhir/reference-target-types.md).

| Candidate type | FHIR projection |
| --- | --- |
| `diagnosis_to_treatment` | Tail MedicationStatement or Procedure gets a reason reference to the head. |
| `procedure_to_indication` | Head Procedure gets a reason reference to the tail. |
| `drug_to_reason`, `drug_to_indication`, `medication_change` | Head MedicationStatement gets a reason reference to the tail. No dose or status change is inferred. |
| `condition_to_relative` | Inline condition code in FamilyMemberHistory; the head Condition is a code template and is never emitted as a patient Condition. |

For reason links, MedicationStatement accepts Condition, Observation and
DiagnosticReport targets; Procedure additionally accepts Procedure. A valid
R4 DocumentReference target is outside this projection's source-free endpoint
subset and becomes an `endpoint_invalid` loss. Laboratory results without an
explicit supported directed relation also become losses. No mapping is guessed
from proximity, label names or missing endpoints. These rules follow the
[R4 MedicationStatement](https://hl7.org/fhir/R4/medicationstatement-definitions.html)
and [R4 Procedure](https://hl7.org/fhir/R4/procedure-definitions.html) definitions.

Family export additionally requires a `FamilyHistoryRecord` whose condition
and relative offsets exactly match the guarded endpoints. Attribution must be
`family`, and the supplied condition code becomes
`FamilyMemberHistory.condition.code`, with `status="partial"` and a controlled
OpenMed family-role coding. An optional tail RelatedPerson must belong to the
same Patient; a known role with no supplied tail resource is allowed. Onset age,
vital status and relative names are omitted. The patient reference identifies
the person whose family history is recorded, not the owner of the condition.
An endpoint used for both an approved family condition and an approved patient
fact is refused with `attribution_conflict`. See the
[R4 FamilyMemberHistory definition](https://hl7.org/fhir/R4/familymemberhistory-definitions.html).

Only affirmed, confirmed or historical assertions can be projected. Unreviewed,
rejected, reopened, expired, negated, uncertain, hypothetical and ordinary
non-patient relations emit no links. `result.losses` reports input indices and
controlled codes such as `unreviewed`, `review_mismatch`, `assertion_refused`,
`endpoint_missing` and `family_binding_invalid`. Unsupported relation types,
duplicate candidates and malformed endpoint structures also produce losses.
Malformed overall input or technical clocks raise `RelationFHIRExportError`
with a fixed message. Arbitrary candidate labels and endpoint values never
appear in losses or exception messages.

### Projection scope and privacy

This is a code-and-link projection, not a complete clinical-record round trip.
It reconstructs only selected endpoints' resource type, opaque identity,
Patient scope, codes and required status. It strips names, narrative, coding
displays/text, identifiers, dates, values, attachments, notes, existing
unreviewed reason links and other fields. `omitted_endpoint_field_count` counts
discarded top-level fields on emitted endpoint resources; nested text and the
discarded family code template are not included in that count. Inputs are not
modified. Endpoints containing modifier extensions, refuted conditions or
entered-in-error status fail closed. Terminology systems and codes are trusted
caller metadata and must contain no patient data; syntax and an allowlist do
not establish that a code is clinically correct or anonymized. The default
system URIs identify terminologies but bundle no restricted terminology assets.

Each accepted relation emits Provenance with half-open head, tail and linking
offsets, the opaque source-document digest, review fingerprint and history
digest. The injected clock supplies only the technical `Provenance.recorded`
time. It never supplies a clinical event date. Resource identifiers are
pseudonymous digests, not proof of anonymization. Outputs remain assistive
clinical data requiring the caller's access controls.

The bounds are 512 relations, 128 review transitions per relation, 1024 input
resources, 4 MiB of serialized resource input, JSON depth 32 and 65,536 values.
Coding concepts allow at most 16 codings from at most 64 declared systems.
Local validation checks the bundled structural subset, including required
FamilyMemberHistory fields and condition codes, plus declared R4 reference
types. It does not certify implementation-guide profiles or complete FHIR
conformance. No additional core dependency is required.

Injected clocks and existing review-policy callbacks are trusted application
code; this boundary does not sandbox them.
