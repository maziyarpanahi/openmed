# Clinical chart-abstraction evidence

`openmed.agent.workflows` provides deterministic, field-level evidence chains
for reviewable chart abstraction. A chain binds one developer-authored field
identifier to source spans, a normalized-fact digest, the digest and kind of
the rule or model that produced it, an uncertainty score, and a human-review
state.

The contract deliberately excludes source text and normalized clinical values.
It performs no filesystem, database, or network access. Digests and offsets
remain sensitive metadata and should receive the same access controls and
retention limits as other clinical audit records.

## Build an evidence chain

Use a digest of the canonical source artifact plus character offsets to locate
evidence without copying clinical text into the chain. Use `generated_text`
only when recording a supplemental generated span; generated text cannot be
the sole evidence for finalization.

```python
from openmed.agent.workflows import (
    AbstractionEvidenceChain,
    ChartAbstractionEvidence,
    ReviewerState,
    SourceLocation,
    TransformationKind,
)

chain = AbstractionEvidenceChain(
    field_id="registry.primary_diagnosis",
    source_locations=(
        SourceLocation(
            source_digest="sha256:" + "a" * 64,
            start_offset=120,
            end_offset=148,
        ),
    ),
    normalized_fact_digest="sha256:" + "b" * 64,
    transformation_kind=TransformationKind.RULE,
    transformation_digest="sha256:" + "c" * 64,
    uncertainty=0.05,
    reviewer_state=ReviewerState.APPROVED,
)

evidence = ChartAbstractionEvidence((chain,))
receipt = evidence.finalize(("registry.primary_diagnosis",))
```

Field identifiers describe registry schema fields, not patient- or
organization-derived values. Fact digests bind values held within the trusted
local abstraction boundary; they are not a substitute for encrypted clinical
storage.

## Produce chains from Journey records

`build_journey_abstraction_evidence()` is a Python producer over existing
`JourneySnapshot`, `ClinicalFact`, `EvidenceLocator`, `ClinicalArtifact`, and
`ConflictSet` contracts. Supply immutable records read at the snapshot revision,
including every relevant conflict. The producer performs no storage reads,
extraction, clinical inference, or network calls. This issue's producer boundary
is scoped to the existing Python Journey and abstraction contracts.

```python
from openmed.agent.workflows import (
    AbstractionFieldBinding,
    AbstractionReviewReceipt,
    ReviewerState,
    SourceKind,
    TransformationKind,
    build_journey_abstraction_evidence,
)

# These records come from the application's trusted local snapshot read.
inputs = dict(
    snapshot=snapshot,
    fields=(AbstractionFieldBinding(
        "registry.primary_diagnosis", (fact.fact_id,), TransformationKind.MODEL
    ),),
    facts=(fact,),
    locators=(locator,),
    artifacts=(artifact,),
    conflicts=conflicts,
    source_kinds={artifact.artifact_id: SourceKind.CLINICAL_RECORD},
)
pending = build_journey_abstraction_evidence(**inputs)

# Only after explicit human approval inside the trusted review application:
review = AbstractionReviewReceipt(
    "registry.primary_diagnosis",
    pending.chains[0].chain_digest,
    ReviewerState.APPROVED,
)
reviewed = build_journey_abstraction_evidence(**inputs, review_receipts=(review,))
receipt = reviewed.finalize(("registry.primary_diagnosis",))
```

The caller declares source origin and rule/model kind; arbitrary artifact types,
fact statuses and Journey review labels cannot prove either origin or approval.
Unknown origin blocks instead of defaulting to a clinical record. Mark generated
artifacts with `SourceKind.GENERATED_TEXT`. A generated-only chain cannot finalize.
Parent facts are not recursively treated as clinical source evidence; a derived
fact must retain its own direct clinical text locators.

The normalized-fact digest commits to the complete immutable fact record, without
copying its value into output. The transformation digest commits to the snapshot,
fact derivation hash and ordered digests of the locator/artifact metadata, including
locator transformations and artifact derivation metadata. Character offsets are
copied exactly from half-open `text_span` locators. No byte-to-character conversion
or text reconstruction is attempted. The caller must ensure those offsets refer
to the canonical artifact identified by its content hash. Missing confidence maps
to uncertainty 1.0; otherwise uncertainty is `1 - confidence`.

Review receipts bind the **pending** chain digest, so changes to facts, snapshot,
offsets, source origin, derivations or transformations invalidate earlier approval.
Receipts carry explicit pending/approved/rejected decisions. They are an adapter
input from the trusted local review boundary, not authenticated credentials or
review UI. The producer does not manufacture approval or infer it from fact status.
The application must authenticate review decisions and prevent stale or revoked
receipts from being supplied.

Coverage blockers persist on the returned `ChartAbstractionEvidence`; calling its
`finalize()` directly cannot discard a failed producer check. Reports use the
existing field/count/digest contract with these distinct codes:

| Code | Trigger |
| --- | --- |
| `missing_field_evidence` | Required field has no chain, binding has no candidates, or a mapped fact is absent |
| `missing_locator_evidence` | Any evidence ID of a mapped fact is unresolved |
| `non_text_locator_evidence` | Any locator is not a text span, even if another span is valid |
| `missing_artifact_evidence` | Text locator's artifact is absent |
| `conflicting_facts` | Multiple candidate facts for one field or a mapped fact in an open conflict |
| `derived_only_evidence` | Parent-derived fact has no direct clinical source span |
| `source_kind_undeclared` | Artifact origin has no explicit declaration |
| `subject_mismatch` | Fact or source artifact belongs to a different subject |
| `review_receipt_mismatch` | Receipt does not bind the current pending chain |

Existing `missing_source_evidence`, `generated_only_evidence`, and
`review_not_approved` gates still apply. Multiple candidates are conservatively
treated as conflicting even if their values agree; callers must resolve the
candidate selection explicitly. Duplicate input IDs are rejected with value-free
errors; repeated locators for an identical source span are checked individually
and deduplicated in the chain. All blockers apply to represented fields, including
fields outside the required set.

Chains, receipts, reports and their representations retain only digests, offsets,
developer-authored field IDs, controlled codes/states/kinds and counts or scores.
They retain no values, raw artifact IDs, source text, locator payloads, paths,
attributes or extensions. Synthetic fixtures prove contract behavior, not
clinical accuracy or release readiness. Extraction, review UI and registry/FHIR
export remain separate work.

## Finalization gates

`evaluate()` returns a deterministic metadata-only report. `finalize()` returns
a content-free receipt only when that report passes. Finalization fails closed
when:

- a required field has no evidence chain;
- a represented field has no source span;
- every source span for a field is generated text; or
- a represented field is pending or rejected by the reviewer.

```python
report = evidence.evaluate(("registry.primary_diagnosis", "registry.stage"))
if not report.is_finalizable:
    send_to_local_review(report.to_dict())
```

Reports expose only closed issue codes, developer-authored field identifiers,
counts, and digests. They do not include source digests, source offsets, fact
digests, transformation digests, or submitted values. Validation errors use
stable field names and codes and never echo rejected input.

This evidence contract does not certify an abstraction, authorize an
autonomous clinical decision, or replace local policy and reviewer controls.
