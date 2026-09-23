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
