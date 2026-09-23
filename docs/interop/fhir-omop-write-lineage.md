# FHIR-to-OMOP write lineage

One FHIR element can produce several OMOP rows, and several elements can feed
one row. `openmed.interop.lineage.fhir_omop_writes` records that many-to-many
relationship before a staged OMOP mutation batch is approved.

The crosswalk is deterministic and performs no network or database access. It
stores hashed resource identities, FHIR element paths, transform rule names,
OMOP row digests, and digest-only vocabulary evidence. It does not retain FHIR
values, resource identifiers, OMOP row values, concept identifiers, or version
strings in reviewer reports. Digests remain sensitive metadata and need the
same access controls and retention limits as other clinical audit records.

## Build the crosswalk

Create a reference for every source element that the transformation consumed.
Use the digest already assigned to the source resource by the local FHIR
pipeline; do not put the original resource ID in the crosswalk. The bridge's
bare SHA-256 hashes are normalized to the canonical `sha256:` form.

```python
from openmed.interop.lineage import (
    FhirElementReference,
    FhirOmopLineageLink,
    FhirOmopWriteLineage,
    VocabularyLineageEvidence,
)

patient_id = FhirElementReference(
    "sha256:" + "a" * 64,
    "Patient.id",
)
observation_code = FhirElementReference(
    "sha256:" + "b" * 64,
    "Observation.code",
)

evidence = VocabularyLineageEvidence.from_mapping(
    mapping_provenance,
    snapshot_digest=target_snapshot.snapshot_digest,
)
lineage = FhirOmopWriteLineage(
    (
        FhirOmopLineageLink(
            (patient_id, observation_code),
            "fhir.measurement.v1",
            tuple(mutation.row_digest for mutation in batch.mutations),
            vocabulary_evidence=(evidence,),
        ),
    )
)
```

A link accepts several sources and several target row digests. This represents
expansion and collapse directly rather than forcing a one-to-one audit model.
Rows that write a non-zero OMOP concept field must have at least one linked
`VocabularyLineageEvidence` record.

## Verify coverage and information loss

The caller supplies the complete inventory of FHIR elements that must be
accounted for:

```python
report = lineage.verify(
    batch,
    required_sources=(patient_id, observation_code),
)
if not report.is_approvable:
    send_to_local_review(report.to_dict())
```

Verification fails closed when:

- a required source element has no lineage link;
- a staged row has no source link;
- a link names a row outside the staged batch;
- a row with a non-zero concept field lacks vocabulary evidence; or
- a link declares a lossy transformation.

Represent an intentionally dropped or degraded element with `loss_reason`.
A link without target rows must declare such a closed reason code. Loss is
reported rather than silently accepted, and this API does not implement a
waiver or autonomous clinical decision policy.

```python
FhirOmopLineageLink(
    (unsupported_element,),
    "fhir.drop_unsupported.v1",
    (),
    loss_reason="unsupported_element",
)
```

## Bind approval and commit

Use `bind_approval` instead of approving the staged batch directly. It reruns
lineage verification and returns a binding that carries the crosswalk, source
inventory, and report digests alongside the ordinary OMOP approval.

```python
approval = lineage.bind_approval(
    batch,
    preview,
    required_sources=(patient_id, observation_code),
    approved_preview_digest=reviewed_preview_digest,
    approval_receipt_digest=receipt_digest,
)
result = approval.commit(batch, local_committer)
```

Approval fails before the batch approval is created if lineage is incomplete
or lossy. The committer still receives the existing value-bearing mutations
only inside the trusted local boundary. This helper does not certify an OMOP
deployment, authorize an autonomous write, or replace vocabulary-snapshot and
transaction controls.
