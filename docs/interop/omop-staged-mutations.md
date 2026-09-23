# Staged OMOP mutations

`openmed.interop.omop.mutation_batch` stages OMOP inserts, updates, and
tombstones as one ordered, local batch. It separates the values needed by a
trusted local writer from the value-free metadata a reviewer or audit store
may retain.

The module is deterministic and does not perform network or database I/O. A
deployment supplies the row identities visible to the preview and an atomic
local committer.

## Build an ordered batch

Use the factory matching each intended operation. Supported OpenMed OMOP
tables derive their primary key from the row. A custom table must pass an
explicit `key`.

```python
from openmed.interop.omop.mutation_batch import (
    OmopMutation,
    OmopMutationBatch,
    OmopRowKey,
)

existing_rows = (
    OmopRowKey("person", {"person_id": 101}),
)
batch = OmopMutationBatch(
    (
        OmopMutation.insert(
            "visit_occurrence",
            {
                "visit_occurrence_id": 201,
                "person_id": 101,
                "visit_source_value": "synthetic-source",
            },
        ),
        OmopMutation.update(
            "visit_occurrence",
            {"visit_occurrence_id": 201},
            {"visit_source_value": "synthetic-revised-source"},
        ),
    )
)
```

Rows are applied in tuple order. Inserts make their identity available to later
mutations; updates and tombstones require a live target. Foreign keys for the
OMOP tables emitted by OpenMed are inferred from changed fields. Custom schemas
can declare additional `OmopRowKey` objects through the `references` argument.

## Preview and reference checks

Previewing does not call the committer and does not send a mutation to a target
system:

```python
preview = batch.preview(existing_rows=existing_rows)
if not preview.is_valid:
    for issue in preview.issues:
        handle_closed_reason(issue.code)
```

The preview contains:

- the ordered operation type, table, and changed field names;
- per-row, batch, existing-reference-snapshot, and preview digests;
- counts by operation; and
- stable referential issue codes with mutation ordinals.

It does **not** contain primary-key values, source values, before/after values,
or adapter exception text. `repr()` for batches, mutations, and row identities
also omits values. Stable digests are still sensitive metadata and should use
the same access controls and retention limits as other clinical audit records.

`existing_rows` is an explicit local snapshot. The batch checks that snapshot
plus preceding staged mutations; it does not discover database rows or inbound
references on its own. A deployment should supply every relevant parent
identity and have its committer revalidate the snapshot before writing.

## Bind review approval

The approval boundary accepts only a digest of a value-free approval receipt.
Token issuance, reviewer-role checks, expiry, and single-use enforcement belong
to the approval service rather than this OMOP module.

```python
approval = batch.bind_approval(
    preview,
    approved_preview_digest=preview.preview_digest,
    approval_receipt_digest="sha256:" + "a" * 64,
)
```

Binding fails closed if reference checks failed, the batch changed, the
reviewed preview digest differs, or either digest has an invalid shape. An
approval integration should supply the reviewed digest independently; copying
it directly from an unreviewed preview is not human approval.

## Commit atomically

A committer implements one small local protocol:

```python
class LocalCommitter:
    def commit_batch(self, mutations, *, batch_digest, approval):
        with local_database_transaction() as transaction:
            revalidate_reference_snapshot(
                transaction,
                approval.reference_snapshot_digest,
            )
            for mutation in mutations:
                transaction.apply(mutation)


result = batch.commit(LocalCommitter(), approval=approval)
```

The committer must apply all mutations atomically or raise without leaving a
partial commit. A successful result reports `committed` and the number of
mutations. Adapter exceptions are converted to `committer_error`; exception
text is never copied into the result. The result contains digests and counts,
not row values.

This API does not certify an OMOP deployment or authorize autonomous clinical
decisions. Keep the reviewer, transaction, vocabulary-snapshot, and rollback
controls required by the surrounding workflow.
