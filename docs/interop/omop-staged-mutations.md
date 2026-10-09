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

## Explicit SQLite transactional adapter

`SQLiteOmopBatchCommitter` implements the batch protocol for the exact eleven
loader-owned SQLite tables. It supports complete-row inserts, non-key updates
and tombstones within a vocabulary-checked batch. Custom tables, explicit
extra references, incomplete inserts, primary-key updates, different DDL,
owned-table triggers and external inbound foreign keys are refused. Batch and
target snapshots are limited to 10,000 rows; object inventory is limited to 256.
It uses parameterized values and quotes identifiers from the checked schema.

Provision the target explicitly with the existing `create_omop_schema` and
`initialize_omop_commit_metadata(connection, target_snapshot)`. Initialization
requires a caller-owned connection outside a transaction and adds only receipt
and vocabulary metadata. It refuses an existing different snapshot rather than
overwriting it. The operator attests vocabulary versions and retirement state
through this protected metadata; the loader's SQLite catalog has no independent
vocabulary-version or retirement column. The adapter also checks actual proposed
target vocabulary identifiers and standard-concept flags against the snapshot.

```python
from openmed.interop.omop import (
    SQLiteOmopBatchCommitter,
    preview_omop_database,
)

packet = preview_omop_database(
    connection,
    batch,
    vocabulary_snapshot=target_snapshot,
    vocabulary_mappings=target_mapping_provenance,
    rollback_manifest=rollback_manifest,
)
```

Previewing takes a coherent read snapshot and hashes complete row values in
private memory. The outer `packet.preview_digest` binds that row state, target
schema, vocabulary-gate report, rollback manifest and inner batch preview. The
approval service must independently review and authorize this outer digest as
well as the inner batch preview. Binding only row identities cannot authorize
this adapter: changing a value under the same identity invalidates the review.

```python
adapter = SQLiteOmopBatchCommitter(
    batch,
    packet,
    vocabulary_snapshot=target_snapshot,
    vocabulary_mappings=target_mapping_provenance,
    rollback_manifest=rollback_manifest,
    connection_factory=open_protected_sqlite_connection,
    authorize=verify_receipt_and_both_review_bindings,
    admit=check_current_workflow_admission,
    rollback_ready=verify_protected_rollback_custody,
)
result = adapter.submit(approval_binding)
```

All three verifiers are required trusted integrations and must return exactly
`True`. Authorization verifies receipt authenticity, reviewer authority,
expiry, single-use issuance and both independently reviewed digests. It checks
an already issued receipt rather than consuming a bearer token on each call.
Admission composes the deployment's default-off controller and emergency stop.
Custody verifies durable protected rollback artifacts; a structurally valid
manifest alone is insufficient. The adapter rechecks these guards before and
during the effects and before commit. There is no default authorization or
admission implementation. Injected callbacks and the SQLite factory are trusted
code; Python types do not provide authentication or storage isolation.

`BEGIN IMMEDIATE` serializes submissions. Before writing, the adapter compares
the fresh complete preview with the reviewed packet, checks vocabulary and
references, then applies all mutations and the value-free receipt in one
transaction. Final loader and foreign-key checks reject invalid resulting rows.
Authorized duplicate submissions use the same approval-receipt digest and
return the identical stored result without replaying mutations, including
concurrent submissions. A changed batch or preview cannot reuse that receipt.
The unchanged vocabulary gate still rejects empty mapping collections; batches
must carry the exact provenance of inserted or updated clinical targets.

Results expose `committed`, `conflict`, `denied`, `failed` or `unknown`, closed
codes and digests. `proposed_count` always describes the proposal; `applied_count`
is `None` for unknown outcomes. A failed commit acknowledgement stays unknown
even when a subsequent rollback call succeeds. Call `adapter.recover(binding)`
to read the exact durable receipt with SQLite query-only mode. Recovery performs
no replay. A missing/unavailable receipt remains unknown rather than asserting
that the transaction never committed. Do not automatically resubmit uncertain
work; reconcile through the approval and recovery workflow.

The original `batch.commit(adapter, approval=binding)` protocol has only success
and failure outcomes. Use `submit` to retain the richer states; after a protocol
failure, `adapter.last_result` preserves any unknown outcome. The adapter disables
SQLite trace callbacks before queries and never copies SQL parameters, patient
values, connection details or driver exception text into results or receipts.
Its factory supplies a fresh connection for each operation, which the adapter
closes. Database, journal, receipt and rollback storage require the deployment's
normal protection, retention and independent backup controls. Restoring or
tampering with the database and its receipt ledger together defeats local
idempotency evidence; use an external protected checkpoint when that threat is
in scope. Offline synthetic transaction checks are not clinical validation.
