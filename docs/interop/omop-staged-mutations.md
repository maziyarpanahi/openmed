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

## Stage NLP loader output with span lineage

`stage_nlp_omop_tables` bridges `load_grounded_notes` output into the existing
batch, vocabulary and rollback contracts. It makes an immutable proposal from
caller-supplied rows, without database discovery, file writes or network access.

```python
from openmed.interop.omop import stage_nlp_omop_tables

staged = stage_nlp_omop_tables(
    loaded_tables,
    pipeline_digest=attested_pipeline_digest,
    vocabulary_snapshot=target_snapshot,
    vocabulary_mappings=target_mapping_provenance,
)

# Trusted local custody stores protected payloads and returns independently
# computed digests using the strategies in staged.rollback_requirements.
instructions = protected_custody.store_rollback_material(staged)
manifest = staged.build_rollback_manifest(instructions)
packet = staged.preview(manifest)
```

Each `NOTE_NLP`, domain and source-mapping mutation needs exactly one lineage
record, bound to its ordinal, closed table name and complete row digest. Records
contain the caller-declared source-note digest, a digest computed from the
current note text, Python character offsets and hashes of the NLP system and
pipeline. Source identity and current text hashes can differ after prior
transformations. A supplied pipeline digest is an attestation by the caller;
the bridge does not prove which model actually ran. Normalized lexical variants
are allowed, while offsets must identify a non-empty span in the current text.

The loader's structural validator checks complete row shapes, concept/FK
references and bidirectional domain-event links. Staging also checks patient,
visit, source-hash and concept consistency and requires one mapping row per
span. Rows are bounded to 10,000 per snapshot and per proposed batch. Unexpected
columns and invalid inputs produce closed error codes without row values.
Public failures discard private exception contexts, and preview operation
counts require exact built-in scalar types rather than caller-defined aliases.

`replace_by_note` requires an explicit complete `existing_tables` snapshot.
Replacement is scoped to patient plus source-note identity, so another patient
with the same declared note hash is preserved. Child rows are tombstoned before
their old note; incoming rows are then inserted in parent-first order. Identical
concept/person/visit parents are reused. Conflicting parent values fail instead
of becoming an implicit update. For every removed `NOTE_NLP` row, supply its
original pipeline digest through `existing_pipeline_digests`, keyed by
`OmopRowKey("note_nlp", {"note_nlp_id": old_id}).digest`. Removed rows retain the
old provenance. Missing or extra old pipeline evidence fails closed.

The existing vocabulary write gate must accept every inserted target concept
and its declared target vocabulary version. Missing/extra mapping evidence,
unmapped targets, retired concepts and snapshot drift block approval; empty
mapping collections remain rejected by the existing policy. The review packet
contains gate-result hashes, counts and compatibility flags rather than
vocabulary names or versions. Rejected input-span counts are reported separately
and do not imply clinical completeness or validation.

`staged.rollback_material(ordinal)` returns a fresh **private** payload containing
row keys and, for tombstones, exact before-images. Keep it in protected local
custody; never log it or include it in an audit artifact. The expected digests in
`rollback_requirements` are requirements, not proof that durable artifacts
exist. Custody must independently store/hash the canonical JSON bytes
(`ensure_ascii=False`, `sort_keys=True`, `separators=(",", ":")`,
`allow_nan=False`, UTF-8) and attest readiness. The manifest builder verifies
exact strategy, payload digest and mutation coverage using the existing rollback
contract. It does not create files or certify storage isolation or durability.

When `packet.is_approvable` is true, a separate approval service reviews its
`preview_digest` and issues a receipt. Bind that independently reviewed digest
with `staged.bind_approval(packet, manifest, approved_preview_digest=reviewed_digest,
approval_receipt_digest=receipt_digest)`. Reviewer authority, receipt expiry and
single-use enforcement remain the approval service's responsibility.

Four optional batch evidence digests bind the full lineage crosswalk, target
vocabulary snapshot, mapping provenance and rollback requirements. The outer
packet also binds the actual rollback manifest. Changing those inputs requires
a new review, including when the staged row values stay identical. Ordinary
`OmopMutationBatch` callers using no evidence retain their existing digest and
wire representation. This bridge adds no committer: a deployment must still
revalidate database identities and all relevant inbound references, enforce its
approved workflow and use an atomic transactional adapter at the effect boundary.

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

When composing this adapter with `stage_nlp_omop_tables`, preserve the reviewed
lineage and protected rollback custody, then obtain a fresh database preview.
The host approval service must independently review and bind both the complete
staging packet and the database outer preview. A receipt that approves only the
staging digest does not authorize database effects. Bind the database packet's
inner batch preview for submission; an existing stage binding is usable only
when its inner reference state agrees and the host also authorizes the exact
current database outer preview. Digests and local fixture callbacks do not
establish reviewer authority, protected storage or production qualification.

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
Public provisioning, preview and constructor failures also raise fresh closed
errors without retaining private driver or iterator exception chains. Callback
error codes outside the declared vocabulary become `transaction_failed`.
Its factory supplies a fresh connection for each operation, which the adapter
closes. Database, journal, receipt and rollback storage require the deployment's
normal protection, retention and independent backup controls. Restoring or
tampering with the database and its receipt ledger together defeats local
idempotency evidence; use an external protected checkpoint when that threat is
in scope. Offline synthetic transaction checks are not clinical validation.
