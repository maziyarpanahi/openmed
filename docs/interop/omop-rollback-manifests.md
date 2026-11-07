# OMOP rollback manifests

Every staged OMOP mutation batch should have a rollback manifest before its
preview reaches human approval. The manifest proves structural coverage: each
mutation has an operation-compatible rollback strategy and is bound to a
protected local rollback artifact. It does not execute SQL, inspect a database,
or prove that a rollback is semantically correct.

`openmed.interop.omop_rollback_manifest` is deterministic and offline. It
retains table names, operation counts, ordered digests, and rollback strategies.
It never accepts row values, patient identifiers, SQL, or connection strings.

## Build complete coverage

Keep the actual rollback artifact inside the trusted local boundary. Pass only
its SHA-256 digest into an instruction:

```python
from openmed.interop.omop import (
    OmopMutation,
    OmopMutationBatch,
    VocabularySnapshot,
)
from openmed.interop.omop_rollback_manifest import (
    OmopRollbackInstruction,
    RollbackStrategy,
    build_omop_rollback_manifest,
)

batch = OmopMutationBatch(
    (
        OmopMutation.insert("person", locally_prepared_person_row),
        OmopMutation.update(
            "visit_occurrence",
            locally_selected_visit_key,
            locally_prepared_changes,
        ),
    )
)
snapshot = VocabularySnapshot(local_versions, local_concepts)

manifest = build_omop_rollback_manifest(
    batch,
    snapshot,
    (
        OmopRollbackInstruction(
            mutation_ordinal=0,
            strategy=RollbackStrategy.DELETE_INSERTED_ROW,
            rollback_artifact_digest=insert_rollback_digest,
        ),
        OmopRollbackInstruction(
            mutation_ordinal=1,
            strategy=RollbackStrategy.RESTORE_BEFORE_IMAGE,
            rollback_artifact_digest=update_rollback_digest,
        ),
    ),
)
```

Each mutation ordinal must appear exactly once. Missing, duplicate, and unknown
ordinals fail closed. The strategy is fixed by the staged operation:

| Staged operation | Required rollback strategy |
| --- | --- |
| `insert` | `delete_inserted_row` |
| `update` | `restore_before_image` |
| `tombstone` | `reinsert_tombstoned_row` |

The artifact digest is a binding, not storage. Deployments must protect the
corresponding key or before-image artifact according to their clinical data
policy and must not copy its contents into the manifest.

## Deterministic rollback order

Instructions can arrive in any order. Manifest entries are always emitted in
reverse mutation order so that dependent writes unwind last-in, first-out.
Table summaries are ordered by table name, operation counts are ordered by
operation name, and canonical JSON uses stable key ordering. The manifest
digest therefore changes if any covered batch, artifact digest, strategy, or
vocabulary snapshot changes.

Before submitting the batch and manifest for approval, validate their binding:

```python
manifest.validate(batch, snapshot)
approval_preview = batch.preview(existing_rows=local_row_snapshot)
```

Validation fails if the batch digest, table or operation counts, rollback
coverage or ordering, manifest digest, or vocabulary snapshot digest differs.
This prevents a manifest prepared against one vocabulary release from being
reused after the target snapshot changes.

## Privacy and operational boundary

`manifest.to_json()` contains only schema metadata, table names, counts,
strategies, ordinals, and SHA-256 digests. Digests can still be sensitive audit
metadata; apply the same access controls and retention policy used for staged
mutation previews.

The module does not execute SQL, manage a database transaction, retain
before-images, verify that an external artifact is available, or establish
semantic reversibility. The local committer remains responsible for those
controls and for atomically applying or rolling back the batch.
