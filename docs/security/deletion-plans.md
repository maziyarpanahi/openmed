# Deletion impact plans

`openmed.risk.deletion_plan` provides a local, non-destructive preview before a
caller removes an OpenMed-managed cache, map, or evidence artifact. It is a
planning aid, not a compliance certification or a guarantee that every copy of
data is discoverable.

## Manifest shape

Pass a mapping with an `artifacts` list, a sequence of entries, or a `str` or
`Path` pointing to a local JSON manifest. Each entry contains an artifact hash,
a safe kind, a retention class, optional dependency links, and an ownership
flag:

```python
from openmed.risk.deletion_plan import plan_deletion_impact

cache_hash = "sha256:" + "0" * 64
map_hash = "sha256:" + "1" * 64

manifest = {
    "artifacts": [
        {
            "artifact_hash": cache_hash,
            "kind": "cache",
            "retention_class": "short",
            "dependencies": [],
            "owned": True,
        },
        {
            "artifact_hash": map_hash,
            "kind": "map",
            "retention_class": "standard",
            "dependencies": [cache_hash],
            "owned": True,
        },
    ]
}

plan = plan_deletion_impact(manifest, cache_hash)
print(plan.to_json())
```

`dependencies` point from an entry to the artifacts it needs. The planner
follows those links in reverse and transitively, so deleting the cache above
reports both the cache and the dependent map as affected. Opaque local
references are converted to SHA-256 digests at the input boundary. Existing
SHA-256 and HMAC-SHA-256 digests are preserved in canonical lowercase form.

## Safe dry-run output

Planning is always a dry-run. `to_dict()`, `to_json()`, and `to_markdown()`
contain counts grouped by artifact kind and retention class, plus integrity
digests and safety counters. They do not include source paths, raw identifiers,
unknown manifest fields, or individual resource values. The plan object can
provide canonical hash references to an injected local executor, but those
references are not emitted by the report serializers.

Unowned affected artifacts and unresolved dependency links are reported as
blocked safety counters. Execution refuses to proceed while either counter is
non-zero, including when an owned target has an unowned dependent.

## Bounded manifest handling

Local files, manifest entries, dependency links, deletion targets, and opaque
references have fixed size limits. JSON files reject duplicate object fields
and non-finite numbers. In-memory manifests reject ambiguous aliases and
unsupported field-name types. Cyclic, infinite, or hostile custom containers
fail with a closed `DeletionPlanError` that does not include input values or a
container-provided exception message.

## Explicit execution boundary

The planner never performs deletion. If an application has its own local
deletion implementation, it must inject that callback and provide the exact
confirmation token for the reviewed plan:

```python
from openmed.risk.deletion_plan import execute_deletion_plan

def delete_owned_artifact(artifact):
    # Resolve artifact.artifact_hash through the application's local registry.
    # Do not pass raw paths or data into the plan or its reports.
    ...

result = execute_deletion_plan(
    plan,
    confirmation=plan.confirmation_token,
    executor=delete_owned_artifact,
)
```

The callback receives only a normalized `DeletionArtifact`, and target
callbacks run in stable hash order. Dependents are included in the impact
preview but are not deleted automatically. Callers remain responsible for
retention policy, legal holds, access control, backups, and human review.

Execution revalidates the manifest digest, plan digest, target and impact sets,
counts, and ownership state immediately before invoking the first callback.

The module has no network dependency or mandatory outbound call. Use synthetic
offline fixtures for tests and never place protected health information,
credentials, or raw paths in a manifest report.

## Federated round purges

`openmed.training.federated_artifact_purge` applies these plans to
private-learning artifacts. Individual client updates can leak training text
even after clipping, and local runs leave tokenized caches and resume state
behind, so a round policy declares a retention class for every artifact class
and the purge runs as soon as the round no longer needs that material.

| Artifact class | Allowed disposition |
| --- | --- |
| `client_update_payload`, `masked_input`, `share_material` | `mandatory_purge` |
| `local_cache`, `resume_checkpoint`, `transcript` | `mandatory_purge` (default) or `best_effort_purge` |
| `commitment`, `evidence` | `preserve` |

`FederatedRetentionPolicy` requires a declaration for every class and refuses
any weaker disposition. The coordinator purges after the aggregate commitment
is recorded (`commitment_recorded`) or the round aborts (`round_aborted`). A
client purges after it submits its update (`update_submitted`) or aborts.

```python
from openmed.training.federated_artifact_purge import (
    FederatedArtifactPurge,
    FederatedPurgeSide,
    FederatedPurgeTrigger,
    FederatedRetentionPolicy,
    PrivateLearningArtifactClass,
    emit_federated_round_evidence,
)

registry = {}  # private digest -> local location, never serialized


def delete_owned_artifact(artifact):
    # Resolve the digest through the private registry and remove it.
    # Treat an already-missing artifact as deleted so restarts are idempotent.
    registry.pop(artifact.artifact_hash, None)


purge = FederatedArtifactPurge(
    FederatedRetentionPolicy(FederatedPurgeSide.COORDINATOR),
    round_reference="synthetic-round-1",
)
payload = purge.register(
    "synthetic-client-update",
    PrivateLearningArtifactClass.CLIENT_UPDATE_PAYLOAD,
)
registry[payload] = "synthetic-client-update"
commitment = purge.register(
    "synthetic-commitment",
    PrivateLearningArtifactClass.COMMITMENT,
    dependencies=[payload],
)

purge.record_trigger(
    FederatedPurgeTrigger.COMMITMENT_RECORDED,
    commitment_digest="sha256:" + "c" * 64,
)
receipt = purge.run(delete_owned_artifact)
evidence = emit_federated_round_evidence(purge, lambda bundle: None)
print(receipt.to_json())
```

`register()` hashes each reference at the boundary and returns the digest the
injected deleter receives. Prefer opaque references or keyed `hmac-sha256:`
digests over raw paths, because a plain SHA-256 of a predictable path can be
guessed. Registration closes when the trigger is recorded, and a dependency
must already be registered.

Each target runs through its own confirmed single-target plan built over the
full round manifest, so one failing deleter call does not stop the others. The
receipt reports the overall plan and manifest digests from
`plan_deletion_impact`, set digests for purged and preserved artifacts, and
counts by class. It never lists individual artifacts, paths, site identities,
tensor values or deleter exception text.

Finalization and evidence emission are gated on the purge:

- `emit_federated_round_evidence()` and `finalize_federated_round()` raise
  `FederatedPurgeBlockedError` with `purge_awaiting_trigger`, `purge_pending`
  or `purge_failed` until every mandatory target has been deleted. The emitter
  is not called while blocked. Best-effort failures are counted in the receipt
  but do not block.
- `finalize_federated_round()` promotes only a coordinator round whose purge
  was triggered by the aggregate commitment, and still validates the lifecycle
  transition.
- `purge_artifact()` refuses commitments and evidence with
  `preserved_artifact_deletion_refused` before any deleter call, and `run()`
  never passes a preserved artifact to the deleter.

Pass `state_sink` to receive digest-only resume state after every registration,
trigger and deletion outcome. After a crash or a failed run, restore with
`FederatedArtifactPurge.from_state_json(state, policy=policy)` and call `run()`
again: only pending or failed targets reach the deleter, and a completed purge
returns the same receipt without deleting anything. Resume state that does not
match the policy digest, side, schema or allowed states is refused with
`purge_state_invalid`. All refusals use fixed codes and never echo inputs.

Data-use policy evaluation, legal holds and backups remain outside this module.
