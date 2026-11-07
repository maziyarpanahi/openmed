# Zero-Downtime Upgrades

A rolling upgrade runs an old and a new OpenMed service build side by side
while they share the async job store, the Journey store, webhook deliveries
and peer traffic. This page defines which builds may coexist (the
compatibility window), how each build proves it before it touches shared
state, and the exact order, hold points, abort conditions and recovery path
for an upgrade or a rollback.

The contract lives in `openmed/service/compatibility.py`. Everything it
publishes is value-free: generation numbers, schema versions, capability
tokens, SHA-256 digests and controlled reason codes. It never contains
hostnames, paths, configuration values, request bodies, identifiers or
exception text. Partial or unknown compatibility is always a blocking state.

## What each build publishes

Every build declares one **contract generation** and a window of at most one
prior generation (N and N-1):

| Field | Meaning |
| --- | --- |
| `contract_generation` | Monotonic generation of this build's wire contracts, durable schemas and capabilities. |
| `min_peer_generation` | Oldest peer generation this build shares state and traffic with: its own generation or the one before. |
| `durable_schemas` | Per durable component, the persisted versions the build operates on (`read`) and the version it writes (`write`). |
| `capabilities` | Worker capability tokens, such as `jobs.deidentify.v1`. |
| `model_generations` | SHA-256 digests of the model generations the node serves. |
| `peer_contract` | Compact `<generation>/<min_peer_generation>` token peers send on requests. |
| `descriptor_digest` | Canonical SHA-256 digest of the descriptor, for rollout evidence. |

The current build declares generation `1`:

| Component | Read | Write | Source |
| --- | --- | --- | --- |
| `jobs` | 1-1 | 1 | Async job metadata store (`OPENMED_SERVICE_JOBS_STORE_PATH`). The unversioned layout written before this contract existed is version 1. |
| `journey_local` | 3-3 | 3 | SQLite Journey store migration chain. |
| `journey_postgres` | 3-3 | 3 | PostgreSQL Journey store migration chain. |

Both Journey stores migrate older schemas forward when they open, so a build
only operates on their latest version. A unit test fails when a store
migration is added without updating this table.

Read the descriptor of a running replica:

```bash
curl -s -H "X-API-Key: $OPENMED_COMPATIBILITY_KEY" \
  http://127.0.0.1:8080/v1/compatibility
```

`GET /v1/compatibility` requires the `compatibility:read` scope when
authentication is enabled and is excluded from the generated OpenAPI schema,
like `/metrics`. Besides the descriptor it reports the persisted-state check
(`state` is `compatible` or `blocked`, with one entry per probed component)
and `admission` (`open`, or `draining` once shutdown has begun).

Read the descriptor or the pairwise matrix of an image without starting it:

```bash
python -m openmed.service.compatibility descriptor
python -m openmed.service.compatibility matrix
```

## Compatibility window rules

1. **The newer build decides.** Two builds are compatible when the newer
   build's `min_peer_generation` is at most the older build's generation.
   An older build never guesses about a newer one; it trusts the window the
   newer build declares.
2. **Each build reads what the other writes.** For every durable component
   both builds use, each build's `write` version must lie inside the other's
   `read` range.
3. **Expand, then contract.** A release may only start *writing* a schema
   version that its predecessor already *reads*. Ship read support in
   generation N (expand), and start writing the new version in N+1
   (contract), which drops N-1 from its window.
4. **Accepted work keeps its contract.** Every job record carries a
   `compatibility` envelope with the admitting build's generation and window,
   the schema versions it was persisted with, the capabilities a claimant
   needs and the pinned model generation. A worker checks the envelope before
   it marks the job running, so a node that cannot handle the work refuses it
   before mutation and leaves it for a compatible worker.
5. **Unknown blocks.** A damaged, unreadable or undeclared component, a peer
   descriptor this build does not understand, and a job without a valid
   envelope are refusals, never assumed successes.

## Readiness and refusals

- The REST service's `/readyz` returns `200` only when preload has finished
  **and** every shared store it uses (today the async job store) is
  compatible or empty. Otherwise it returns `503` with `error.code`
  `incompatible_state` and value-free details:

  ```json
  {"components": {"jobs": {"state": "incompatible", "reason": "schema_ahead_of_window"}}}
  ```

  The check re-probes the shared stores at most every five seconds, so a
  replica notices when a peer migrates shared state underneath it.
- The Journey worker's `/readyz` already reports its store as `incompatible`
  (`schema_unsupported`) when the persisted schema is newer than the build.
  Probe an opened Journey store with `journey_store_observation()` to get the
  same per-component decision as the REST gate.
- A job store written outside the window is never loaded or overwritten. Job
  routes return `503` `incompatible_state`, and the store re-checks the file
  before every write in case another process replaced it.
- Peers announce themselves with `X-OpenMed-Peer-Contract: <generation>/<min>`
  on REST, or the `x-openmed-peer-contract` gRPC metadata key. An announced
  peer outside the window is refused before parsing, loading or processing:
  REST returns `409` `peer_incompatible` (details carry the reason and this
  build's `peer_contract`), gRPC aborts with `FAILED_PRECONDITION` and the
  reason code. A malformed or repeated token returns `400`
  `peer_contract_malformed` or gRPC `INVALID_ARGUMENT`. Probes and
  `/v1/compatibility` are never refused, so peers can always read why.
- Requests without the header are ordinary clients and are not affected.

| Reason code | Where | Meaning |
| --- | --- | --- |
| `peer_compatible` | handshake | The peer may run beside this build. |
| `peer_generation_outside_window` | handshake | The newer build does not declare the older one. |
| `peer_schema_outside_window` | handshake | A shared component is written in a version the other build cannot read. |
| `peer_descriptor_unsupported` | handshake | The peer descriptor is not one this build understands. |
| `peer_contract_malformed` | handshake | The peer contract token is missing, repeated or invalid. |
| `work_compatible` | claim | This worker may claim the work. |
| `work_generation_outside_window` | claim | The work was admitted by a build outside the window. |
| `work_schema_outside_window` | claim | The work was persisted in a schema version this build cannot read. |
| `work_component_unknown` | claim | The work references a durable component this build does not declare. |
| `capability_missing` | claim | The work needs a capability this worker lacks. |
| `model_generation_unavailable` | claim | The work is pinned to a model generation this node does not serve. |
| `work_envelope_invalid` | claim | The work carries no valid envelope. |
| `schema_compatible` / `schema_absent` | readiness | The component is usable or empty. |
| `schema_migration_required` | readiness | Persisted state is older than this build operates on; run the migration first. |
| `schema_ahead_of_window` | readiness, rollback | Persisted state was written by a newer build outside the window. |
| `schema_unknown` | readiness | The component could not be read or is not declared. |
| `rollback_within_window` | rollback | Returning to the target build is safe. |

## Rolling upgrade order

Use a surge rollout that never removes a ready replica before its replacement
is ready: `maxUnavailable: 0` and `maxSurge: 1` in the
[model operator](operator.md) `rolloutStrategy`, or the equivalent Deployment
strategy. Keep `terminationGracePeriodSeconds` above
`OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS` (see [Helm](helm.md)).

**Hold point 0: preflight.**

1. Print the target image's descriptor and check it against a running
   replica's descriptor:

   ```python
   import json
   import urllib.request

   from openmed.service.compatibility import CURRENT_CONTRACT, negotiate_peer

   request = urllib.request.Request(
       "http://127.0.0.1:8080/v1/compatibility",
       headers={"X-API-Key": "<operator key with compatibility:read>"},
   )
   with urllib.request.urlopen(request, timeout=5) as response:
       running = json.load(response)
   peer = {key: running[key] for key in CURRENT_CONTRACT.to_dict()}
   print(negotiate_peer(CURRENT_CONTRACT, peer).to_dict())
   ```

   Run it with the *target* image. Continue only on `peer_compatible`.
2. Confirm every replica reports `state: compatible` and `/readyz` `200`.
3. Take a verified backup of the job store and Journey store. It is the only
   recovery path once a contract step has written a newer schema.

**Steps.**

1. Surge one replica of the new build. **Hold** until its `/readyz` returns
   `200` and its `/v1/compatibility` reports `compatible`.
2. Drain one old replica. Readiness drops first, then admission closes, and
   in-flight requests, streams and jobs finish or checkpoint within the drain
   budget. A checkpointed job keeps its envelope and is resumed by any
   compatible worker.
3. Repeat steps 1 and 2 until no old replica remains. **Hold** at the end
   until every replica reports the new generation.
4. Run Journey migrations only when the previous generation reads the target
   schema. Because the Journey stores migrate forward on open, a migration to
   a version the old generation cannot read is a contract step: start it only
   after step 3, never while an older replica is running.

**Abort conditions.** Stop the rollout and hold the current replica set if
any of these appear:

- a replica returns `503` `incompatible_state`, or `/v1/compatibility`
  reports `state: blocked`;
- a peer handshake returns `peer_incompatible` or `peer_contract_malformed`;
- job claims are refused with `work_*`, `capability_missing` or
  `model_generation_unavailable` and no compatible worker remains to take
  them;
- the number of ready replicas drops below the desired count minus one.

## Rollback

**Inside the window.** Before rolling back, confirm the target build still
reads the persisted state and accepts every build that will run beside it:

```python
from openmed.service.compatibility import (
    CompatibilityDescriptor,
    SchemaObservation,
    rollback_decision,
)

target = CompatibilityDescriptor.from_dict(target_descriptor)
running = [CompatibilityDescriptor.from_dict(item) for item in running_descriptors]
observations = {
    name: SchemaObservation.at(item["schema_version"])
    for name, item in running_snapshot["components"].items()
    if "schema_version" in item
}
print(rollback_decision(target, running=running, observations=observations))
```

Continue only on `rollback_within_window`, then use the same surge-and-drain
order in reverse. Jobs admitted by the newer build that need a capability the
older build lacks stay queued and untouched; they complete after the next
roll-forward. Nothing is lost or re-run because every claim takes a fresh
lease and webhook deliveries keep a generation-independent identity.

**Outside the window.** If the newer build has already written a schema the
target cannot read, the target reports `schema_ahead_of_window`, never becomes
ready and refuses to mutate shared state. The model operator's automatic
rollback therefore stalls safely instead of corrupting newer state. Do not
force it:

1. Keep the newer build serving, or stop admission entirely.
2. Restore the backup taken at hold point 0 into a fresh store.
3. Start the older build against the restored store and verify `/readyz` and
   `/v1/compatibility` before routing traffic to it.
4. Keep the newer state untouched for roll-forward or investigation.

## Integration points

These pieces are delivered separately; the contract is designed for them to
plug in without changing its decisions.

- **Drain coordinator (#3576).** Register the compatibility surfaces as drain
  participants with `register_drain_participant(app, participant)`. Its
  stop-admission phase should be reflected in `/v1/compatibility`
  `admission`, which today follows the service's shutting-down flag, and the
  job participant must checkpoint leases with the record's `compatibility`
  envelope unchanged.
- **Durable-schema registry and migration rehearsal (#3578).** Replace the
  static `DURABLE_SCHEMAS` table with ranges derived from the registry, and
  have the rehearsal evaluate `evaluate_persisted_state()` for every
  generation in `CONTRACT_HISTORY` against the rehearsed version. The drift
  test that pins the table to the store migration chains then moves to the
  registry.
- **Durable jobs with leases (#3574).** Call
  `check_work(local_descriptor(runtime), record["compatibility"])` before
  taking a lease, replaying a checkpoint or delivering a webhook. Keep
  idempotency keys, fencing tokens and webhook delivery identities
  independent of the generation that created them.
- **Backup and restore (#3577).** The restore step above relies on it for any
  rollback outside the window.

## Evidence

`tests/integration/test_mixed_version_upgrade.py` runs offline, in process,
with versioned fakes of the job store and Journey store. It treats the shipped
generation as N-1 and adds a synthetic expand generation and a synthetic
contract generation, then exercises:

- every supported old/new API and worker combination, in both directions,
  for request, stream, job, callback, store and model-generation workflows,
  with no lost or duplicate committed effects;
- a rolling upgrade that keeps a ready replica at every step while drained
  replicas stop admission and checkpoint in-flight jobs;
- lease fencing, idempotency keys and webhook replay across a mixed-version
  handoff;
- capability and model-generation refusals before claim;
- a rollback inside the window followed by a roll-forward, and a rollback
  outside the window that is refused until a backup is restored, with the
  newer state left byte-identical.

The machine-readable matrix comes from `compatibility_matrix()`, and each
scenario records `RolloutEvidence` with generations, descriptor digests,
decisions, reason codes and counts only. A test fails if the published matrix
lists a supported combination the harness does not run.

## Adding a generation

1. Bump `CONTRACT_GENERATION`, set `MIN_PEER_GENERATION` to the previous
   generation, and keep the previous contract as a literal in
   `CONTRACT_HISTORY` (at most two entries).
2. In an expand generation, widen `read` ranges and add capabilities, but
   keep every `write` version readable by the previous generation.
3. In the following contract generation, raise `write`, and set
   `MIN_PEER_GENERATION` to the expand generation.
4. Run the compatibility unit tests and the mixed-version harness, and update
   the harness's synthetic generations if the new contract changes shape.
