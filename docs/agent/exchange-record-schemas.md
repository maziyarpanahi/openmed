# Agent exchange record JSON Schemas

`openmed.agent.exchange_record_schemas` exports strict Draft 2020-12 contracts
for the content-free governance records that agent clients, CLIs, services, and
adapters exchange: single-use [human approval tokens](human-approval-tokens.md)
and their [receipts](human-approval-receipts.md),
[reviewer handoff packets](reviewer-handoffs.md),
[capability grant manifests](capability-grants.md), and the durable
[workflow recovery](workflow-recovery.md) checkpoints and decisions. The export
is deterministic, self-contained, and safe to generate offline.

## Export the schemas

Use `build_exchange_record_schema()` when an adapter needs a JSON-compatible
mapping, or `render_exchange_record_schema()` when it needs canonical compact
JSON bytes:

```python
import json

from openmed.agent.exchange_record_schemas import (
    build_exchange_record_schema,
    build_exchange_record_schema_catalog,
    list_exchange_record_schema_names,
    render_exchange_record_schema,
)

names = list_exchange_record_schema_names()
# ('approval_receipt', 'approval_token', 'capability_grant',
#  'recovery_checkpoint', 'recovery_decision', 'reviewer_handoff')

schema = build_exchange_record_schema("reviewer_handoff")
schema_json = render_exchange_record_schema("reviewer_handoff")
assert json.loads(schema_json) == schema

catalog = build_exchange_record_schema_catalog()
assert set(catalog) == set(names)
```

Unknown names raise `ExchangeRecordSchemaError` with a fixed
`unknown exchange record schema` message that never echoes the rejected value.

Repeated calls return independent mappings and identical JSON text. The
renderer uses sorted keys, ASCII escaping, and fixed separators. It does not
read configuration, import a schema validator, contact a registry, load an
agent provider, or execute a tool.

Every schema uses only a fragment-local `$ref` into its own `$defs`. Consumers
can therefore validate them without network resolution or a hosted registry.
Each contract describes one record; no schema carries a remote `$id`, a hosted
registry entry, or a cross-record reference.

## Closed contract

Each record object sets `additionalProperties: false` and requires exactly the
keys its Python type serializes: metadata-only approval tokens and receipts,
reviewer handoff packets, capability grant manifests, recovery checkpoints,
and recovery decisions. Unknown fields are invalid, including raw clinical
identifiers, credentials, secrets, private paths, request arguments, results,
and examples.

Version fields are closed constants bound to their Python sources, and enums
are closed to the Python enumerations: artifact kinds, requested decisions,
effect kinds, effect states, effect observation states, compensation limits,
recovery phases, recovery dispositions, and recovery reasons. Identifiers,
digests, nonces, idempotency keys, key ids, and timestamps use bounded strings
with anchored patterns, and numeric bounds track the public constants used by
the constructors.

The structural rules that JSON Schema can express are enforced by the record
contracts:

- a pending effect carries no commit evidence, and a committed effect carries a
  commit evidence digest;
- a checkpoint carries all three approval fields or none of them, requires a
  receipt digest once approval is recorded, requires one for an approval-gated
  dispatch or completion, and marks every effect committed once the phase is
  `completed`;
- the first checkpoint has no predecessor digest and every later sequence does;
- a resumable decision names the effects to retry and compensates none, a
  completing decision retries and compensates none, and a review decision
  retries none;
- evidence references are bounded by the public handoff limit, a grant carries
  at least one unique constraint, and a checkpoint carries at least one effect.

## Structure only

The approval token schema describes the serialized token structure and the
HMAC-SHA256 signature format. It never embeds a signature value, a nonce value,
a key, key material, an example token, or a default. The same is true for grant
manifests: the schema accepts a signature-shaped string and documents nothing
about the secret that produces it. Digest fields describe sha256-prefixed or
bare lowercase hex encodings; they are not checksums of any bundled payload.

## Runtime invariants

JSON Schema validation checks the serialized contract. Constructing the Python
records remains the source of truth for invariants that a structural validator
cannot express:

- approval receipts reject a `consumed_at` that is not earlier than
  `expires_at`, and a consumed token is rejected once expired or replayed;
- reviewer handoff packets require `expires_at` after `issued_at` and reject
  evidence references past validation time;
- capability grant constraints are sorted and deduplicated, and signatures are
  verified against a resolved key;
- recovery effects require contiguous ordinals, unique action and idempotency
  keys, and an idempotency key derived from the run, action, tool, kind, and
  operation digest;
- recovery checkpoints require permitted phase transitions, a valid lineage,
  matching approval action and plan digests, and recomputed checkpoint and
  evidence digests.
