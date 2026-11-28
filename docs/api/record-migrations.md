# Stored record migrations

`openmed.core.record_migrations` is the single place that answers "how do I
bring this stored record from version A to version B?". Each record family
(review packets, decision results, budget ledgers and similar) registers its
migrators as directed edges keyed by `(schema_id, from_version, to_version)`.
The registry resolves a chain, runs it in memory and returns the migrated
record with a value-free report.

```python
from openmed.core.record_migrations import migrate_record

result = migrate_record(stored_record, schema_id="example_note")
migrated = result.record          # a new mapping; stored_record is untouched
report = result.report.to_dict()  # field paths, versions, digests, counts
```

Nothing is written. Persist the migrated record only after your own privacy and
review gates pass.

## Registering a record family

Register the schema first, then one migrator per version step:

```python
from openmed.core.record_migrations import register_migrator, register_schema

register_schema(
    "example_note",
    current_version="1.1.0",
    matcher=lambda record: "note_id" in record,  # optional, for identification
)
register_migrator(
    "example_note",
    "1.0.0",
    "1.1.0",
    upgrade_note,               # takes a deep copy, returns the new record
    lossless=True,
    fixture="example_note/1.0.0-to-1.1.0",
)
```

- `schema_id` is lower-case and normally matches the packaged JSON Schema name.
- Versions are compared as text; integer versions such as `2` become `"2"`.
  `version_field` (default `schema_version`) names the record field that holds
  the version, and `version_when_missing` declares the version of records that
  carry no version field at all.
- `lossless` declares whether the step keeps every input value. Reports mark a
  chain lossy when any step is lossy.
- `fixture` names a directory under `tests/fixtures/migrations/` holding a
  synthetic `before.json` and `after.json` for the edge.

Registration fails with `MigratorRegistrationError` (code
`record_migrator_registration_error`) and one of these reasons:

| Reason | Meaning |
|---|---|
| `duplicate_edge` | A migrator is already registered for the same schema and versions. |
| `cyclic_edge` | The edge would close a cycle, including a step to the same version. |
| `duplicate_schema` | The schema id is already registered. |
| `unknown_schema` | The edge names a schema that is not registered. |
| `invalid_schema` | The schema id, a version, the version field or the matcher is malformed. |
| `invalid_edge` | A version is malformed, the migrator is not callable, or `lossless` is not a boolean. |
| `invalid_fixture` | The fixture name is empty, absolute, or leaves the fixture root. |

## Resolving a chain

`migration_path(schema_id, from_version, to_version=None)` returns the edges to
apply in order, or an empty tuple when the record is already at the target. The
target defaults to the schema's current version. Because cycles cannot be
registered, a chain never revisits a version. When more than one route joins
two versions, for example a direct shortcut next to a two-step chain, the
request is refused as `ambiguous_path` instead of picking one.

## Reports

`RecordMigrationReport.to_dict()` contains:

- `schema_id`, `source_version`, `target_version` and `status` (`current` when
  no step ran, otherwise `migrated`);
- `lossy`, true when any step is declared lossy;
- `added`, `dropped`, `changed` and `moved` JSON Pointer field paths, overall
  and per step, with list indexes collapsed to `*`;
- `counts` and SHA-256 `source_digest` and `target_digest` of the canonical
  input and output records.

A value that leaves one path and reappears unchanged at exactly one other path
is reported as `moved`. Empty containers, empty strings, booleans and `null`
are never matched as moves. Reports never contain field values, and the
migrated record is excluded from the result's `repr()`.

## Refusals

Every failure raises `RecordMigrationRefused` (code
`record_migration_refused`, an `InputError`). Its `refusal` attribute is a
typed `MigrationRefusal` with a reason code, the schema id and the versions
only. Identifiers that are not safe tokens are reported as `null` rather than
echoed, and a migrator's exception message is replaced by its class name.

| Reason | Meaning |
|---|---|
| `unknown_schema` | The schema id is not registered, or no matcher recognises the record. |
| `ambiguous_schema` | More than one registered matcher recognises the record. |
| `no_path` | No registered chain joins the two versions. |
| `ambiguous_path` | More than one chain joins the two versions. |
| `invalid_record` | The record is not a JSON-compatible mapping. |
| `invalid_version` | The record version is missing or is not a safe version token. |
| `migrator_failed` | A migrator raised; only the exception class name is kept. |
| `migrator_output_invalid` | A migrator returned something that is not a JSON-compatible mapping. |
| `version_mismatch` | A migrator returned a record that is not at the edge's target version. |

## Release gate for schema version bumps

`scripts/release/record_migration_gate.py` links packaged JSON Schema version
bumps to record migrators. It reads every packaged `*.schema.json` at the last
release tag (`git describe --tags --abbrev=0 --match 'v[0-9]*' HEAD`) straight
from Git and compares each record version with the working tree: the record
`schema_version` property's `const` or highest `enum` value, else the
top-level `schema_version`, else the version in `$id`. For every schema whose
version changed, it requires one of:

- a registered chain from the released version to the new one, under the same
  schema id as the packaged schema name; or
- an entry in `scripts/release/record_migration_exemptions.json` that names the
  schema and both versions and gives a non-empty `reason`:

```json
{
  "schema_version": 1,
  "no_record_migration": [
    {
      "schema": "decision_request",
      "from_version": "1.1.0",
      "to_version": "1.2.0",
      "reason": "Why stored records need no migration."
    }
  ]
}
```

The gate also requires every registered edge to name an existing
`before.json` and `after.json` fixture pair. It fails (exit code 1) on a bump
with neither a migrator nor an exemption, an exemption without a reason, a
malformed exemption, or a missing fixture. Unreadable refs or exemption files
exit with code 2, including a shallow clone without tags. Exemptions that match
no bump, and exemptions for bumps that also have a migrator, are reported
without failing so they can be cleaned up. The `Model Release Gates` workflow
runs the gate on every manual release run:

```bash
python scripts/release/record_migration_gate.py --json report.json
```

## Default registry

The module-level functions act on `default_registry()`, a process-wide
registry that loads the built-in registrations on first use. Pass
`registry=MigratorRegistry()` to any function to work with an isolated
registry, for example in tests.

## Built-in migrators

`openmed.core.record_migration_catalog` registers the existing migration
functions in the default registry. It adds no migration logic of its own, and
the owning packages are imported only when one of their migrators runs.

| Schema id | Edge | Lossless | Existing function | Identified by shape |
|---|---|---|---|---|
| `clinical_review_packet` | `1.0.0` to `1.1.0` | yes | `openmed.clinical.review_packet_migrations.migrate_review_packet` | yes |
| `clinical_review_packet_mapping` | `1` to `2` | yes | `openmed.clinical.review_packet_mapping_migrations.migrate_review_packet` | no, pass `schema_id` |
| `decision_result` | `1.0.0` to `1.1.0`, `1.1.0` to `1.2.0` | yes | `openmed.structured.decision.migrate_decision_result` | yes |
| `dp_budget_ledger` | `unversioned` to `openmed.dp_budget_ledger.v1` | no | `openmed.risk.budget_migration.BudgetLedgerSnapshot`, checked with `enforce_budget_migration` | yes |

The budget step turns an accountant payload (`compositions`, `policies` and the
per-spend `ledger`) into the canonical aggregate snapshot. It drops the
per-spend ledger and policy details, so it is registered as lossy and its
report lists `/compositions`, `/ledger` and `/policies` as dropped. The budget
verifier still rejects the step if the aggregate budget would weaken.

Integer-versioned review packet mappings carry no distinctive fields, so they
are not identified automatically; pass `schema_id="clinical_review_packet_mapping"`.

Each edge has a synthetic `before.json` and `after.json` under
`tests/fixtures/migrations/<schema id>/<edge>/`, and the tests require every
registered edge to turn its `before` fixture into exactly its `after` fixture.
