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

## Default registry

The module-level functions act on `default_registry()`, a process-wide
registry that loads the built-in registrations on first use. Pass
`registry=MigratorRegistry()` to any function to work with an isolated
registry, for example in tests.
