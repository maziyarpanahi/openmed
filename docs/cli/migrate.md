# Checking stored records with `openmed migrate`

`openmed migrate check` tells you whether one stored OpenMed record (a review
packet, a decision result, a budget ledger snapshot or any other family in the
[record migration registry](../api/record-migrations.md)) is current, and what
a migration to the current version would change. It is a dry run: the command
reads the file, migrates a copy in memory, prints a value-free report and
writes nothing.

```bash
openmed migrate check stored/packet.json
openmed migrate check stored/packet.json --json
openmed migrate check stored/result.json --schema decision_result --to-version 1.1.0
```

The command identifies the record's schema id from its fields and reads its
version from `schema_version`. Pass `--schema SCHEMA_ID` for records that cannot
be identified by shape, and `--to-version VERSION` to stop at an earlier
version than the current one.

## Output

A record that needs migrating prints the field paths the migration adds,
moves, changes or drops, one line per registered step:

```text
migrate check: migratable (clinical_review_packet 1.0.0 -> 1.1.0, lossless)
  added: /compatibility_policy, /extensions, /transition_ids
  moved: none
  changed: /schema_version
  dropped: none
  step: 1.0.0 -> 1.1.0 (lossless)
dry run: no file was written
```

A current record prints `migrate check: current (<schema id> <version>)`.

With `--json`, the [result envelope](result-envelope.md) `data` object holds
`status` (`current` or `migratable`), `schema_id`, `source_version`,
`target_version`, `lossy`, the `added`, `moved`, `changed` and `dropped` JSON
Pointer paths, per-step reports, counts, SHA-256 digests of the input and
migrated records, and `dry_run: true` with `written: false`. List indexes are
collapsed to `*`. Field values and the file path never appear in the output or
in error messages.

## Exit codes

| Exit code | When |
| ---: | --- |
| 0 | The record is current, or a migration path exists and the report was printed. |
| 1 | A registered migrator failed on the record (`migration_migrator_failed`, `migration_migrator_output_invalid`, `migration_version_mismatch`). |
| 2 | The file could not be read or parsed (`record_unreadable`, `record_too_large`, `record_invalid`), or no unique migration path exists (`migration_unknown_schema`, `migration_ambiguous_schema`, `migration_no_path`, `migration_ambiguous_path`, `migration_invalid_record`, `migration_invalid_version`). |

Errors name only the schema id and versions, for example
`Record migration refused (no_path) for clinical_review_packet from 1.0.0 to
9.0.0.` A failing migrator is reported by exception class name only.

## Limits

- The file must be a regular file holding one JSON object of at most 16 MiB.
  FIFOs, devices and directories are refused without being opened for reading.
- Writing migrated records is out of scope for this release: migrate in your
  own protected process with `openmed.core.record_migrations.migrate_record`,
  run your privacy and review gates, and persist the result only then.

## Dry-running a directory with `openmed migrate records`

Before upgrading a deployment, count how many stored records each migration
touches without reading any of them into a report:

```bash
openmed migrate records exports/ --dry-run
openmed migrate records exports/ --dry-run --json
openmed migrate records exports/review-packets/ --dry-run --schema clinical_review_packet
```

`--dry-run` is required: in this release the command can only migrate in
memory. It walks every `.json` and `.jsonl` file under the directory. A `.json`
file holds one record or a JSON array of records; a `.jsonl` file holds one
record per line, and blank lines are skipped. Each record is identified (or
taken as `--schema`), its chain runs in memory, and the result is counted as
one of these outcomes:

| Outcome | Meaning |
| --- | --- |
| `current` | The record is already at its schema's current version. |
| `migratable` | A lossless chain reaches the current version. |
| `lossy` | A chain exists but at least one step is declared lossy. |
| `no_path` | The schema is unknown, or no unique chain reaches the current version. |
| `invalid` | The file or record could not be read, parsed or migrated. |

The report holds only counts: files scanned, invalid and ignored; records per
outcome; record counts grouped by schema id, version and outcome; counts of
controlled reason codes such as `symlink_outside_root`, `file_too_large`,
`line_too_long`, `record_invalid`, `unknown_schema` or `migrator_failed`; and
the limits that applied. File names, paths, digests and record values never
appear.

```text
migrate records: dry run over 2 records in 2 files (no file was written)
  current: 1, migratable: 1, lossy: 0, no_path: 0, invalid: 0
  clinical_review_packet 1.0.0: migratable 1
  clinical_review_packet 1.1.0: current 1
```

The walk is bounded and never leaves the directory:

- Symlinks that resolve outside the directory, broken symlinks, FIFOs and
  devices are counted as `invalid` and never opened. Symlinks to files inside
  the directory are read once; directory symlinks are not followed.
- `--max-file-bytes` (default 64 MiB) and `--max-line-bytes` (default 1 MiB)
  count larger files and longer JSON Lines rows as `invalid` without parsing
  them.
- `--max-files` (default 10,000), `--max-records` (default 100,000) and a
  nesting depth of 32 directories stop the run with `records_limit_exceeded`
  (exit code 2), so a partial count is never reported as complete.

The command exits `0` once the walk completes, whatever the outcome counts, and
`2` when the directory cannot be read or a limit stops the run.
