# Diagnostic and Recovery Bundle

`openmed service diagnose` writes a small, deterministic JSON bundle that helps
an operator or maintainer understand a failed or degraded OpenMed service. It
records versions, configuration provenance, readiness, store schema versions and
record counts, model registry digests, audit-continuity state, controlled reason
codes, and suggested recovery actions. It never records values.

The bundle is diagnostic metadata only. It is **not a backup**, it cannot restore
service state, and the header always carries `"restorable_state": false`.

## Quick start

```bash
# Collect everything the local process can see and write a 0600 file.
openmed service diagnose --out diagnostics.json \
  --journey-store /var/lib/openmed/journey \
  --audit-chain /var/lib/openmed/audit-chain.json

# Verify a bundle you received, fully offline.
openmed service verify-bundle diagnostics.json
```

Without `--out` the canonical bundle is written to standard output. With
`--json`, the command prints the standard CLI envelope with a summary (entry
count, collection status, status counts, recovery action ids and the manifest
root digest). An existing `--out` file is never replaced unless you pass
`--overwrite`. `verify-bundle` exits `0` for a valid bundle and `1` with a
stable reason code otherwise.

From Python:

```python
from openmed.service.diagnostics import (
    DiagnosticContext,
    DiagnosticLimits,
    DiagnosticRequest,
    DiagnosticSources,
    collect_diagnostic_bundle,
    verify_diagnostic_bundle,
)

bundle = collect_diagnostic_bundle(
    DiagnosticRequest(
        categories=("readiness", "operational_events", "jobs"),
        limits=DiagnosticLimits(max_elapsed_ms=10_000),
    ),
    sources=DiagnosticSources(journey_store_root="/var/lib/openmed/journey"),
    context=DiagnosticContext(app=app),  # a running service app, optional
)
assert verify_diagnostic_bundle(bundle.to_bytes()).valid
```

Passing a running application through `DiagnosticContext(app=...)` adds live
readiness, metrics and model residency evidence. The command-line tool runs out
of process, so those sections report `not_in_process`.

## What it never contains

Raw PHI, secrets, credentials, keys, tokens, reversible mappings, identifiers,
file paths, request or response bodies, job payloads, model inputs, audit
payloads, environment values, and exception messages never enter the bundle,
its errors, or logs. The collector does not log at all. Specifically:

- Configuration is reported as key names with a provenance class (`default`,
  `file`, `environment`, `cli`), never values. Credential keys never report a
  value comparison (`conflict_category` is `not_reported`). Unknown keys in a
  configuration file are counted, not named.
- Service settings are reported by category as the names of recognized
  `OPENMED_SERVICE_*` variables that are set. Unrecognized `OPENMED_*`
  variables are counted, not named. Only built-in profile names are reported;
  a custom profile is reported as `custom_file`.
- Stores are read for schema versions, migration state and record counts.
  Payload columns, identifiers and blob contents are never read into the bundle.
- Model repository ids, registry pointer targets and audit-chain heads are
  replaced by domain-separated SHA-256 digests. These digests are unkeyed; they
  are only ever computed over public model identifiers and integrity anchors,
  never over PHI or credentials.
- Timestamps are rounded down to the hour (`timestamp_precision_seconds` is
  `3600`); `header.generated_at` is the only time field.

Collection never opens a network connection and never writes to the inspected
system. SQLite stores are opened read-only: `immutable` when no WAL sidecars
exist (so none are created), or `mode=ro` when the service already holds them.

## Evidence categories

Bundles contain one entry per requested category, in this canonical order.
Callers can choose a subset with `--category` (repeatable) or
`DiagnosticRequest(categories=...)`; there is no category for arbitrary paths,
environment dumps, database exports or payloads.

| Category | Source | Evidence |
| --- | --- | --- |
| `runtime` | interpreter and package metadata | OpenMed, Python and selected dependency versions; platform class |
| `capabilities` | `openmed.core.capabilities` | which optional backends and extras are available |
| `configuration` | configuration file, profiles directory, environment | profile class, key provenance, service setting presence by category |
| `health` | `openmed.core.doctor.value_free_health_checks` | doctor check names, statuses and codes |
| `readiness` | live app state; service setting parsers | process state, in-flight count, per-group setting validity |
| `operational_events` | live metrics registry | operational event counts, HTTP status classes, circuit-breaker and shedding counters |
| `jobs` | async job metadata store | record counts by status, error and webhook-failure counts |
| `journey_store` | local Journey store (`journey.sqlite3`, `artifacts/`) | migration state, record counts per known table, quarantine counts, integrity check, blob counts |
| `model_registry` | registry slot state | slot keys and digests, checkpoint and lineage counts, pointer digests |
| `model_activation` | live runtime, environment, or a provider | served-model policy, counts and digests, residency |
| `audit_continuity` | audit chain document | chain verification result and counts |
| `backup` | a registered provider | backup state and counts reported by the provider |
| `disk` | volumes holding the configured sources | free-space class (`ok`, `low`, `critical`) |

Source options for the command: `--jobs-store` (default
`OPENMED_SERVICE_JOBS_STORE_PATH` or `~/.cache/openmed/service-jobs.json`),
`--journey-store` (no default), `--registry-state` (default: the source
checkout's `gates/registry_state.json`), `--audit-chain` (no default), and the
global `--config-path` (default `OPENMED_CONFIG` or the standard configuration
file). A source path selects the input of one typed collector; it is never
copied into the bundle.

The `backup` and `model_activation` categories accept a provider through
`DiagnosticContext(providers=...)`. A provider receives the section's budget
handle, must read sources through it, and returns a mapping. Only the fields
listed in `PROVIDER_FIELDS` are kept, and every value passes the value-safety
contract below.

## Limits

Limits are validated before collection starts: a value above the maximum or
below the minimum is refused (`limit_above_maximum`, `limit_below_minimum`,
`limit_invalid`), and each limit is enforced again while collecting.

| Limit | Default | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| `max_elapsed_ms` | 60000 | 10 | 300000 |
| `section_timeout_ms` | 15000 | 10 | 120000 |
| `max_files` | 32 | 1 | 256 |
| `max_file_bytes` | 8 MiB | 1 | 64 MiB |
| `max_total_bytes` | 32 MiB | 1 | 256 MiB |
| `max_depth` | 6 | 1 | 12 |
| `max_dir_entries` | 4096 | 1 | 65536 |
| `max_items` | 256 | 1 | 4096 |
| `max_bundle_bytes` | 256 KiB | 32 KiB | 2 MiB |

- Categories run one at a time. Each runs under the smaller of its
  `section_timeout_ms` and the time left in `max_elapsed_ms`. A collector that
  overruns is abandoned, its partial data is discarded, and the next category
  still runs while time remains. Long SQLite queries are interrupted at the
  deadline.
- `max_files` counts every source file opened (or attempted).
  `max_file_bytes` is checked from the file size before reading and again while
  reading; `max_total_bytes` spans all sources.
- `max_depth` bounds directory recursion below a store root and
  `max_dir_entries` bounds the directory entries visited in total. Symlinks are
  never followed. Parsed JSON sources are additionally limited to 32 nesting
  levels.
- `max_items` bounds each list in a section. When the serialized bundle would
  exceed `max_bundle_bytes`, the largest sections drop their data and report
  `section_too_large`.

## Value-safety contract

After collection, every section is checked against a shape-level contract,
which the verifier applies again. Section data may only contain mappings with
lowercase `snake_case` keys, lists, booleans, `null`, integers within
`±2^53`, and strings that are lowercase controlled tokens, `sha256:` digests,
package versions, registry slot keys, or hour-precision timestamps. Anything
else is omitted and counted under `omissions`.

Omission reasons: `unsafe_value`, `unsafe_key`, `unknown_code`, `unknown_key`,
`unknown_variable`, `invalid_identifier`, `unrecognized_entry`.

The shape check rejects free text, paths, URLs, mixed-case names and
punctuation. It cannot recognize a lowercase token chosen by a provider, so
providers must emit controlled vocabularies.

## Bundle layout and verification

```text
format, schema_version
header     generator, generated_at, timestamp_precision_seconds, categories,
           limits, collection {status, status_counts}, restorable_state
entries    [{sequence, category, status, reason_codes, findings, omissions,
             data, digest}]
recovery   {actions: [{action, kind, triggers, sections, doc}]}
manifest   entry_count, entries [{sequence, category, digest}],
           header_digest, recovery_digest, root_digest
```

Serialization is canonical JSON (sorted keys, no whitespace, ASCII). Each entry
digest, the header digest, the recovery digest and the manifest root are
domain-separated SHA-256 digests. Repeated collection from the same state is
byte-identical apart from `generated_at`, which also changes the header and
root digests.

`openmed service verify-bundle` (or `verify_diagnostic_bundle`) works offline
and reports one stable reason code: `bundle_valid`, `bundle_too_large`,
`bundle_not_json`, `bundle_not_object`, `format_unsupported`,
`schema_version_unsupported`, `missing_field`, `unexpected_field`,
`header_invalid`, `header_modified`, `manifest_invalid`, `manifest_modified`,
`entry_invalid`, `entry_missing`, `entry_added`, `entry_reordered`,
`entry_modified`, `unsafe_content`, `recovery_invalid` or
`recovery_inconsistent`. It detects missing, added, reordered and modified
entries, re-applies the value-safety contract, and checks that the recovery
guidance matches the entries.

The digests are integrity checks, not signatures: someone who rewrites a bundle
can recompute them. Sign or checksum the file through your own channel when you
need to prove who produced it.

## Section statuses and partial-result codes

Each entry has a status: `complete`, `partial` (some evidence missing),
`unavailable` (nothing to collect), `timeout`, `cancelled`, or `failed` (the
collector raised; its message is never recorded). The reason codes explain why:

| Code | Meaning |
| --- | --- |
| `source_not_configured` | No source was named for this category. |
| `source_absent` | The named source does not exist. |
| `source_unsafe` | The source is not a regular file (directory, FIFO, device). |
| `source_unreadable` | The source could not be opened or read. |
| `source_corrupt` | The source could not be parsed. |
| `schema_unsupported` | The source uses a schema this release cannot read. |
| `provider_not_registered` | No provider is registered for the category. |
| `not_in_process` | Live evidence needs a running app passed to the collector. |
| `section_timeout` | The category exceeded its time budget. |
| `elapsed_budget_exhausted` | The total time budget ran out before the category. |
| `collection_cancelled` | Collection was cancelled. |
| `file_limit_reached` | `max_files` was reached. |
| `file_too_large` | A source exceeded `max_file_bytes`. |
| `total_bytes_limit_reached` | `max_total_bytes` was reached. |
| `depth_limit_reached` | `max_depth` or the JSON nesting limit was reached. |
| `entry_limit_reached` | `max_dir_entries` was reached. |
| `item_limit_reached` | A list was truncated to `max_items`. |
| `section_too_large` | The section was dropped to fit `max_bundle_bytes`. |
| `collector_failed` | The collector raised an unexpected error. |

## Findings

Findings are controlled observations about the inspected system:

| Finding | Observation |
| --- | --- |
| `persisted_credential` | The configuration file stores `hf_token`. |
| `config_value_invalid` | A configuration source could not be resolved. |
| `profile_unresolved` | `OPENMED_PROFILE` names no built-in or custom profile. |
| `service_setting_invalid` | A service setting group fails to parse. |
| `served_models_not_explicit` | The `prod` profile serves models without an explicit list. |
| `health_check_failed` | A doctor check failed. |
| `process_not_ready` | The running service has not finished startup. |
| `process_shutting_down` | The running service is draining. |
| `server_errors_observed` | HTTP 5xx responses were counted. |
| `operational_failures_observed` | Operational events ended in a non-success state. |
| `circuit_breaker_open` | At least one circuit breaker is open. |
| `load_shedding_observed` | Batch or admission queues shed requests. |
| `job_store_corrupt` | The job metadata store cannot be parsed. |
| `jobs_failed_present` | Failed jobs are recorded. |
| `jobs_incomplete_present` | Queued or running jobs are recorded. |
| `webhook_delivery_failed` | A terminal webhook delivery failed. |
| `journey_store_corrupt` | The Journey database cannot be read. |
| `integrity_check_failed` | SQLite `quick_check` reported a problem. |
| `migration_uninitialized` | The Journey database has no migration table. |
| `migration_pending` | The Journey database is behind this release. |
| `migration_ahead` | The Journey database is newer than this release. |
| `migration_gap` | Applied migration versions are not contiguous. |
| `migration_checksum_mismatch` | An applied migration differs from this release. |
| `quarantined_records_present` | Quarantined ingestion results await review. |
| `staged_blobs_present` | Staged blob files remain from an interrupted write. |
| `unexpected_store_entries` | The artifact store holds unrecognized entries. |
| `registry_state_invalid` | The registry state document is invalid. |
| `registry_migration_required` | The registry state uses a retired schema. |
| `audit_chain_broken` | The audit chain fails verification. |
| `audit_chain_invalid` | The audit chain document cannot be parsed. |
| `backup_missing` | A backup provider reports no backup. |
| `backup_stale` | A backup provider reports an old backup. |
| `backup_verification_failed` | A backup provider reports a failed verification. |
| `disk_space_low` | A source volume has under 2 GiB or 10% free. |
| `disk_space_critical` | A source volume has under 512 MiB or 2% free. |

## Recovery actions

Recovery guidance lists action categories, the codes that triggered them, and
the sections they came from. Kinds are ordered `inspect`, `retry`, `restore`,
`quarantine`, `escalate`. Actions are suggestions for a human operator;
collection never repairs, restores or deletes anything.

### review_configuration

Kind: inspect. Check configuration files, profiles and service settings for the
reported keys or groups. Remove unknown keys, fix values that fail to parse, and
set an explicit served-model list in production.

### inspect_readiness

Kind: inspect. Look at the service's readiness, logs and metrics for the
reported state: startup not finished, draining, 5xx responses, open circuit
breakers, or load shedding.

### inspect_failed_jobs

Kind: inspect. Use the jobs API to review failed jobs by id from your own
records. The bundle carries counts only.

### inspect_interrupted_writes

Kind: inspect. Staged or unrecognized entries in the artifact store usually come
from an interrupted write. Confirm the service is stopped before removing staged
files, and keep a copy until the store verifies.

### free_disk_space

Kind: inspect. Free space on the volume that holds the reported source.

### inspect_backup_schedule

Kind: inspect. The latest backup is older than expected; check the backup
schedule and its last run.

### rerun_with_larger_limits

Kind: retry. A limit stopped collection. Rerun with a larger value for the
limit named by the reason code, up to its documented maximum.

### retry_collection

Kind: retry. Collection was cancelled; run it again.

### rerun_migrations

Kind: retry. Apply the pending migrations with the release's own tooling, for
example opening the store with the current release or running the registry
migration script.

### resubmit_interrupted_jobs

Kind: retry. Queued or running job records whose payloads were lost in a restart
cannot resume. Resubmit those jobs from the original source.

### retry_webhook_delivery

Kind: retry. Check the webhook receiver and retry delivery from your own
records.

### restore_from_backup

Kind: restore. Restore the affected store from a verified backup with your
backup tooling. The diagnostic bundle cannot be used as a backup.

### restore_registry_state

Kind: restore. Restore the registry state document from version control or a
verified backup.

### quarantine_job_store

Kind: quarantine. Stop the service, move the job metadata store aside for
investigation, and start with an empty store.

### quarantine_journey_store

Kind: quarantine. Stop writers, copy the Journey store aside read-only, and do
not resume ingestion until it verifies.

### review_quarantined_records

Kind: quarantine. Review the quarantined ingestion results before promoting or
discarding them.

### rotate_credentials

Kind: escalate. A credential was persisted in a configuration file. Rotate it
with its owner, remove it from the file, and supply it through the environment
or a secret store.

### escalate_audit_discontinuity

Kind: escalate. The audit chain fails verification. Treat it as a possible
integrity incident: preserve the chain file and involve your security or
compliance owner.

### escalate_backup_gap

Kind: escalate. No verified backup is available; involve the operator who owns
backups before any destructive recovery step.

### align_release_version

Kind: escalate. The store or registry was written by a different release.
Run the release that matches the store, or upgrade with its migration notes.

### report_collector_failure

Kind: escalate. A collector failed unexpectedly. Report the category and code to
the maintainers; the bundle never includes the error message.
