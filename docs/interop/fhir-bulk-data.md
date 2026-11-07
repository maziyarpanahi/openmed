# FHIR Bulk Data privacy gateway

OpenMed provides a local-first FHIR Bulk Data 3.0.0 gateway for supported R4
resources. It reads NDJSON one resource at a time, applies the existing FHIR
de-identification policy, and commits each output file atomically. Synthetic
offline exports are useful for validating the workflow without a FHIR server.

## Local export

```python
from openmed.interop.fhir.bulk import BulkDataGateway, BulkGatewayConfig

gateway = BulkDataGateway(
    BulkGatewayConfig(
        input_dir="./synthetic-export",
        output_dir="./deidentified-export",
        policy="hipaa_safe_harbor",
        method="replace",
        schema_policy="fhir_hipaa_safe_harbor",
        date_shift_secret=date_shift_secret,
        max_buffered_resources=1,
    )
)
report = gateway.export(job_id="synthetic-run")
print(report.to_dict())
```

The report contains counts, resource types, output hashes, timings, policy
provenance, and rejection records. Rejection records contain only a reason,
line number, structural path, and SHA-256 hash of the rejected resource. They
never contain a resource body, identifier, local absolute path, or URL.

Completed files are recorded in an atomic JSON checkpoint. If a process stops
after a file commit, a later run verifies both the input and output hashes,
skips that file, and resumes the remaining files. A partial `.part` file is
never promoted to the final output. Output serialization is deterministic, so
an interrupted and resumed synthetic export has byte-identical completed
files and no duplicate resources.

When `schema_policy` is set, the checkpoint compatibility hash also covers the
complete field policy, date-shift settings, and opaque fingerprints of the
subject/date keys. A changed policy or key is therefore reprocessed instead of
resuming incompatible output; raw key material is never checkpointed.

## Field-level schema policies

Schema policies map canonical FHIR or OMOP paths to `suppress`, `generalize`,
`date-shift`, `route-to-deidentify`, or `keep`. Bundled Safe Harbor-oriented and
research limited-dataset variants are available for both schemas. They are
technical controls, not a legal determination that a data set satisfies HIPAA
or another regulatory framework.

Before processing, use `lint_schema_policy(data, policy)` to find observed
fields without explicit rules and `validate_schema_policy(policy,
schema_fields)` to compare a policy with a known schema. Uncovered
identifier-shaped or identifier-declared fields are suppressed; other fields
follow the policy's `default_action` and produce reviewable warnings.

```python
from openmed.structured import apply_schema_policy, lint_schema_policy

findings = lint_schema_policy(bundle, "fhir_hipaa_safe_harbor")
if any(finding.severity == "error" for finding in findings):
    raise ValueError("FHIR schema has uncovered identifier fields")

deidentified = apply_schema_policy(
    bundle,
    "fhir_hipaa_safe_harbor",
    date_shift_secret=date_shift_secret,
)
```

FHIR Bundle wrappers and their resources are inspected separately. Unsafe
Bundle identifiers, signatures, request/response metadata, links, and external
`fullUrl` values are removed. References to removed external URLs are rewritten
to safe relative references when their target entry has a valid resource type
and id. Opaque `urn:uuid` and `urn:oid` linkage remains intact. Resource types
without rules in the selected schema policy are rejected instead of copied.

For OMOP, pass a mapping of linked table names to rows or use `apply_omop_file`
for a CSV/Parquet file. Linked rows use `person_id` or an explicit `subject_key`
for consistent date offsets. Blank optional dates stay blank, direct person
source identifiers are suppressed, and clinical `*_source_value` fields are
routed through the configured text de-identifier.

```python
from openmed.structured import apply_omop_file, apply_schema_policy

tables = apply_schema_policy(
    {"person": person_rows, "visit_occurrence": visit_rows},
    "omop_hipaa_safe_harbor",
    date_shift_secret=date_shift_secret,
)

apply_omop_file(
    "visit_occurrence.csv",
    "deidentified/visit_occurrence.parquet",
    "omop_hipaa_safe_harbor",
    date_shift_secret=date_shift_secret,
)
```

## Fail-closed behavior

The default allow-list covers the supported FHIR R4 resource subset. `Binary`
resources, unknown resource types, malformed nested resource paths, malformed
or unsafe XHTML narratives, scripts, external narrative links, and event
attributes are rejected. Rejected resources are not copied to output. The
gateway does not use a best-effort pass-through fallback for unsafe input.

## REST jobs

REST filesystem access is disabled unless the operator configures
`OPENMED_SERVICE_BULK_OUTPUT_ROOT` and, for local input,
`OPENMED_SERVICE_BULK_INPUT_ROOTS` (comma-separated roots). Keep these roots
and their parents owned by the service operator, not writable by untrusted
local users; the policy is a network-request boundary, not an OS sandbox.
Provision any parent directories before launching the service.

Requests use normalized relative paths such as `input_dir: "synthetic-export"`
and `output_dir: "run-001"`, resolved only inside those roots. Absolute paths,
home expansion, traversal, symlink escapes, links within input trees, and
caller-selected `checkpoint_path` are rejected. Output directories must not
already exist: each accepted job exclusively reserves a private directory
and derives its checkpoint name from the generated job id. Reusing an output
directory returns `path_exists`, never overwrites another job's files.
Filesystem errors are reconstructed outside handlers without retaining private
filename contexts. The generated checkpoint identity is bounded and cannot
contain path components. Other controlled codes are `bulk_paths_disabled`, `path_invalid`, and
`path_outside_root`; none echoes the submitted path. The same output policy
applies to `/fhir/smart-backend/ingestions`.

Migration: move operator-approved input/output locations under configured
roots, replace absolute request paths with relative names, and stop sending
checkpoint paths. The direct Python `BulkDataGateway` API and its checkpoint
resume behavior are unchanged.

The service exposes asynchronous local/SMART-compatible job routes:

- `POST /fhir/bulk/exports` or `POST /fhir/bulk/imports` starts a job and
  returns `202 Accepted` with a `Content-Location` polling URL.
- `GET /fhir/bulk/exports/{job_id}` polls status; the import path is an alias.
- `GET /fhir/bulk/exports/{job_id}/manifest` returns the PHI-free output
  manifest.
- `GET /fhir/bulk/exports/{job_id}/report` returns the PHI-free job report.
- `DELETE /fhir/bulk/exports/{job_id}` requests cancellation.

For an offline job, provide `input_dir` and `output_dir`. To use SMART
backend-services authentication, omit `input_dir` and provide the configured
FHIR base URL, token URL, client ID, and an operator-supplied private key. The
key, client assertion, access token, raw resources, and source URLs are held
only in the active request/job path; they are not returned, checkpointed,
logged, or placed in manifests, exceptions, or reports.

SMART status polling honors `Retry-After`. Manifest file downloads are bounded
by `max_inflight_downloads` and `max_buffered_resources`, and each file is
written to a temporary sibling before an atomic rename. Keep credentials in
the caller's secret-management system; do not put them in source-controlled
fixtures or committed configuration.

### Idempotent starts

A client that loses the response to a start request should retry it with an
`Idempotency-Key` header instead of starting a second export. The header is
accepted by `POST /fhir/bulk/exports`, `POST /fhir/bulk/imports` and
`POST /fhir/smart-backend/ingestions`:

```bash
curl -sS -X POST http://127.0.0.1:8080/fhir/smart-backend/ingestions \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: 8e03978e-40d5-43e8-bc93-6894a57f9324" \
  --data @smart-ingestion.json
```

- The key is 1-255 letters, digits or `._:~+/=-` characters, sent once, bare
  or as a quoted string. Anything else returns `400 idempotency_key_invalid`
  before any validation, storage reservation or job start.
- A repeat with the same key and the same request returns the job the first
  request created: the same status code, `Content-Location` and body as the
  first response, plus `Idempotency-Replayed: true`. No second task is
  scheduled and no second token, kickoff or download request reaches the
  source FHIR server, even after the first job finished, failed or was
  cancelled. Use a new key to start a new job.
- The same key with a different request returns
  `409 idempotency_key_conflict` with no side effect. While the first request
  is still being processed, a duplicate returns `409
  idempotency_key_in_progress` with `Retry-After: 1`.
- Keys are scoped to the start route and to the authenticated principal (see
  [Job Ownership](../serving/authentication.md#job-ownership)). Another
  principal using the same key starts its own job and never learns that the
  key was used. With authentication disabled, all callers share one scope.
- "The same request" means every validated request field except
  `private_key_pem`, after defaults are applied. Credentials are excluded, so
  a retry with a rotated signing key still matches; paths and URLs are
  included, so a different `output_dir` or FHIR server is a conflict.
- Only accepted starts are remembered. A start rejected by request
  validation or the storage checks (`400` or `422`) leaves the key unused, so
  the corrected request can reuse it.

The service stores neither the key nor any request field. Each manager keeps
an in-memory entry of two keyed HMAC-SHA256 digests (route, owner and key; the
non-credential request fields), the job id and the PHI-free initial status
payload, under the deployment `OPENMED_SERVICE_DIGEST_KEY`. Entries expire
after `OPENMED_SERVICE_IDEMPOTENCY_TTL_SECONDS` (default `86400`, at most
seven days); each manager remembers at most
`OPENMED_SERVICE_IDEMPOTENCY_MAX_KEYS` keys (default `1024`, at most
`100000`) and evicts the oldest completed entry when full; if every entry
is still being processed, a new key gets `503 idempotency_capacity_exhausted`
with `Retry-After: 1`. Entries do not survive a restart. After expiry, eviction or a restart, a retried request is
processed as new; its `output_dir` is still reserved by the first job, so it
fails with `path_exists` instead of writing duplicate output.
