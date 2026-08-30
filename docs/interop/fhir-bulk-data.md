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

## Fail-closed behavior

The default allow-list covers the supported FHIR R4 resource subset. `Binary`
resources, unknown resource types, malformed nested resource paths, malformed
or unsafe XHTML narratives, scripts, external narrative links, and event
attributes are rejected. Rejected resources are not copied to output. The
gateway does not use a best-effort pass-through fallback for unsafe input.

## REST jobs

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
