# OpenMRS Adapter

OpenMed can pull notes and observations from a facility-local OpenMRS server,
de-identify free text on the facility box, and then write or export only the
de-identified copies. The adapter supports both the legacy REST API and the
OpenMRS FHIR2 R4 module.

```text
KenyaEMR / UgandaEMR
        |
        | facility LAN (REST or FHIR2)
        v
OpenMed on the facility box
  pull -> de-identify -> verify manifest
        |
        | de-identified payloads only
        v
destination OpenMRS / FHIR transaction Bundle / NDJSON
```

Raw notes do not need to leave the facility network. OpenMed does not cache
source payloads, write raw PHI to logs, or make a network request during module
import. Keep the adapter and its output directory on a facility-controlled
machine and restrict the OpenMRS account to the resources it needs.

## Install

The HTTP client is optional:

```bash
uv pip install "openmed[openmrs]"
```

Importing `openmed.interop` and listing `available_adapters()` remains lazy;
neither the OpenMRS module nor the HTTP client is imported until requested.

## Connect to KenyaEMR or UgandaEMR

Include the OpenMRS application context in the base URL. Many deployments use
`/openmrs`, but confirm the local installation path with the facility
administrator.

```python
from openmed.interop.openmrs import OpenMRSAdapter, OpenMRSClient, OpenMRSConfig

config = OpenMRSConfig(
    base_url="https://kenyaemr.facility.local/openmrs",
    username="openmed-adapter",
    password="read-from-a-secret-store",
    page_size=100,
    max_retries=3,
)

with OpenMRSClient(config) as client:
    adapter = OpenMRSAdapter(client)
    observations = adapter.pull_rest("obs", params={"patient": "patient-uuid"})
    encounters = adapter.pull_rest(
        "encounter",
        params={"patient": "patient-uuid"},
    )
```

Legacy REST collection requests use `limit` and `startIndex` pagination. The
adapter follows server-provided next links when available, retries transient
failures, and rejects pagination links that leave the configured OpenMRS
origin or application path.

The low-level client can pull the REST `patient` endpoint for facility-local
lookup, but the adapter does not write legacy Patient objects back. Their name
and identifier structure is not safely represented by the narrow REST
obs/comment walker. Use the FHIR2 Patient path when a de-identified Patient
resource must be handed off.

OpenMRS also accepts an existing servlet session token. Pass only one
authentication mode:

```python
config = OpenMRSConfig(
    base_url="https://ugandaemr.facility.local/openmrs",
    session_token="token-returned-by-the-local-session-endpoint",
)
```

The token is sent as the `JSESSIONID` cookie. Do not put credentials or tokens
in a URL, source file, log line, or committed configuration.

## What the REST walker changes

The legacy REST model is not FHIR. Its walker transforms only string-valued
`value`, comment, and encounter-note fields. It never changes UUIDs, coded
concepts, references, dates, or numeric observation values. Every result
includes an OperationOutcome-style manifest of the changed paths:

```python
for record in observations:
    print(record.source_id, record.transformed_paths)
```

Use transformed paths for audit counts, not the original or transformed text.
The manifest contains paths and informational diagnostics only.

## FHIR2 pull and export

The FHIR2 path uses OpenMed's existing FHIR de-identification operation. Pull
Patient, Encounter, and Observation resources before assembling a Bundle so
all internal references have a target:

```python
with OpenMRSClient(config) as client:
    adapter = OpenMRSAdapter(client)
    records = (
        adapter.pull_fhir("Patient", params={"_id": "patient-uuid"})
        + adapter.pull_fhir("Encounter", params={"patient": "patient-uuid"})
        + adapter.pull_fhir("Observation", params={"patient": "patient-uuid"})
    )

    bundle = adapter.export_bundle(records, doc_id="facility-job-2026-07-17")
    summary = adapter.export_ndjson(records, "out/openmrs-deidentified.ndjson")
```

`export_bundle` delegates to `openmed.clinical.exporters.fhir.to_bundle`. It
produces a FHIR R4 transaction Bundle with deterministic `urn:uuid` full URLs,
request blocks, and rewritten in-Bundle references. `export_ndjson` uses the
existing streaming FHIR bulk path and returns its PHI-safe summary and output
digest.

Never use a patient name, phone number, national identifier, or other direct
identifier as `doc_id`; the identifier seeds deterministic full URLs and may
be retained in operational metadata.

## Write-back and dry runs

Set a separate destination when de-identified copies should go to another
facility-controlled OpenMRS server:

```python
config = OpenMRSConfig(
    base_url="https://kenyaemr.facility.local/openmrs",
    destination_url="https://research-openmrs.facility.local/openmrs",
    username="openmed-adapter",
    password="read-from-a-secret-store",
)

with OpenMRSClient(config) as client:
    adapter = OpenMRSAdapter(client)
    records = adapter.pull_fhir("Observation", params={"patient": "patient-uuid"})

    planned = adapter.write_back(records, dry_run=True)
    assert all(result.dry_run for result in planned)

    # Perform this only after the local manifest/no-leak review.
    completed = adapter.write_back(records)
```

FHIR resources with an `id` use `PUT`; new FHIR resources use `POST`. Legacy
REST resources follow OpenMRS conventions and use `POST` for both create and
partial update (`POST /resource/{uuid}`).

For a legacy obs workflow that stores the de-identified value in the original
obs comment without replacing the full resource, use the explicit convention:

```python
adapter.write_back(observations, obs_comment=True, dry_run=True)
```

This mode requires an obs UUID and sends only `{"comment": "..."}`. It is
available for REST obs records only.

## Facility deployment checklist

1. Use a least-privilege OpenMRS service account and facility-managed secret
   storage.
2. Keep TLS verification enabled; install the facility CA rather than setting
   `verify_tls=False` in production.
3. Start with `dry_run=True`, review only resource IDs and transformed paths,
   and avoid printing payload bodies.
4. Run the no-leak guard against seeded identifiers before egress.
5. Store Bundle or NDJSON output on encrypted facility-controlled storage and
   apply the site's retention policy.

The unit suite uses synthetic recorded-shape fixtures under
`tests/unit/interop/fixtures/openmrs/`; it never contacts a live OpenMRS
installation.

## OpenMRS references

- [OpenMRS REST API client documentation](https://openmrs.atlassian.net/wiki/spaces/docs/pages/25469830/)
- [OpenMRS REST resource reference](https://rest.openmrs.org/)
- [OpenMRS API overview](https://openmrs.atlassian.net/wiki/spaces/docs/pages/25520547/)
