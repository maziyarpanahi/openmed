# FHIR reference-server compatibility matrix

The `openmed.fhir.reference-server.v1` matrix covers FHIR R4 (`4.0.1`) with
HAPI FHIR, Medplum, and Aidbox. Every run binds one server name to the exact
`CapabilityStatement.software.version` returned by its local container. A
version mismatch fails the profile; no product release is certified by this
document. Re-run the same profile for each deployed version. The committed
matrix and synthetic resources are in
`tests/fixtures/fhir/reference_servers.py`.

| Case | Probe | Result boundary |
| --- | --- | --- |
| Read, search, pagination | Two synthetic Patients, read and `_count=1` search | Pass when a next link is returned; otherwise fail |
| CapabilityStatement preflight | R4 Patient update declaration | Pass, unsupported, or fail before the write probes |
| Conditional write | Repeat `If-None-Exist` create | Pass for 201 then 200; absent declaration is unsupported |
| ETag conflict | Stale `If-Match` update | Pass for 409, 412, or 428; missing version metadata fails |
| Transaction atomicity | Valid and invalid synthetic write in one transaction | Pass only when the transaction fails and the first resource remains absent |
| Partial failure | Batch with one valid and one invalid synthetic write | Pass if the content-free compensation report detects partial failure; absent batch declaration is unsupported |
| Subscription duplicate | Local checkpoint exercise | Server interoperability unsupported pending #2776 |
| Token refresh | Replacement credential custody exercise | Server interoperability unsupported pending #2772 |
| Scope narrowing | Local least-privilege exercise | Server interoperability unsupported pending #3084 |
| Token revocation | Local custody revocation exercise | Server interoperability unsupported pending #2772 |

Results use `pass`, `fail`, `expected_variance`, or `unsupported`. The
`expected_variance` classification is reserved for a documented, version-bound
deviation and is never assigned automatically to an unexpected response. This
first matrix has no accepted version-specific variances. A new variance needs
an observed container result and a reviewed expectation change. `unsupported`
means the server did not advertise the interaction or the cross-service
dependency remains outside this probe. These are distinct from a successful
conformance result.

## Run the suite

The standard command uses only in-memory synthetic transport and a temporary
local SQLite checkpoint. It does not open a network connection:

```sh
.venv/bin/python -m pytest tests/integration/interop/test_fhir_reference_servers.py -q
```

For an isolated container profile, start a disposable FHIR R4 server using the
vendor's container instructions and publish its FHIR base path on loopback.
Use a fresh database with no patient data. Set the exact version reported by
`GET /metadata`; the runner checks it before any writes. Run each of HAPI FHIR,
Medplum, and Aidbox separately. For example, when a HAPI container is already
listening on port 8080:

```sh
OPENMED_FHIR_REFERENCE_TEST=1 \
OPENMED_FHIR_REFERENCE_SERVER=hapi \
OPENMED_FHIR_REFERENCE_VERSION=8.0.0 \
OPENMED_FHIR_REFERENCE_URL=http://127.0.0.1:8080/fhir \
.venv/bin/python -m pytest tests/integration/interop/test_fhir_reference_servers.py -q
```

Replace `hapi` with `medplum` or `aidbox`, and supply that container's actual
version and local FHIR base path. The test refuses non-loopback endpoints,
credentials in URLs, redirects, and proxy settings. It creates random synthetic
Patients, attempts to delete them in `finally`, and should run only against a
disposable database. Discard the container after a failed cleanup. No server
credentials or real clinical resources are bundled. Server authentication and
subscription provisioning remain caller-owned integration work.

The result serializer emits only matrix version, server, declared software
version, fixed case name, outcome, and fixed reason code. It omits URLs,
tokens, search values, resource bodies, generated IDs, server diagnostics, and
response headers. Raw HTTP exceptions are never copied into results. Treat
the optional container's own access logs as sensitive and disable or isolate
them according to the deployment's policy.
