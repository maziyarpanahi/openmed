# Separated projection boundaries

OpenMed can store identified and de-identified Journey projections in
physically separate local namespaces while applying the same purpose, role,
attribute, consent, export, and audit rules to every access. The boundary is a
storage and policy primitive. It is not a claim that a projection is clinically
correct, legally sufficient, or safe for an autonomous care action.

The default implementation is offline after installation. It makes no network
request, sends no telemetry, and does not require a hosted policy service.

## Physical and capability separation

`ProjectionBoundary` creates three private paths beneath a caller-owned root:

```text
projections/
├── identified/
│   ├── metadata.sqlite3
│   └── objects/sha256/...
├── deidentified/
│   ├── metadata.sqlite3
│   └── objects/sha256/...
└── audit/
    └── events.sqlite3
```

On POSIX systems, directories use owner-only permissions. On Windows, provision
the root with an ACL restricted to the service account; Python permission bits
do not establish equivalent Windows access control. Each namespace has its own SQLite
metadata database and content-addressed object tree. An identifier present in
the identified database is unknown to the de-identified database unless a
separate de-identified record was explicitly written.

The APIs also have different capabilities:

- `DeidentifiedProjectionAPI` can read, write, correct, review, and export only
  de-identified records. It has no transform-vault resolver and exposes no
  identified storage path.
- `IdentifiedProjectionAPI` is created only when the caller supplies a
  `TransformVault`. It can record and resolve protected pseudonymization or
  date-shift material after the same policy checks.

Do not use filesystem permissions alone as an authorization system. Run the
process under a dedicated OS identity, encrypt the volume, and isolate backups
according to the deployment's retention and access requirements.

## Create the two APIs

```python
from openmed.compliance import (
    InMemoryTransformVault,
    ProjectionBoundary,
)

boundary = ProjectionBoundary("./journey-data/projections")
deidentified = boundary.deidentified()

# The in-memory vault is suitable for tests and short-lived local workflows.
# Production callers should supply a protected TransformVault implementation.
identified = boundary.identified(InMemoryTransformVault())
```

The boundary rejects symlinked namespace directories. Metadata migrations are
ordered and checksummed. A newer or drifted schema returns the typed
`unsupported` state through `ProjectionBoundary.open()` rather than attempting
an unsafe downgrade.

## Policy requests

Every operation requires a patient-free `ProjectionPolicyRequest`:

```python
from openmed.compliance import (
    ProjectionNamespace,
    ProjectionOperation,
    ProjectionPolicyRequest,
)

request = ProjectionPolicyRequest(
    operation=ProjectionOperation.WRITE,
    namespace=ProjectionNamespace.DEIDENTIFIED,
    purpose="care",
    role="clinician",
    attributes=(),
    data_use_tags=(),
    consent_state="active",
)
```

The request accepts controlled policy codes, not subject identifiers, patient
values, reviewer names, or free-text explanations. The default policy:

- permits identified access only to `clinician`, `data_steward`, and
  `privacy_officer` roles carrying `identified_access`;
- permits de-identified access to those roles and `researcher`;
- denies research and quality purposes in the identified namespace;
- denies unknown roles, purposes, namespace changes, and operation changes;
- applies existing data-use tags, including `no-export` and
  `consent-withdrawn`;
- returns `review` for unknown consent and exports without explicit approval;
- requires `export_approved` for de-identified export and both
  `export_approved` and `identified_export_approved` for identified export.

`allow`, `deny`, and `review` are distinct outcomes. Review is returned as a
typed partial result and never converted into success. An optional consent hook
may return `allow`, `deny`, `review`, or `unknown`. Hook failure and unsupported
hook output deny safely without persisting exception text.

## Write and read a de-identified projection

```python
written = deidentified.write(
    "projection_0123456789abcdef",
    b'{"synthetic":"deidentified"}',
    request,
    consent_scope="care-summary",
    consent_revision="receipt-v1",
    occurred_at="2026-01-02T03:04:05Z",
)
assert written.ok

read_request = ProjectionPolicyRequest(
    operation=ProjectionOperation.READ,
    namespace=ProjectionNamespace.DEIDENTIFIED,
    purpose="care",
    role="clinician",
)
payload = deidentified.read(
    "projection_0123456789abcdef",
    read_request,
    consent_scope="care-summary",
    consent_revision="receipt-v1",
    occurred_at="2026-01-02T04:04:05Z",
)
assert payload.ok
```

Bytes are stored by SHA-256 digest and reverified on every read. Public
`ProjectionRecord` metadata contains only digest, byte count, consent
fingerprints, namespace, version, and timestamp. Projection content is excluded
from `repr`, serialization, audit events, and error messages.

Writing the same first version is idempotent. Reusing an identifier for other
content returns `conflict`. `correct()` appends a new immutable version; it does
not overwrite the earlier blob. Reads return the latest version.

## Consent withdrawal and cached projections

Each API uses the existing fingerprint-only `ConsentCache`. Successful writes
and reads populate it under consent scope and revision fingerprints. Withdrawal
invalidates matching cached values and leaves a revocation tombstone:

```python
event = deidentified.withdraw_consent("care-summary", "receipt-v1")
assert event.invalidated_count >= 0
```

Later reads, writes, exports, corrections, and reviews using that exact consent
pair return `denied` with `consent_withdrawn`, even if a caller mistakenly sends
`consent_state="active"`. Applications that require revocation persistence
across process restarts must supply a consent hook backed by their authoritative
consent system; the built-in cache is intentionally process-local.

## Protected transform vault

`TransformVault` is a narrow protocol for transform material. Its public
`TransformRecord` contains only transform type, input/output digests, timestamp,
and an opaque identifier. The secret mapping or date offset is passed separately
and is available only through `IdentifiedProjectionAPI.resolve_transform()`.

`InMemoryTransformVault` deliberately provides no persistence or encryption and
is not a production key-management solution. A production implementation must
protect material at rest, keep keys outside public records, enforce its own
retention policy, and return typed `StoreResult` outcomes.

## Value-free audit events

Every read, write, export, correction, and review attempt produces a durable
`ProjectionAuditEvent`. Events contain:

- operation and namespace;
- `allow`, `deny`, `review`, or `failure` outcome;
- controlled reason code;
- policy-request and resource digests;
- timestamp and schema version.

They never accept or serialize projection bytes, subject identifiers, reviewer
identity, vault material, credentials, or exception messages. If the audit event
cannot be persisted, the API returns `failure`; it does not report the protected
operation as successful without its audit evidence.

Run `boundary.integrity_check()` after restart or restore. It verifies canonical
metadata hashes, namespace bindings, every referenced content digest, and audit
event hashes. It does not prove consent validity, clinical correctness, or the
security of a caller-supplied vault.

## Schemas and local verification

`ProjectionRecord`, `ProjectionPolicyDecision`, `ProjectionAuditEvent`, and
`TransformRecord` use schema version `1.0.0`. Their JSON Schemas are bundled in
the package and load without network access:

```python
from openmed.compliance import load_all_projection_schemas

schemas = load_all_projection_schemas()
assert schemas["record"]["schema_version"] == 1
```

The focused synthetic suite covers canonical schemas, physical separation,
restart, correction transactions, consent invalidation, vault isolation,
denied/review/allowed exports, value-free audit events, migration refusal, and
cross-namespace bypass attempts:

```bash
pytest \
  tests/unit/compliance/test_projections.py \
  tests/unit/security/test_projection_boundary_bypass.py -q
```
