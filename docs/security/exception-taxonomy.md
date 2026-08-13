# No-PHI exception taxonomy

OpenMed privacy gates sometimes need a deliberate, local exception—for
example, a reviewed false positive or a narrowly bounded compatibility
allowance. This taxonomy makes those exceptions reviewable without carrying a
reason string, patient value, source path, or other arbitrary payload into
telemetry or audit artifacts.

The implementation is [`openmed/risk/exception_taxonomy.py`](../../openmed/risk/exception_taxonomy.py).
It is versioned independently of application configuration:

| Field | Value |
|---|---|
| Schema version | `1` |
| Taxonomy version | `1.0` |
| Report type | `no_phi_exception_taxonomy` |

## Closed record schema

Every telemetry or audit exception record must contain exactly these fields:

| Field | Allowed value |
|---|---|
| `schema_version` | Integer `1` |
| `taxonomy_version` | String `1.0` |
| `category` | One of the four categories below |
| `reason_code` | A category-specific code below |
| `scope` | `telemetry` or `audit`; it must match the validator used |
| `evidence` | One to eight typed references with a `sha256:` digest |
| `expires_at` | Explicit UTC ISO-8601 timestamp |
| `approval` | Status, role, digest, and explicit UTC approval timestamp |

Evidence references contain only an allow-listed kind (`test`, `review`,
`policy`, `fixture`, or `incident`) and a lowercase SHA-256 digest. Evidence
kinds may occur only once. Approval metadata has no owner, name, email, ticket
body, notes, or other free-form field; its role is one of
`privacy_reviewer`, `release_reviewer`, `maintainer`, or `test_reviewer`.

The canonical shape is therefore bounded and content-free:

```json
{
  "schema_version": 1,
  "taxonomy_version": "1.0",
  "category": "local_suppression",
  "reason_code": "false_positive_reviewed",
  "scope": "telemetry",
  "evidence": [
    {"kind": "test", "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"},
    {"kind": "review", "digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111"}
  ],
  "expires_at": "2026-09-01T00:00:00Z",
  "approval": {
    "status": "approved",
    "role": "privacy_reviewer",
    "approval_digest": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "approved_at": "2026-08-09T00:00:00Z"
  }
}
```

## Version 1 categories

| Category | Reason codes | Required evidence | Maximum lifetime |
|---|---|---|---:|
| `local_suppression` | `false_positive_reviewed`, `policy_exclusion` | `test` and `review` | 90 days |
| `local_allowance` | `false_positive_reviewed`, `compatibility_boundary` | `test` and `review` | 90 days |
| `synthetic_fixture` | `synthetic_only` | `fixture` and `test` | 30 days |
| `operational_fallback` | `bounded_degradation` | `incident` and `test` | 7 days |

An expiry is measured from `approval.approved_at` and must not exceed the
category bound. Callers that need time-aware validation pass an explicit
`as_of` value; omitting it performs structural validation without reading the
system clock. This keeps repeated validation deterministic and offline.

## Validation API

```python
from datetime import datetime, timezone

from openmed.risk import validate_audit_record, validate_telemetry_record

as_of = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
telemetry_result = validate_telemetry_record(record, as_of=as_of)
audit_result = validate_audit_record(audit_record, as_of=as_of)

if not telemetry_result.valid:
    safe_report = telemetry_result.to_dict()
```

`validate_exception_record` is the shared entry point, and the
`validate_telemetry_record` and `validate_audit_record` wrappers make the
surface explicit. A valid result exposes only the canonical typed record and a
stable record digest. An invalid result exposes fixed finding codes, structural
paths, and fixed messages; it never includes rejected values. Unknown fields,
free-form reasons, owner fields, raw payloads, duplicate evidence, unsupported
versions, missing evidence, invalid digests, and expired records fail closed.

This is an operational review aid, not a compliance certification or clinical
decision guarantee. The validator performs no mandatory network call and does
not provide a release approval decision.

Focused tests:

```bash
.venv/bin/python -m pytest tests/unit/risk/test_exception_taxonomy.py -q
```
