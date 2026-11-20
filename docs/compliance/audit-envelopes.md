# Audit-envelope metadata

`openmed.compliance.audit_envelope` parses a bounded, signed JSON envelope for
local audit workflows. It keeps the header, signature metadata, payload type
and size, and a canonical SHA-256 payload fingerprint. The parsed object never
stores the payload, and its reports and exceptions contain no payload text.
This is an integrity aid, not a compliance certification or clinical decision
guarantee.

## Envelope shape

The parser accepts this closed shape. Header fields may also be flattened at
the top level for producers that do not nest them:

```json
{
  "schema_version": 1,
  "header": {
    "envelope_id": "synthetic-envelope",
    "producer": "synthetic-fixture",
    "created_at": "2026-08-11T00:00:00Z",
    "content_type": "application/json"
  },
  "signature": {
    "algorithm": "HMAC-SHA256",
    "key_id": "synthetic-key",
    "value": "synthetic-signature"
  },
  "payload_fingerprint": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
  "payload_size": 42,
  "payload_type": "object"
}
```

An envelope may include a transient `payload` while it is being parsed. When
present, the parser computes its canonical JSON fingerprint and checks the
declared fingerprint and size. The payload is discarded before the
`AuditEnvelope` is returned. A metadata-only envelope may omit `payload` but
must retain its fingerprint. The supported schema version is `1`; duplicate,
unknown, unsigned, malformed, or conflicting fields fail closed.

Bounds are deliberately finite: the encoded envelope is limited to 64 KiB,
the canonical payload inspected transiently to 8 MiB, header metadata to 12
fields with 128-character values, and signature values to 4096 characters.
Callers may select smaller limits but cannot raise these defaults.

## Parse and report

```python
from openmed.compliance import parse_audit_envelope

envelope = parse_audit_envelope(envelope_json)
report = envelope.to_dict()
```

The report includes `payload_fingerprint`, `payload_size`, `payload_type`, and
signature algorithm/key/fingerprint metadata. It does not include `payload` or
the signature value. `AuditEnvelopeError.to_dict()` returns only a fixed error
code, a structural field name, and `redacted: true`; rejected values are never
included in error messages.

The parser performs no key lookup, filesystem access, clock read, or network
call. It validates that a signature value is present and structurally bounded;
cryptographic verification must happen separately with a caller-supplied key
and trust policy.

Use synthetic payloads only in tests and committed fixtures. Do not place raw
clinical text, identifiers, credentials, or other sensitive values in headers,
logs, reports, or fixture data.

## AuditReport JSON Schema

The envelope above carries a payload fingerprint. The signed de-identification
report itself, `openmed.core.audit.AuditReport`, has its own JSON Schema,
`openmed/core/schemas/json/audit_report.schema.json`
(`$id` `https://openmed.dev/schemas/v1/audit_report.schema.json`, title
`OpenMedAuditReport`). It describes exactly what `AuditReport.to_dict()` and
`to_json()` emit, signed or unsigned: the policy and resolved profile, detector
provenance, the safety sweep, spans, thresholds, residual risk, the OpenMed
version, the manifest, input and output hashes, the `repro_hash` and the
`signature` (`null` until the report is signed).

```python
from jsonschema import Draft202012Validator

from openmed.core.audit import load_audit_report_schema

Draft202012Validator(load_audit_report_schema()).validate(report.to_dict())
```

The schema keeps the report value-free:

- The report, every span, every detector, the signature and each span's
  `context.before` and `context.after` descriptors are closed objects
  (`additionalProperties: false`), so a raw-text key such as `text` fails
  validation.
- Span and context offsets are non-negative integers. Every hash field
  (`text_hash`, `input_hash`, `deidentified_text_hash`, `manifest_hash`,
  `repro_hash`) must match `sha256:<64 hex>`.
- `signature.algorithm` must be `HMAC-SHA256` and `signature.value` a 64-digit
  lowercase hex HMAC; any other algorithm fails validation, as it fails
  `AuditReport.verify()`.

A JSON Schema cannot check that `end >= start`, that offsets fall within
`document_length`, or that the `repro_hash` and HMAC are correct; use
`AuditReport.verify()` with the release key for that.

Reports do not embed a `schema_version`. The schema declares the property for
registry consistency but does not require it; the contract version is the
`audit_report` entry in `openmed/core/schemas/json/schema-fingerprints.json`,
which `tests/unit/core/test_schemas_versioning.py` pins so a removed field or a
breaking change without a version bump fails.

Do not confuse it with `audit.schema.json` (`OpenMedAuditEvent`), which
describes one per-span audit event with an event id, document id, timestamp,
actor and provenance reference. Tools that validate "the audit report" must
use `audit_report.schema.json`.

Focused tests:

```bash
.venv/bin/python -m pytest tests/unit/compliance/test_audit_envelope.py -q
.venv/bin/python -m pytest tests/unit/core/test_audit_report.py -q
```
