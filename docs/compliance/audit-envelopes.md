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

Focused tests:

```bash
.venv/bin/python -m pytest tests/unit/compliance/test_audit_envelope.py -q
```
