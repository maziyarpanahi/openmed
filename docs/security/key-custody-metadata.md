# Key Custody Metadata

`openmed.core.key_custody` validates the descriptive metadata used by signing
and surrogate workflows. It is a local schema and lifecycle gate, not a key
store, cryptographic implementation, compliance certification, or clinical
decision guarantee.

## Accepted boundary

The validator accepts one mapping, an iterable of mappings, or a
`KeyCustodyMetadata` value. Each record contains:

- a synthetic `key_id`;
- one `purpose` (`signing`, `surrogate`, `attestation`, `audit`,
  `authentication`, `encryption`, `integrity`, `backup`, `key_agreement`, or
  `key_wrapping`);
- a supported `algorithm`;
- a timezone-aware `created_at` timestamp; and
- an optional lifecycle `state`, which defaults to `active`.

Lifecycle timestamps are `rotated_at`, `retired_at`, and `destroyed_at`.
They must be strictly ordered after creation. A key can move from `active` to
`rotated` or `retired`, from `rotated` to `retired`, and from `retired` to
`destroyed`. A destroyed record must include retirement and destruction times.
The optional `transitions` list can record the same ordered states explicitly.

Only one active record may hold a purpose. Retired, rotated, and destroyed
records may coexist with the current active record during a documented
rotation history.

## Sensitive-input rule

This validator must run before a workflow selects or loads a key. It never
accepts bytes, byte buffers, private-key fields, secret-like fields, or
unknown metadata fields. It does not read files, consult a clock, make network
calls, or persist input.

Validation failures contain only stable codes and record/field locations. The
result's `to_dict()` and `to_json()` methods include counts, normalized public
metadata categories, and one-way SHA-256 digests of synthetic key identifiers;
they do not include raw input values or key material. Use
`require_valid_key_custody_metadata()` when a workflow should fail closed with
a `KeyCustodyValidationError`.

```python
from openmed.core.key_custody import validate_key_custody_metadata

result = validate_key_custody_metadata(
    {
        "key_id": "signing-primary-2026",
        "purpose": "signing",
        "algorithm": "HMAC-SHA256",
        "created_at": "2026-08-01T10:00:00Z",
        "state": "active",
    }
)

if not result.valid:
    # Safe for an audit record: no input values are included.
    safe_report = result.to_dict()
```

The validator does not prove that a key exists, that a key was destroyed, or
that a cryptographic operation is secure. Those claims belong to the caller's
custody provider and operational controls.
