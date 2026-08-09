# Audit-report signing key rotation

OpenMed audit reports use deterministic HMAC-SHA256 signatures and record a
non-secret `key_id` with each signature. The rotation adapter lets an operator
sign new reports with a current key while continuing to verify reports signed
with a retained previous key.

Key custody remains outside OpenMed. The caller supplies a callback or mapping
that resolves a key ID to key material. OpenMed does not create, persist, log,
serialize, or rotate that material, and this feature makes no network calls.

## Sign with the active key

```python
from openmed.core.audit_key_rotation import AuditKeyRotationSigner


def signing_key_provider(key_id: str) -> bytes:
    # Resolve from the caller's local vault or HSM. Do not return key material
    # from a report, log message, exception, or audit envelope.
    return caller_owned_key_store.get_signing_key(key_id)


signer = AuditKeyRotationSigner(
    key_id="audit-2026",
    key_provider=signing_key_provider,
)
signed_report = signer.sign(report)
```

The provider is called only with the non-secret key ID. To use a different
active key for one operation, pass `key_id` to `sign`, or construct a signer
with the new ID:

```python
signed_report = signer.sign(report, key_id="audit-2027")
```

`AuditKeyRotationSigner` returns the same `AuditReport` after calling its
existing deterministic `sign` operation. The serialized signature contains the
algorithm, key ID, and authentication value; it never contains the resolved
key.

## Verify current and retained keys

The verifier reads the report's `signature.key_id` and asks the caller-owned
provider for that key. The provider can therefore retain both the current and
previous keys during the required audit-retention window:

```python
from openmed.core.audit_key_rotation import AuditKeyRotationVerifier


def verification_key_provider(key_id: str) -> bytes:
    # The caller decides which active and retired keys remain trusted.
    return caller_owned_key_store.get_verification_key(key_id)


verifier = AuditKeyRotationVerifier(key_provider=verification_key_provider)
if not verifier.verify(stored_report):
    raise ValueError("audit report signature or evidence is invalid")
```

Verification fails closed for unsigned reports, unknown or invalid key IDs,
provider failures, changed report content, and unavailable retired keys. It
returns `False` without copying provider exception text into the report or an
OpenMed log. Optional `original_text` and `deidentified_text` bindings are
hashed in memory by `AuditReport.verify`; they are not serialized by the
rotation layer.

## Rotation boundary

1. Add the new key to the caller-owned provider under a safe metadata ID.
2. Sign new reports with that ID.
3. Keep the former key available for verification until the retention policy
   permits retirement.
4. Remove the former key from the provider only after that policy is met.

Key IDs must be stable, non-secret metadata identifiers such as
`audit-2026`. Never use a raw key, token, patient identifier, or document text
as a key ID. HMAC is symmetric: verification proves integrity to callers that
can access the provider, not public non-repudiation.

This feature provides an offline signing and verification boundary. It does not
choose key-retention periods, implement a vault/HSM, establish access control,
or provide a compliance certification or clinical decision guarantee.
