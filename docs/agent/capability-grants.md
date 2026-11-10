# Signed Capability-Grant Manifests

`openmed.agent.permissions.grants` defines an offline authority boundary for
local clinical agents. A manifest signs one or more exact combinations of:

- tool;
- resource class;
- action;
- policy profile; and
- exclusive expiry time.

Verification fails closed for a missing manifest, invalid signature, missing
key, expired manifest, or request outside the exact signed constraints. There
are no wildcard, prefix, inheritance, or implicit-default rules.

## Governance metadata only

The four constraint values use canonical, developer-authored identifiers:

```text
tool:<reverse-domain>/<local-name>[@<version>]
resource:<reverse-domain>/<local-name>[@<version>]
action:<reverse-domain>/<local-name>[@<version>]
policy:<reverse-domain>/<local-name>[@<version>]
```

They describe stable capability classes, not runtime data. Never place a
patient, clinician, tenant, device, encounter, record, credential, tool
argument, or clinical value in these fields. Do not encode or hash such a value
into an identifier to work around this boundary. Keep record-level selection
and minimum-data projection in the separate access-ticket layer.

Grant objects use value-free `repr()` output, and grant failures expose only a
stable `code` and fixed `field_name`. Applications should log those controlled
diagnostics, not serialized manifests or rejected request values.

## Issue and verify a manifest

The v1 signature algorithm is HMAC-SHA-256 over canonical JSON. The
application supplies at least 32 bytes of local key material; OpenMed does not
load keys from a service, transmit a manifest, or perform any network request.
HMAC uses shared-key trust, so issuers and verifiers with the same key can both
create manifests. Keep issuance in a more trusted local component when that
distinction matters.

```python
from openmed.agent.permissions.grants import (
    CapabilityGrantConstraint,
    CapabilityGrantRequest,
    CapabilityGrantSigner,
    CapabilityGrantVerifier,
    dispatch_with_capability_grant,
)

key = b"replace-with-32-or-more-local-key-bytes"
constraint = CapabilityGrantConstraint(
    tool="tool:org.example/redact@1.0.0",
    resource="resource:org.example/clinical-document@1.0.0",
    action="action:org.example/read@1.0.0",
    policy_profile="policy:org.example/minimum-necessary@1.0.0",
)
manifest = CapabilityGrantSigner(key).issue(
    [constraint],
    expires_at=2_000_000_000,
)

request = CapabilityGrantRequest(
    tool=constraint.tool,
    resource=constraint.resource,
    action=constraint.action,
    policy_profile=constraint.policy_profile,
)
verifier = CapabilityGrantVerifier(key)
result = dispatch_with_capability_grant(
    manifest,
    request,
    verifier,
    lambda: "local tool result",
    now=1_999_999_999,
)
```

The zero-argument callback keeps tool arguments outside the grant layer. It is
not invoked unless signature, expiry, and exact-scope verification all pass.
Pass `now` explicitly for deterministic replay and tests, or inject a local
integer clock into `CapabilityGrantVerifier` for runtime use.

## Canonicalization and key rotation

`CapabilityGrantSigner.issue()` requires an explicit Unix expiry timestamp.
It sorts constraints by all four fields, rejects duplicates, serializes with
sorted JSON keys and compact separators, and signs every manifest field except
`signature`. Equivalent inputs therefore produce byte-identical JSON and
signatures.

The manifest contains a non-secret `key_id`. Use
`MappingCapabilityGrantKeyProvider` for local key rotation or implement
`CapabilityGrantKeyProvider` to connect application-owned key custody. A
provider failure is reported only as `key_unavailable`; key material and
provider errors are never included in the grant exception.

Any added, removed, or changed constraint invalidates the signature. Unknown,
extra, or omitted manifest and constraint fields are rejected so a downstream
consumer cannot interpret unsigned metadata as authority. Expiry is exclusive:
a manifest is expired when `now >= expires_at`.

## Boundary of this contract

Capability grants answer whether a local agent may request an exact operation
class under an exact policy profile. They do not provide record-level access
tickets, permission delegation, human approval, replay protection, audit
ledgers, compliance certification, or an autonomous clinical decision
guarantee. They also do not execute a network request or grant host operating
system permissions.

Run the focused offline tests with:

```text
.venv/bin/python -m pytest tests/unit/agent/permissions/test_grants.py -q
```
