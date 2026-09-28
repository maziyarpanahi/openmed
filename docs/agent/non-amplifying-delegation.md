# Non-amplifying clinical-agent delegation

`openmed.agent.permissions.delegation` provides a signed, offline contract for
delegating a bounded task from one local clinical agent to another. A child can
receive only the intersection of the active parent's authority and the
requested task scope.

The delegated scope covers three independent dimensions:

- exact capability constraints (tool, resource, action, and policy profile);
- developer-authored data classes; and
- developer-authored purposes.

The child's exclusive expiry is the earlier of the parent and request expiry.
Its remaining delegation depth is the smaller of the requested depth and one
less than the parent's remaining depth. Empty intersections, expired parents,
exhausted depth, repeated principals, invalid signatures, and parent-child
links that broaden any dimension fail closed.

## Derive a signed child grant

The application supplies at least 32 bytes of locally managed signing key
material. No key, grant, request, or audit record is transmitted by this
module, and no network call is performed.

```python
from openmed.agent.permissions import (
    CapabilityGrantConstraint,
    DelegationGrantSigner,
    DelegationGrantVerifier,
    DelegationRequest,
    DelegationScope,
)

key = b"replace-with-32-or-more-local-key-bytes"
capability = CapabilityGrantConstraint(
    tool="tool:org.example/redact@1.0.0",
    resource="resource:org.example/clinical-document@1.0.0",
    action="action:org.example/read@1.0.0",
    policy_profile="policy:org.example/minimum-necessary@1.0.0",
)
parent_scope = DelegationScope(
    capabilities=(capability,),
    data_classes=("data:org.example/medications@1.0.0",),
    purposes=("purpose:org.example/care-summary@1.0.0",),
)

signer = DelegationGrantSigner(key)
verifier = DelegationGrantVerifier(key)
parent = signer.issue_root(
    principal="agent:org.example/coordinator@1.0.0",
    scope=parent_scope,
    expires_at=2_000_000_000,
    remaining_depth=2,
)
decision = signer.derive_child(
    parent,
    DelegationRequest(
        principal="agent:org.example/redactor@1.0.0",
        scope=parent_scope,
        expires_at=1_999_999_900,
        remaining_depth=0,
    ),
    verifier,
    now=1_999_999_000,
)
child = decision.grant
```

`issue_root()` is an application trust-boundary operation. It does not infer
authority from host permissions. The trusted local issuer must construct the
root from authority it has already approved. `derive_child()` first verifies
the parent's signature and active validity, intersects all requested scope,
then signs and re-verifies the child link.

The request may include authority outside the parent. Such entries are removed
by set intersection; they never appear in the child. If any authority
dimension has an empty intersection, derivation is rejected instead of issuing
a grant that cannot represent the requested task.

## Verify the complete ancestry

Each child signs its full principal chain and the SHA-256 digest of its direct
parent. Verify the root-to-leaf sequence before relying on a grant received
from another component:

```python
leaf = verifier.verify_chain(
    (parent, child),
    now=1_999_999_000,
)
```

For each parsed grant, chain verification checks its signature before comparing
its signed authority with its parent. It then requires exact parent-digest
linkage, one-principal chain extension, no repeated principal, subset authority,
non-increasing validity, and at least one consumed depth level. A component
that knows the shared HMAC key can sign, so keep issuance in a more trusted
local component when issuer/verifier role separation matters.

Pass `now` explicitly for deterministic replay and tests, or inject a local
integer clock into `DelegationGrantVerifier`. Expiry is exclusive: a grant is
inactive when `now >= expires_at`.

## Digest-only audit evidence

Successful derivation returns a `DelegationDecision` with a signed grant and a
`DelegationAuditRecord`. Rejection exceptions expose the same fixed audit
schema:

```json
{
  "schema_version": "openmed.agent.delegation_audit.v1",
  "reason_code": "scope_amplified",
  "parent_grant_digest": "sha256:<digest>",
  "child_grant_digest": "sha256:<digest>"
}
```

Only stable reason codes and whole-grant digests belong in delegation audit
records. Do not log or copy the grant, request, principal chain, capability
identifiers, purposes, data classes, signing keys, record identifiers, tool
arguments, or clinical values. A digest is linkable metadata and should remain
under the workflow's audit-data controls.

Agent, capability, data-class, and purpose values are developer-authored
governance identifiers. Never embed a patient, clinician, tenant, encounter,
record, credential, endpoint, or clinical value in them, and do not hash such
a value into an identifier to work around this boundary.

## Relationship to grants and access tickets

A capability-grant manifest verifies an exact operation class. An access
ticket verifies one run's purpose, minimum data projection, opaque record
selectors, and tool action. A delegation grant limits which operation classes,
data classes, and purposes can flow from a parent agent to a child. These are
independent checks; delegation does not replace capability verification or a
run-bound access ticket.

This contract does not execute a tool, grant operating-system access, establish
consent, certify compliance, or authorize autonomous clinical decisions.

Run the focused offline tests with:

```text
.venv/bin/python -m pytest tests/unit/agent/permissions/test_delegation.py -q
```
