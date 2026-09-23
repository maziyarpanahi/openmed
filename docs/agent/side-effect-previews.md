# Side-effect previews for clinical writes

`openmed.agent.approvals.side_effect_preview` renders an offline, value-free
review view for typed resource writes. A preview lists each resource type and
opaque resource handle, each changed schema field path, whether each side is
absent or redacted, and the workflow state transition. A keyed digest commits
to the complete before and after values, action ID, write order, resource
handles, and workflow states. The module performs no network call or write.

```python
from openmed.agent.approvals.side_effect_preview import (
    ResourceWrite, WorkflowState, WriteIntent, WriteKind,
    render_side_effect_preview, require_current_preview,
)

secret = b"application-private-key-at-least-32-bytes"
current = {"status": "preliminary"}  # In memory only; use a fresh local read.
intent = WriteIntent(
    action_id="11111111-1111-4111-8111-111111111111",
    writes=(ResourceWrite(
        WriteKind.UPDATE, "Observation", "res_" + "a" * 32,
        current, {"status": "final"},
    ),),
    workflow_before=WorkflowState.AWAITING_REVIEW,
    workflow_after=WorkflowState.APPROVED,
)
preview = render_side_effect_preview(intent, secret=secret)
# Show preview.resources and the workflow transition to a reviewer.
# Keep the approved preview digest with the separate approval token.
require_current_preview(
    intent,
    observed_before=({"status": "preliminary"},),  # Fresh read at dispatch.
    observed_workflow_state=WorkflowState.AWAITING_REVIEW,
    approved_digest=preview.digest,
    secret=secret,
)
```

The application must generate a random UUIDv4 action ID and random `res_`
handles, then keep the handle mapping to real resources in its trusted local
review UI. Field paths must come from a
trusted schema, never from clinical payload keys. Flat paths such as `status`
or `subject.reference` are allowed; nested field values are compared as
atomic values. The preview intentionally shows no identifiers or values.
Its digest is a private keyed commitment, not a substitute for the review UI.
Keep the key private and stable through review and dispatch; rotating it
invalidates pending previews.

At dispatch, the application must supply newly observed resource and workflow
states. `require_current_preview` rejects any changed state or proposal with
`stale_preview`, requiring another review. It does not validate a human
approval token, read a target system, or dispatch the mutation. The caller
must validate the separate single-use approval token and use target-side
atomic preconditions such as FHIR `If-Match` to close the race after the fresh
read. For conditional creates, the caller must also resolve search ambiguity
and enforce the target's idempotency contract. Never log the raw write intent,
resource mapping, key, or clinical values.
