# FHIR conditional write plans

`openmed.interop.fhir.conditional_writes` builds local, typed FHIR R4
conditional-create and conditional-update plans. It makes no network request,
holds no resource payload, and never executes a write. The caller supplies a
private deployment secret and explicit matching criteria.

```python
from openmed.interop.fhir.conditional_writes import (
    ConditionalWriteKind,
    assess_matches,
    build_conditional_write_plan,
)
from uuid import UUID

# The value and key below are synthetic. Load the real key privately at runtime.
secret = b"synthetic-deployment-secret-32-bytes-long"
plan = build_conditional_write_plan(
    ConditionalWriteKind.CREATE,
    "Observation",
    {"identifier": "urn:synthetic|example"},
    operation_id=UUID("00000000-0000-4000-8000-000000000001"),
    secret=secret,
)
decision = assess_matches(plan, 0, search_complete=True)
decision.require_ready()
```

The predicate is sorted by search-parameter name and encoded with RFC 3986
percent encoding. `plan.predicate.canonical_query` is sensitive request content:
use it only in the authorized request path or `If-None-Exist` header. Never put
it in logs, exceptions, traces, or reports. The plan and predicate redact it
in their representations. A single value per parameter is supported; repeated
parameters and comma-separated OR values are rejected rather than assigning
server-dependent match semantics.

The stable `fhir-cw-v1-` key is an HMAC-SHA256 commitment to an opaque operation
UUID, interaction, resource type, and canonical predicate, separated by a
versioned domain. Persist the same UUID and private secret across retries of
one operation. Use a new UUID for a distinct intended write, even when its
predicate is unchanged. Rotate the secret only when those retries have ended.
The key is planning metadata. A server may not
support an idempotency header, so the key by itself never guarantees a
duplicate-free write. The caller must use FHIR conditional request semantics
and still handle concurrent changes or server conflicts.

Before any request, check the already-cached server `CapabilityStatement` with
[`preflight_write_plan()`](fhir-write-preflight.md), setting `conditional=True`
for the same create or update interaction. Search for the plan's predicate and
evaluate a **complete** result set with `assess_matches()`:

| Interaction | Matches | Decision |
| --- | --- | --- |
| Create | 0 | Ready for explicit conditional create |
| Create | 1 | No-op: existing match |
| Update | 1 | Ready for explicit conditional update |
| Update | 0 | Review: a server may create on conditional update |
| Either | More than 1 | Review: ambiguous matches |
| Either | Incomplete search | Review: match count is unreliable |

`require_ready()` raises a value-free exception for every no-op or review
decision. A reviewer must resolve ambiguous matches; the planner never picks a
resource. Search results can change after this check. An executor must also
stop on server-side ambiguity and conflicts, and must satisfy its own approval,
concurrency, provenance, and recovery controls before any write. These controls
are tracked separately by issues #2774, #2777, #3199, and #3197.
