# FHIR optimistic concurrency guard

`openmed.interop.fhir.concurrency_guard` provides local checks for an explicit
FHIR R4 resource update. It does not fetch resources, perform a write, log
clinical content, or merge clinical changes. The caller must separately pass
capability preflight, approval, and any conditional-write match checks.

Capture both `meta.versionId` and `meta.lastUpdated` from the original read.
Immediately before the authorized update, fetch the same resource again and
compare both fields. If they differ, stop for review. Send the returned
`If-Match` header with the update so the server checks the version atomically
at commit time:

```python
from openmed.interop.fhir.concurrency_guard import (
    VersionEvidence,
    guard_update,
    require_no_server_conflict,
)

expected = VersionEvidence.from_resource(original_resource)
observed = VersionEvidence.from_resource(fresh_resource)
precondition = guard_update(expected, observed)
response = authorized_client.put(
    resource_url,
    json=proposed_resource,
    headers={"If-Match": precondition.if_match},
)
require_no_server_conflict(response.status_code)
```

The example's client, resources, and URL are supplied by the application. Do
not send an update when either evidence field is missing. A local comparison
alone cannot close the race after the fresh read. The server must honor
`If-Match`; an executor must confirm that behavior for its target server and
stop if it cannot. HTTP `409`, `412`, and `428` map to `FHIRWriteConflict` with
fixed reason codes, without parsing or copying a server response body. Other
error statuses still require the caller's normal error handling.

For a conflict, `summarize_conflict(original_resource, fresh_resource,
proposed_resource)` reports counts of server changes, proposed changes,
overlaps, divergent overlaps, and already-applied overlaps. It exposes no
resource paths or values. FHIR `meta` is excluded from clinical change counts,
and arrays are treated as single values. This is a review aid, not a merge
algorithm or authorization to retry. A reviewer must inspect the actual
resources in the authorized clinical system before any new write attempt.

This guard is separate from [conditional write planning](fhir-conditional-writes.md).
The planner's stable key does not enforce optimistic concurrency. The
precondition header and typed conflict handling apply to an explicit update
request, including one selected through an approved conditional workflow.
