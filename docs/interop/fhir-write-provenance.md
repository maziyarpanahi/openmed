# FHIR write provenance gate

`openmed.interop.fhir.write_provenance.require_write_provenance()` checks a
proposed FHIR create or update before an authorized caller commits it. It uses
the same `WriteIntent` and private key as the reviewed side-effect preview,
plus one conditional write plan and one FHIR Provenance target per resource.
Every changed schema field needs a source span, an ordered transformation-step
digest chain, and a policy digest. Unchanged fields cannot acquire lineage.

```python
from openmed.interop.fhir.write_provenance import (
    ApprovalBinding,
    EvidenceSpan,
    FieldLineage,
    require_provenance_target,
    require_write_provenance,
    write_provenance_digest,
)

# intent and plan come from the caller's reviewed write flow.
# These identifiers and digests are synthetic examples.
handle = "res_" + "a" * 32
lineage = (
    FieldLineage(
        handle,
        "status",
        (EvidenceSpan("a" * 64, 5, 12),),
        ("b" * 64,),
        "c" * 64,
    ),
)
review_digest = write_provenance_digest(
    intent, {handle: plan}, lineage, {handle: "Observation/synthetic-123"},
    secret=private_key,
)
# The reviewer approves this complete digest; the receipt is verified locally.
approval = ApprovalBinding(review_digest, "d" * 64)
manifest = require_write_provenance(
    intent,
    {handle: plan},
    lineage,
    {handle: "Observation/synthetic-123"},
    approval,
    secret=private_key,
    verify_approval=verify_approval_receipt,
)
require_provenance_target(
    manifest.records[0], "Observation/synthetic-123", "Observation", secret=private_key
)
```

The example shows one field; a real call must include **every changed field**
in the write intent. The verifier must check the real approval token's decision,
reviewed action digest, expiry, and one-time use, then return `True` only for an
approved write. This module does not mint or consume approval tokens. A caller
must also verify that evidence digests and step digests resolve to its trusted
local evidence and transformation records. Do not treat arbitrary hex strings
as proof of evidence or approval.

The approval digest commits to the reviewed write preview, conditional plan
keys, each field's source offsets and transformation chain, policy digests, and
FHIR target commitments. A change to any of these inputs invalidates the
approval binding. The verifier must authenticate the receipt for this complete
digest, not merely the preview digest.

The manifest contains only schema paths, opaque resource handles, keyed source
digests and offsets, step and policy digests, the approval receipt digest, the
conditional plan key, and keyed commitments to FHIR Provenance targets. Source
digests should be keyed commitments when source text has low entropy. Actual
`ResourceType/id` references stay in caller memory and must be placed into
`Provenance.target` by the authorized FHIR transaction builder. For a new
resource without a server ID, use a transaction `urn:uuid` fullUrl and target
the same URN in that transaction. The target commitment lets the caller check
that the later Provenance target is the one reviewed by calling
`require_provenance_target()` for each record, without retaining a raw clinical
identifier in this packet.

Run this gate immediately before dispatch, after any fresh-read, capability,
match, and concurrency checks. Rebuild the preview and reverify approval if the
write intent changes. The gate performs no server call and does not dispatch,
persist, or guarantee atomicity of a FHIR write and its Provenance resource.
Integration with the guarded-output manifest (#2578), action ledger (#2766),
and approval-token implementation (#2768) remains with their respective
workflows. A FHIR executor must reject the write if any of those controls or
this gate fails.
