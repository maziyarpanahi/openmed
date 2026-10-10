# FHIR transaction compensation reports

`openmed.interop.fhir.compensation_report.build_compensation_report()` compares
an in-memory FHIR R4 transaction or batch request Bundle with its response
Bundle. It returns an ordered, content-free review packet. It makes no network
request and never sends a corrective clinical write.

```python
from openmed.interop.fhir.compensation_report import build_compensation_report

packet = build_compensation_report(intended_bundle, received_bundle)
for effect in packet.effects:
    print(effect.entry_index, effect.classification.value, effect.proposed_action.value)
```

The caller supplies the original request and the actual server response. Pass
`None` as the response when delivery or acknowledgment is uncertain. Response
entries are paired with request entries by position, as in FHIR bundle
processing. A reported `fullUrl` that disagrees with the request forces review.
The packet contains only entry positions, methods, numeric statuses, fixed
reason codes, and fixed action codes. It omits resource bodies, identifiers,
locations, URLs, headers, and server diagnostics. Treat the input Bundles as
sensitive and keep them inside the authorized clinical system.

| Server observation | Classification | Proposed reviewer action |
| --- | --- | --- |
| `POST` with `201` and a location | Reversible candidate | Verify the created resource, dependencies, and current version; consider a separately approved deletion. |
| Successful `DELETE` | Irreversible from this packet | Review the deletion and determine recovery from authorized history or backup. |
| Successful `PUT` or `PATCH`, or a create without a location | Review required | Inspect prior state and current server state before proposing an undo. |
| Failed, missing, malformed, or uncorrelated response | Review required | Reconcile actual server state before any retry or compensation. |

“Reversible” means a possible compensation path, not proof that a delete is
clinically safe or that the resource still exists. A successful response is a
server report, not independent confirmation of persistence. `has_partial_failure`
is true only when at least one entry reports success and another is failed or
uncertain. Even when it is false, review the packet before acting. All actions
require human approval and fresh server checks; this module contains no
executor, approval token, action ledger, or autonomous recovery policy.

The action ledger (#2766), conditional write planning (#2773), and optimistic
concurrency guard (#2774) have separate contracts. Applications integrating
them must preserve their approval and version checks; this report does not
substitute for those controls.
