# Privacy release-gate aggregation

`openmed.compliance.privacy_gate` combines local risk, policy, evidence, and
dependency gate results into one deterministic release record. It is a
technical release aid, not a compliance certification or clinical decision
guarantee.

## Create a decision

Gate producers should provide a stable identifier, one explicit state, and an
aggregate count. A producer can also provide a SHA-256 fingerprint over its
private finding set:

```python
from openmed.compliance import PrivacyGateResult, aggregate_privacy_gates

record = aggregate_privacy_gates(
    [
        PrivacyGateResult("risk", "blocking", finding_count=1),
        PrivacyGateResult("policy", "warning", finding_count=2),
        PrivacyGateResult("dependency", "waived", waiver_code="reviewed"),
    ]
)
```

The states are deliberately explicit:

| Gate state | Meaning |
|---|---|
| `blocking` | A finding prevents release. |
| `warning` | Release needs review but is not blocked by this gate. |
| `waived` | The gate is acknowledged as waived and remains visible in the record. |

Aggregate precedence is `blocked` when any gate is blocking, `warning` when
there are warnings but no blockers, and `released` when only waived gates
remain. Waived counts are never silently discarded.

## Safe record format

`record.to_dict()` and `record.to_json()` contain the decision, counts for all
three states, sorted gate summaries, per-gate fingerprints, and an aggregate
record fingerprint. They do not contain findings, free-text reasons, source
values, identifiers, or arbitrary metadata. `from_findings` can be used at a
local boundary when a producer has a sized collection; only its length is
retained.

The implementation uses the local Python runtime and stable hashing only. It
performs no network calls and has no telemetry or remote service dependency.

## Federated update egress approval

A federated model update is a data product that leaves the institution, so
`openmed.training.federated_egress_report` requires a site approval of a
value-free report before an update is submitted. The module never seals,
transports, or opens tensors; sealing, the approval authority, the clock, and
the transport are injected.

```python
# Fragment: metadata, sealed, approver, clock and transport are supplied by the site.
from datetime import timedelta

from openmed.training.federated_egress_report import (
    FederatedEgressClipping,
    FederatedEgressMechanism,
    FederatedEgressPrivacySpend,
    FederatedEgressSubmissionGuard,
    approve_federated_egress,
    render_federated_egress_report,
)

report = render_federated_egress_report(
    metadata,  # validated FederatedUpdateMetadata
    sealed_update=sealed,  # the exact bytes that would leave the site
    round_digest=round_digest,
    clipping=FederatedEgressClipping("per_example", "l2", clip_config_digest),
    mechanism=FederatedEgressMechanism("gaussian", "v1", "rdp", noise_digest),
    privacy_spend=FederatedEgressPrivacySpend(
        epsilon_spent, epsilon_budget, delta_spent, delta_budget, ledger_digest
    ),
)
approval = approve_federated_egress(
    report, approver=approver, clock=clock, valid_for=timedelta(hours=4)
)
guard = FederatedEgressSubmissionGuard(approver=approver, clock=clock)
guard.submit(
    sealed,
    report=report,
    approval=approval,
    round_digest=round_digest,
    transport=transport,
)
```

The report (`to_json()` or `to_markdown()`) is byte-stable and contains only
parameter names, shapes, dtypes and element counts, the update metadata
digests and schema fingerprint, clipping and noise-mechanism codes with
configuration digests, epsilon and delta spend bands, the privacy ledger
digest, the round digest, and the SHA-256 digest of the sealed bytes. Exact
spend and budget values stay local and are hidden from `repr`. Spend above a
budget, an update that does not declare clipping, and any report field in
`FORBIDDEN_EGRESS_REPORT_FIELDS` (site and client identifiers, paths,
endpoints, examples, text, tensors, gradients, messages, local metrics, and
patient counts, at any nesting level) fail closed. Parsing is strict: unknown
fields, duplicate keys, floats, inconsistent element counts, and unsorted
parameters are refused.

The injected approver receives the report and a domain-separated statement
binding the report digest, sealed update digest, round digest, approval time,
and expiry, and returns an opaque signature or `None` to decline. Approval
windows are whole seconds and at most seven days. Before calling `transport`, the
guard refuses a missing approval, an approval for a different report, different
sealed bytes, or another round, a signature that does not verify, and a clock
reading before approval or at or after expiry. Every refusal raises
`FederatedEgressError` with a stable `reason_code` and a fixed message; failures
in the approver, clock, or transport are reported as codes without chained
exception context.

Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/test_federated_egress_report.py -q
```
