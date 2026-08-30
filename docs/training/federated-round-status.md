# Federated round status

OpenMed can render a federated training round as a deterministic operator
summary without publishing participant identities or local training data. The
summary accepts aggregate counts and digest references only. It has no fields
for client IDs, site names, patient counts, local losses, gradients, or
per-client metrics.

## Disclosure rules

Participant and completed-participant counts are released only when each count
is at least the configured `minimum_group_size`, which defaults to `5`. Smaller
counts are replaced with a suppression marker in both JSON and Markdown. The
minimum group size must be at least `2`.

Quorum is calculated before suppression, so operators still see `met` or
`not_met` when the exact participant count is hidden. The required quorum is
not included in the rendered summary.

Completion is reported as one of four coarse bands:

| Band | Aggregate meaning |
| --- | --- |
| `not_started` | No participants have completed |
| `under_half` | More than zero but fewer than half have completed |
| `half_or_more` | At least half but fewer than all have completed |
| `complete` | Every participant has completed |

## Usage

```python
from openmed.training import (
    FederatedRoundReasonCode,
    FederatedRoundState,
    build_federated_round_status,
)

status = build_federated_round_status(
    state=FederatedRoundState.HELD,
    participant_count=8,
    completed_participant_count=8,
    required_quorum=5,
    minimum_group_size=5,
    aggregate_digest_refs=("sha256:" + "a" * 64,),
    reason_code=FederatedRoundReasonCode.QUALITY_REVIEW_REQUIRED,
)

json_artifact = status.to_json()
markdown_artifact = status.to_markdown()
```

Digest references must be lowercase `sha256:` values. Duplicate references are
rejected and valid references are sorted before rendering, so caller ordering
cannot change the artifact. Held and aborted rounds require a reason from their
respective stable reason-code sets; other lifecycle states reject reason codes.

| Round state | Allowed reason codes |
| --- | --- |
| `held` | `privacy_review_required`, `quality_review_required`, `policy_review_required` |
| `aborted` | `quorum_not_met`, `privacy_gate_failed`, `quality_gate_failed`, `policy_gate_failed`, `operator_cancelled`, `round_error` |

## Privacy boundary

The builder uses exact aggregate counts only to calculate quorum, completion,
and disclosure. It immediately replaces sub-threshold counts with `null` in the
immutable status object, so later renderers cannot recover the omitted number.

Minimum-group suppression is an output control, not differential privacy or a
privacy accountant. Transport, client attribution, local metrics, and privacy
budget enforcement remain outside this module.
