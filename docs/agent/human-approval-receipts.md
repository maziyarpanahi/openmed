# Human-approval receipt failures

Human-approval verification must fail closed before a high-impact action is
dispatched. The runnable example in
`examples/agent_approval_failures.py` exercises five deterministic failures
against the approval-token contract:

| Scenario | Stable reason | Result |
| --- | --- | --- |
| Token reaches its exclusive expiry | `expired` | Reject before claiming the nonce |
| Token is presented a second time | `replayed` | Reject the replay |
| Reviewed action digest differs | `action_mismatch` | Consume the token and require fresh approval |
| Required reviewer role differs | `reviewer_role_mismatch` | Consume the token and require fresh approval |
| Token schema is unsupported | `unsupported_schema_version` | Reject before signature or nonce processing |

Run the example from the repository root:

```bash
python examples/agent_approval_failures.py
```

It runs locally without network access and prints only scenario names and
stable reason codes. Fixed timestamps, digests, roles, nonces, and signing key
material are synthetic test inputs. Never reuse the example key or deterministic
nonces in an application; production issuance needs protected key custody and a
cryptographically secure nonce source.

## Verification failure is not clinical denial

A verification failure means the presented approval evidence is invalid for
the exact dispatch attempt. It does **not** mean that a reviewer denied the
clinical judgment, that the proposed action is clinically incorrect, or that a
different reviewer role would approve it. The application must not dispatch the
action and should request a new review when appropriate.

A human denial is a separate application-level decision. Record it through the
workflow's governed decision or outcome contract, not by manufacturing an
expired, mismatched, replayed, or malformed approval token. The failure reason
codes above describe only mechanical verification outcomes.

## Fail-closed integration

Catch the specific approval exception needed for local control flow and use its
stable `code` attribute. Do not branch on exception prose, and do not add action
payloads, clinical content, reviewer identities, bearer tokens, nonces,
signatures, or keys to logs. A mismatch in action digest or reviewer role
consumes the valid signed token, so retrying it correctly produces `replayed`;
obtain a fresh human approval instead.
