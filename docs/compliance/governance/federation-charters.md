# Federation charters

`FederationCharter` declares permitted task codes and roles, decision quorum,
reviewer quorum, per-round epsilon/delta ceilings, incident hold policy, appeal
review policy, termination disposition and final permitted round sequence.
`CharterHistory` retains sequential amendments with strictly increasing activation
rounds. Each canonical policy has a domain-separated SHA-256 digest; exports link
each amendment to the previous policy digest.

`validate_federation_round(history, declaration)` resolves the active policy for
the round sequence and checks its exact version and digest. It retains every
applicable hold reason: unapproved tasks/roles, missing quorum, excessive declared
privacy budgets, unresolved incidents/appeals, termination and expired round
limits. Reports contain only fixed reasons, outcome, active version and digest.
Invalid contracts raise one controlled error without echoing input values.

Use anonymous role declarations and aggregate vote counts, never participant
identities. Only the participant, coordinator, reviewer and incident-responder
roles are supported. Task codes use lowercase ASCII metadata vocabulary. Counts
are bounded at one million and histories at 128 amendments. Decimal budgets must
be finite, nonnegative, epsilon at most 1000, delta below one, and no more than
18 fractional places. These limits are serialization bounds, not recommended
privacy budgets.

This is an offline declaration gate. It does not authenticate voters, prove
independent review, enroll participants, execute deletion, verify a cryptographic
signature, compose privacy accounting, or certify compliance. An `accept` result
means only that the supplied round declarations satisfy the active charter.
The orchestrator must obtain authorized evidence, execute incident/appeal and
termination policies, and separately account for cumulative privacy expenditure.

Run the synthetic controls with:

```sh
.venv/bin/python -m pytest tests/unit/compliance/governance/test_federation_charter.py -q
```
