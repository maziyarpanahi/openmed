# Agent Capability Validity

`check_capability_validity` decides whether a capability grant is usable at a
given instant. It is a lifetime and audience check only: it does not issue
grants, verify signatures, evaluate authorization policy, or execute anything.

```python
from openmed.agent.capability_validity import (
    CapabilityGrant,
    check_capability_validity,
)

grant = CapabilityGrant(
    capability_id="capability:openmed.agent/summarize",
    audience="local-agent",
    issued_at=1_700_000_000,
    expires_at=1_700_003_600,
)
report = check_capability_validity(
    grant, now=1_700_001_800, expected_audience="local-agent"
)
print(report.status.value, report.reason_codes)
```

## What a grant carries

A grant is two identifiers and up to three integer epoch timestamps. There is
no field for a token, a signature, a key, a scope string, or tool arguments,
so none of those can enter the module or a report.

`capability_id` must be a canonical `capability:`
[governance identifier](governance-identifiers.md). `audience` is a local
service name matching `^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$` — a name,
deliberately not a URL, because a network endpoint is not something this check
should carry. `expires_at` must be strictly after `issued_at`, and an optional
`not_before` must sit inside that window.

## Verdicts

The caller supplies `now`, so the check is offline, reproducible, and
independent of the host clock. All findings are collected in one pass and the
worst status wins, ranked `valid < not_yet_valid < expired < rejected`.

| Reason code | Status | Meaning |
| --- | --- | --- |
| `audience_mismatch` | `rejected` | The grant was minted for a different audience. |
| `clock_skew_exceeded` | `rejected` | `issued_at` is further in the future than the tolerance allows. |
| `lifetime_exceeded` | `rejected` | The declared lifetime is longer than `max_lifetime_seconds`. |
| `grant_not_yet_valid` | `not_yet_valid` | `now` precedes the activation time. |
| `grant_expired` | `expired` | `now` follows the expiry. |

`rejected` means the grant is unusable whenever it is evaluated, while
`not_yet_valid` and `expired` are statements about this instant only.
Reason codes are returned in the fixed order above.

`max_clock_skew_seconds` (default 60, capped at `MAX_CLOCK_SKEW_SECONDS`)
widens both window edges symmetrically, so a grant is still usable a little
after expiry and a little before activation. It also bounds how far in the
future a grant may claim to have been issued; beyond that the grant is
rejected rather than merely reported as early. Zero skew makes the window
exact.

`seconds_until_expiry` is signed, so a caller can schedule a refresh from a
valid report and see how stale an expired one is.

## Failures

Structural problems fail closed with `CapabilityValidityError`, which carries
a stable `.code` and a `.field_name`. Timestamps must be integers, which
excludes booleans because `type(value) is int` is checked rather than
`isinstance`, and floats and numeric strings are refused rather than coerced.
Codes are `invalid_capability_id`, `invalid_audience`, `invalid_timestamp`,
`timestamp_out_of_range`, `expiry_not_after_issue`, `not_before_out_of_window`,
`invalid_bound`, `bound_out_of_range`, `lifetime_bound_invalid`,
`invalid_grant_type` and `invalid_schema_version`. A rejected value is never
echoed and a rejected identifier is not chained onto the raised error.

`to_dict()` preserves declared field order and `to_json()` sorts keys for
byte-identical payloads. A `valid` verdict is a lifetime statement, not an
authorization decision and not a clinical or security approval.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/agent/test_capability_validity.py -q
```

Fixtures are synthetic and no test reads the host clock. The table pins valid,
expired, premature, wrong-audience and excessive-skew grants; separate tests
cover both window edges at default and zero skew, `not_before`, the lifetime
ceiling, reason-code order, worst-status selection, serialization stability,
and every structural failure code.
