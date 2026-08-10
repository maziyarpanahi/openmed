# Access-scope minimization evidence

`openmed.compliance.access_scope` is a local, deterministic gate for proving
that a workflow requests only the resource/action scopes it uses. It compares
three caller-supplied sets:

- **requested** — scopes the workflow asks for;
- **used** — concrete scopes observed during the workflow; and
- **approved** — the maximum scope the local policy approves.

The evaluator does not contact a registry, inspect ambient credentials, read a
clock, or transmit workflow data. It is technical evidence, not a compliance
certification, access grant, or clinical decision guarantee.

## Evaluate a minimal request

Scopes use the structured `resource:action` form. Components are normalized to
lower case, deduplicated, and sorted. A mapping is also accepted when it is
convenient to express one resource with several actions.

```python
from openmed.compliance import evaluate_access_scope

result = evaluate_access_scope(
    requested={"records:read"},
    used={"records:read"},
    approved={"records:*"},
)

assert result.allowed
assert result.decision == "allow"
print(result.to_json())
```

The gate blocks when:

- a concrete used scope is neither requested nor covered by an explicit
  escalation rule (`undeclared_use`);
- a requested or used scope is not covered by approval
  (`unapproved_request` or `unapproved_use`); or
- a requested declaration matches no observed use (`overbroad_request`).

An approved set may be broader than the request. That is a policy ceiling, not
permission for the workflow to use every approved scope.

## Wildcards and escalations are explicit

An approved wildcard such as `records:*` can approve a narrow concrete request.
Wildcards in `requested` or in an escalation rule are blocked unless the
caller explicitly opts in:

```python
from openmed.compliance import evaluate_access_scope

result = evaluate_access_scope(
    requested={"records:*"},
    used={"records:read"},
    approved={"records:*"},
    wildcard_rules={"records:*"},
)
assert result.allowed
```

Observed use must always be concrete. A used wildcard is blocked even when an
approved wildcard exists, because an unbounded observation cannot demonstrate
least privilege.

An operation outside the requested set must have an explicit escalation rule,
and that rule must itself be covered by approval:

```python
result = evaluate_access_scope(
    requested={"records:read"},
    used={"records:read", "records:export"},
    approved={"records:read", "records:export"},
    escalation_rules={"records:export"},
)
assert result.allowed
assert result.counts.escalated_used == 1
```

## Count-only evidence

`AccessScopeEvaluation.to_dict()`, `to_json()`, and
`render_access_scope_evidence()` intentionally emit counts, stable reason
codes, and rule settings only. They do not include resource names, action
names, request content, records, identities, credentials, or raw sensitive
values. A blocked evaluation can be raised with `enforce_access_scope()`;
`AccessScopeViolationError` contains the same count-only report and its error
message contains only reason codes and counts.

```python
from openmed.compliance import (
    AccessScopeViolationError,
    enforce_access_scope,
)

try:
    enforce_access_scope(
        requested={"records:read"},
        used={"records:read", "records:write"},
        approved={"records:read", "records:write"},
    )
except AccessScopeViolationError as error:
    print(error.report.to_json())
```

Use synthetic scope metadata in tests and retain any detailed workflow audit
records outside this aggregate evidence surface. This helper does not inspect
or validate the underlying resource contents.
