# SMART scope audit

`openmed.interop.smart_scope_audit` compares requested SMART v2 scopes with
the resource and launch-context permissions a workflow declares. Run it before
an agent obtains credentials or clinical data. The comparison is local and
does not perform OAuth, inspect tokens, contact endpoints, or grant access.

```python
from openmed.interop.smart_scope_audit import audit_smart_scopes

audit = audit_smart_scopes(
    required_scopes=("patient/Observation.rs", "launch/patient"),
    requested_scopes=("patient/*.r", "launch"),
)
if not audit.is_least_privilege:
    send_for_review(audit.to_dict())
```

The workflow declaration is a list of scope names, without patient IDs or
other workflow payloads. Supported clinical scopes use `patient`, `user`, or
`system` context, a FHIR resource type or `*`, and SMART v2 operations `c`
(create), `r` (read), `u` (update), `d` (delete), and `s` (search). The helper
also accepts `launch`, `launch/patient`, and `launch/encounter`. It normalizes
operation order and deduplicates scopes. SMART v1 `.read` and `.write`, other
OAuth scopes, and custom launch contexts are outside this audit.

The report contains only reason codes, normalized scope names, and resource
types. `missing_scope` means the request does not cover a declared need.
`excessive_scope` means it requests an unneeded operation or launch context.
`overbroad_resource` means a wildcard requests access beyond the workflow's
specific resource types. A wildcard can satisfy a specific required scope,
while still being overbroad. Any finding calls for operator review before a
run; the helper does not authorize a run or recommend production permissions.

The separate offline examples tracked in issue #3112 can demonstrate this
API with synthetic workflows once their PR is reconciled with this module.
