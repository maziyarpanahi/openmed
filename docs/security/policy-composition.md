# Privacy-policy composition

`openmed.risk.policy_composition` composes local privacy rules when field,
resource, and transport constraints overlap. It is a deterministic policy
gate, not a compliance certification or a clinical decision guarantee.

## Evaluation model

Each `PrivacyPolicy` has a scope, an `allow` or `deny` decision, and a selector:

| Scope | Selector | Inheritance |
| --- | --- | --- |
| `field` | exact field name or `*` | exact match |
| `resource` | slash-delimited resource path | parent paths inherit by default |
| `transport` | exact transport name or `*` | exact match |

Resource selectors may use `*` for one path component and a final `**` for
descendants. A resource rule with `inherit=False` applies only to its exact
path. A nested resource does not erase a parent rule: all applicable rules are
considered together.

The default scope precedence is explicit and most-specific-first:

1. field
2. resource
3. transport

Precedence selects a winner among rules with the same decision. `deny` always
overrides every applicable `allow`, including an allow at a more specific
scope or a child resource. This makes a broader transport or resource deny a
safe boundary for narrower rules. Rule priority is considered after scope and
resource specificity; a policy fingerprint is the final deterministic tie
breaker.

The default decision is `deny` when no rule matches. Callers may choose an
explicit default with `default_decision="allow"` when their surrounding policy
requires it.

## Value-free traces

`compose_policies(...)` returns a `PolicyDecisionResult` with a
`PolicyDecisionTrace`. The trace includes the effective decision, stable
conflict category, matching scopes, inheritance flags, precedence ranks, and
SHA-256 fingerprints for the policy set, context, selectors, and matching
rules. It does not serialize selectors, resource paths, transport names,
policy IDs, metadata, or other policy values. `to_dict()` and `to_json()` are
safe forms for reports and logs.

The stable conflict categories are:

| Category | Meaning |
| --- | --- |
| `none` | One explicit rule determined the result |
| `default` | No rule matched, so the configured default determined the result |
| `deny_overrides` | At least one deny and one allow matched |
| `multiple_denies` | Multiple denies matched and no allow was present |
| `precedence` | Multiple allows matched and precedence selected one |
| `inherited_deny` | A single inherited deny matched |
| `inherited_allow` | A single inherited allow matched |

All composition is in-process and dependency-free. It does not make a network
call, load a model, or read a remote policy source.
