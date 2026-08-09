# Structured redaction contract

`openmed.structured.redaction_contract` provides a deterministic, local-only
contract for redacting selected scalar leaves in nested JSON resources. It is a
shape-preserving utility, not a FHIR compliance implementation or a clinical
decision system.

## Select paths explicitly

Paths use dots or slashes for object keys and brackets for array positions:

| Syntax | Meaning |
| --- | --- |
| `resource.subject.reference` | Named object keys |
| `entry[0].resource.id` | One exact array position |
| `entry[*].resource.name[*].text` | Every element of explicitly named arrays |

`[]` is accepted as a shorthand for `[*]`. A bare `*` is rejected because it
does not identify whether the caller intended an object key or an array index.
Rules that overlap, such as `entry[*].resource.id` and
`entry[0].resource.id`, are also rejected rather than applying two competing
actions to one scalar.

## Choose a scalar action

```python
from openmed.structured import (
    ACTION_HASH,
    ACTION_REMOVE,
    ACTION_REPLACE,
    RedactionContract,
    RedactionRule,
    redact_resource,
)

contract = RedactionContract(
    rules=(
        RedactionRule(
            "entry[*].resource.name[*].text",
            action=ACTION_REPLACE,
            replacement="[SYNTHETIC_NAME]",
        ),
        RedactionRule("entry[*].resource.identifier[*].value", action=ACTION_HASH),
    )
)

result = redact_resource(resource, contract)
released_resource = result.resource
safe_report = result.report.to_dict()
```

The supported actions are:

- `keep`: explicitly leave a selected scalar unchanged;
- `replace`: set a scalar to the supplied replacement, or `[REDACTED]` by
  default;
- `mask`: the same fixed-replacement behavior as `replace`, named for policy
  readability;
- `null`: set the selected scalar to JSON `null`;
- `remove`: remove an object member. An array element becomes `null` instead of
  being removed, so array positions remain stable;
- `hash`: replace the scalar with a deterministic SHA-256 digest. This is a
  stable digest for local linkage, not a claim of irreversible anonymization.

Replacement values must be JSON scalars. Mapping and list targets are rejected;
select their leaf fields instead. By default a `null` input remains `null`, so
redaction does not turn absent optional data into a replacement marker. Set
`preserve_null=False` on the contract or a rule when that behavior is
deliberately required.

## Structural invariants

The input is not mutated. The output always preserves:

- object key order for keys that remain;
- array order and array lengths, including when an element is removed;
- `resourceType`, `id`, and `fullUrl` fields, which are structural and cannot
  be transformed by a normal rule. Set `preserve_resource_identifiers=False`
  only when an explicitly reviewed policy is responsible for those fields.

Use `strict_paths=True` when a missing path should fail instead of being treated
as an optional field. No action performs a model download or network request.

## Raw-value-free evidence

`RedactionReport.to_dict()` contains only actions, counts, concrete schema paths,
and SHA-256 digests of the complete input and output. It does not include input
values or replacement values. `RedactionResult` hides the transformed resource
from its `repr`; log `result.to_audit_report()` when emitting audit metadata.

Exceptions similarly report contract or shape categories without echoing the
resource value that caused the failure. Keep committed examples and fixtures
synthetic, and do not use the report as a substitute for a formal privacy or
clinical-safety review.
