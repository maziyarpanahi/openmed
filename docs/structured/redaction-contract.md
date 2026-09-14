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

Only `[*]` denotes an array wildcard. Empty brackets and a bare `*` are rejected
because they are ambiguous. Rules that overlap at the same value or through an
ancestor path, such as `entry[*].resource` and
`entry[0].resource.name[0].text`, are also rejected rather than applying
competing transformations.

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

Mapping policies are closed: each path maps to a supported action string or an
object containing `action` and, when applicable, `replacement` and
`preserve_null`. Unknown options and implicit scalar replacements fail closed.

## Structural invariants

The input is not mutated. The output always preserves:

- object key order for keys that remain;
- array order and array lengths, including when an element is removed;
- `resourceType`, `id`, and `fullUrl` fields, which are structural and cannot
  be transformed by a normal rule. Set `preserve_resource_identifiers=False`
  only when an explicitly reviewed policy is responsible for those fields.

The resource root cannot be transformed, including when identifier preservation
is disabled. This prevents a container-level action from bypassing structural
field checks.

Use `strict_paths=True` when a missing path should fail instead of being treated
as an optional field. No action performs a model download or network request.

## Bounded inputs

Contracts accept at most 256 rules, 256 preserved paths, 32 identifier keys,
and 64 segments or 4,096 characters per path. Resources are limited to 64
levels, 100,000 nodes, 10,000 items per object or array, 10,000 redaction
matches, one million characters per string, and ten million string characters
in total. Integers must fit signed 64-bit range, and explicit replacement
strings are capped at 65,536 characters. Object keys must be non-empty,
printable strings of at most 256 characters.

Inputs are copied into plain local dictionaries and lists before matching. A
cycle, hostile container protocol, unsupported subtype, or exceeded bound fails
before the output is returned.

## Raw-value-free evidence

`RedactionReport.to_dict()` contains only a schema version, counts, concrete
schema paths, and SHA-256 digests of the complete input and output. It does not
include input values, actions, or replacement values. Reports validate their
count relationships, canonical paths, and digest syntax even when callers
construct them directly. `RedactionResult` hides the transformed resource from
its `repr`; log `result.to_audit_report()` when emitting audit metadata.

Exceptions similarly report contract or shape categories without echoing the
resource value that caused the failure. Keep committed examples and fixtures
synthetic, and do not use the report as a substitute for a formal privacy or
clinical-safety review.
