# Agent Governance Identifiers

`openmed.agent` provides five typed, developer-authored identifiers for naming
agent governance metadata:

- `CapabilityId` names a capability.
- `PurposeId` names an intended purpose.
- `PolicyId` names a policy.
- `WorkflowId` names a workflow.
- `ToolId` names a tool.

These are names, not generated correlation tokens. They are accepted only in
canonical form and are returned unchanged by `serialize()` and `str()`.

## Spelling and exact grammar

The complete spelling is:

```text
<kind>:<reverse-domain>/<local-name>[@<version>]
```

The grammar below is the grammar implemented by the typed parsers. All
terminals are ASCII; the length constraints following the grammar are
independent constraints.

```text
identifier = kind, ":", namespace, "/", local_name, [ "@", version ] ;

kind = "capability" | "purpose" | "policy" | "workflow" | "tool" ;

namespace = label, ".", label, { ".", label } ;
label = alnum | alnum, { alnum | "-" }, alnum ;

local_name = letter, { letter | digit | "-" } ;

version = number, ".", number, ".", number ;
number = "0" | nonzero_digit, { digit } ;

alnum = letter | digit ;
letter = "a" … "z" ;
digit = "0" … "9" ;
nonzero_digit = "1" … "9" ;
```

The additional constraints are:

- Each namespace label is 1–63 characters, contains only lowercase ASCII
  letters, digits, and internal hyphens, and cannot start or end with a
  hyphen.
- A namespace has at least two non-empty labels and is at most 253 characters
  including dots. Numeric labels and labels beginning with a digit are valid.
- A local name is 1–64 characters, starts with a lowercase ASCII letter, and
  then contains only lowercase ASCII letters, digits, or hyphens. A trailing or
  repeated hyphen is valid in this field.
- A complete identifier is at most 512 ASCII characters. It contains exactly
  one structural slash and no backslash, path segment, URL scheme, query,
  fragment, control character, or other non-ASCII character.
- A version, when present, has exactly three non-negative decimal components.
  Each component uses `[0-9]` and has no leading zero unless it is exactly
  `0`. There are no `v` prefixes, prerelease labels, or build metadata.

For example:

```python
from openmed.agent import PolicyId, ToolId, WorkflowId

policy = PolicyId.parse("policy:org.example/default")
workflow = WorkflowId("workflow:org.example/document-intake@0.1.0")
tool = ToolId.parse("tool:org.example/redact@1.2.3")

assert policy.version is None
assert workflow.serialize() == "workflow:org.example/document-intake@0.1.0"
assert tool.namespace == "org.example"
```

`ToolId.parse()` accepts only a `tool:` identifier. The other typed parsers
apply the same rule to their own kind, so a valid `policy:` string is rejected
by `ToolId.parse()` rather than being reclassified. Unknown and mixed-case
kinds are rejected as well. Parsing never trims, lowercases, decodes,
normalizes, repairs separators, or supplies a missing version.

## Diagnostics and provenance boundary

Failures raise `GovernanceIdError`, a `ValueError` subclass. Its `code` and
fixed `field_name` are stable diagnostics such as `invalid_identifier`,
`invalid_identifier_type`, `wrong_kind`, `unknown_kind`, `namespace_too_long`,
or `identifier_too_long`. Exception messages, arguments, attributes, and
object `repr()` values do not contain the rejected identifier. The explicit
data interfaces `serialize()` and `str()` are the only intended ways to obtain
the canonical value.

Governance identifiers are developer-authored names. Do not derive them from
patient, clinician, tenant, or device identities or from clinical content, and
do not transform such content into an identifier with hashing, encoding,
truncation, or slugification. Syntax validation only proves that a string has
the required shape. It does not prove that the caller used an appropriate
source, owns the namespace, or is authorized to use the name; this module
cannot identify every syntactically valid value with improper provenance.

## Relationship to correlation IDs

`RunId` and `ActionId` remain separate contracts. They are opaque, randomly
generated correlation identifiers for a run and an action, using their existing
`run_` and `act_` forms. Governance identifiers are stable, human-chosen names
for metadata. This change does not migrate existing consumers, add a registry,
make authorization decisions, or perform DNS, ownership, network, or clinical
content checks.

Run the focused offline tests with:

```text
uv run --frozen --extra dev pytest tests/unit/agent/test_identifiers.py -q
```
