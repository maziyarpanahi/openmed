# Privacy-safe agent policy decision matrices

`openmed.agent.PolicyDecisionMatrix` renders a compact view of policy
decisions for tests and human review. It compares governance metadata without
copying the tool input or any clinical content into the artifact.

## Matrix contract

Each row contains exactly:

- an explicitly versioned `PolicyId`;
- a `CapabilityId`, `PurposeId`, and `ToolId`;
- one reason code from the closed workflow outcome vocabulary; and
- a Boolean reviewer-required flag.

The decision key is the tuple of policy version, capability ID, purpose ID, and
tool ID. `PolicyDecisionMatrix.from_rows()` sorts by that complete key and
rejects duplicate keys, including duplicates whose result fields disagree.
Empty matrices are valid. JSON is compact and key-sorted; Markdown uses the
same stable row order. Repeated renders of the same rows are therefore
byte-identical.

```python
from openmed.agent import (
    CapabilityId,
    PolicyDecisionMatrix,
    PolicyDecisionRow,
    PolicyId,
    PurposeId,
    ToolId,
)

row = PolicyDecisionRow(
    policy_version=PolicyId("policy:org.example/agent-access@1.0.0"),
    capability_id=CapabilityId("capability:org.example/read-evidence"),
    purpose_id=PurposeId("purpose:org.example/care-review"),
    tool_id=ToolId("tool:org.example/search-evidence"),
    outcome_reason_code="human_gate",
    reviewer_required=True,
)
matrix = PolicyDecisionMatrix.from_rows([row])

json_artifact = matrix.to_json()
markdown_artifact = matrix.to_markdown()
```

Use `from_dict()` or `from_json()` at trust boundaries. They reject unknown or
missing fields, malformed or wrong-kind identifiers, unversioned policies,
unknown outcome codes, non-Boolean review flags, duplicate JSON fields, and
duplicate decision keys. Errors contain only stable codes and public field
names; rejected values are not echoed.

## Privacy and authority boundary

The schema has no fields for prompts, arguments, outputs, evidence text,
bearer values, filesystem paths, free-text notes, or patient and clinician
identifiers. Governance identifiers must be developer-authored names and must
not be derived from clinical content or identities.

The matrix reports decisions supplied by a caller. It does not evaluate a
policy, grant a capability, authorize a clinical action, execute a tool, read
artifact content, or persist an audit record. A reviewer-required value is
metadata for a separately governed reviewer workflow, not proof that review
occurred.

Run the focused offline tests with:

```text
.venv/bin/python -m pytest tests/unit/agent/test_policy_matrix.py -q
```
