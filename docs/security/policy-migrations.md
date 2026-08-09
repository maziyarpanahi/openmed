# Policy migration reviews

Policy configuration is part of OpenMed's privacy boundary. A change from one
policy version to another should be reviewed as a migration, especially when a
redaction action, safety setting, or protected-label set changes.

The offline checker in `openmed.risk.policy_migration` compares two
`PolicyProfile` objects, JSON-like mappings, JSON documents, or local JSON
paths. It classifies the result as:

- `compatible` when no protection is weakened;
- `stricter` when protection only becomes stronger; or
- `incompatible` when a protection becomes weaker or a protected setting
  changes without a safe interpretation.

The comparison is deterministic and makes no network call. Reports contain
policy digests, safe schema paths, action names, aggregate counts, and hashes
or counts for arbitrary values. They never copy policy payloads into reports,
Markdown, JSON, or exceptions. Do not add source documents, patient text, or
cleartext identifiers to policy fixtures.

## Review and acknowledgement

Inspect a migration before enforcing it:

```python
from openmed.risk import check_policy_migration, compare_policy_versions

before = {
    "schema_version": 1,
    "default_action": "redact",
    "actions": {"EMAIL": "redact"},
}
after = {
    "schema_version": 1,
    "default_action": "mask",
    "actions": {"EMAIL": "mask"},
}

review = compare_policy_versions(before, after)
assert review.classification.value == "incompatible"

# Copy this only after a human has reviewed the safe diff.
approved = check_policy_migration(
    before,
    after,
    acknowledgement_token=review.acknowledgement_token,
)
assert approved.approved
```

`check_policy_migration` raises
`PolicyMigrationAcknowledgementRequired` when a weakening is not explicitly
acknowledged. The acknowledgement token is deterministic and bound to the two
policy digests and weakened paths; it is not a credential and does not provide
compliance certification or a clinical decision guarantee.

For ordinary policy evolution, prefer a stricter change or an explicit,
reviewed compatible metadata change. Keep the report as the review artifact,
not the original policy payload.
