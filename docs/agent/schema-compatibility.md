# Agent artifact schema compatibility

`openmed.agent.schema_compatibility` provides explicit, content-free compatibility
verification for agent-produced artifacts. Run summaries, handoff packets, policy
matrices, and evidence references evolve independently across versions. Consumers
use this module to enforce schema boundaries rather than silently accepting
unsupported schemas.

## Compatibility outcomes

Evaluation returns one of four closed outcomes with a stable reason code:

- `compatible`: The incoming schema version falls within the caller's declared
  inclusive range (`min_version <= incoming <= max_version`).
  Reasons: `exact_match` or `within_range`.
- `upgrade-required`: The incoming schema is older than the consumer's minimum
  supported version (`incoming < min_version`).
  Reason: `upgrade_required`.
- `downgrade-required`: The incoming schema is newer than the consumer's maximum
  supported version (`incoming > max_version`).
  Reason: `downgrade_required`.
- `unsupported`: The version or kind cannot be reconciled.
  Reasons: `unknown_artifact_kind`, `malformed_version`, `prerelease_unsupported`,
  or `invalid_range`.

## Usage

Check compatibility with a declared `SchemaRange`:

```python
from openmed.agent import (
    ArtifactKind,
    CompatibilityOutcome,
    SchemaRange,
    check_schema_compatibility,
)

range_policy = SchemaRange(min_version="1.0.0", max_version="2.1.0")

result = check_schema_compatibility(
    artifact_kind=ArtifactKind.EVIDENCE,
    incoming_version="1.4.2",
    supported_range=range_policy,
)

if result.is_compatible:
    print("Schema is compatible:", result.reason_code)
elif result.outcome == CompatibilityOutcome.UPGRADE_REQUIRED:
    print("Artifact migration required before use")
```

For multi-artifact agent workflows, use `SchemaCompatibilityMatrix`:

```python
from openmed.agent import ArtifactKind, SchemaCompatibilityMatrix, SchemaRange

matrix = SchemaCompatibilityMatrix(
    {
        ArtifactKind.EVIDENCE: SchemaRange("1.0.0", "2.0.0"),
        ArtifactKind.PREVIEW: SchemaRange("1.0.0", "1.5.0"),
        ArtifactKind.FHIR: SchemaRange("1.0.0", "1.2.0"),
    }
)

result = matrix.check(ArtifactKind.EVIDENCE, "1.2.0")
payload = result.to_json()
```

## Deterministic serialization

`CompatibilityResult.to_dict()` and `to_json()` serialize deterministically using
only categorical artifact kind and semantic version strings:

```json
{
  "artifact_kind": "evidence",
  "incoming_version": "1.4.2",
  "max_version": "2.1.0",
  "min_version": "1.0.0",
  "outcome": "compatible",
  "reason_code": "within_range"
}
```

`to_json()` generates compact JSON with sorted keys for byte-for-byte identical
output across runs and platforms.

## Privacy and content-free boundary

Compatibility checks never inspect, read, or infer schema characteristics from
artifact content or payload files. Unknown mandatory schema kinds and malformed
version identifiers fail closed without echoing unvalidated values.
