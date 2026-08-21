# CLI result envelope

OpenMed CLI integrations can use `openmed.cli.result_envelope` when a command
needs a machine-readable result. The contract is versioned, deterministic, and
safe to write to logs or reports: it has no free-text payload or exception
message field.

## Wire shape

Every result contains the same six top-level fields:

```json
{
  "artifacts": [],
  "category": "validation",
  "counters": {"processed": 4},
  "remediation_codes": ["check_input"],
  "schema_version": 1,
  "status": "failure"
}
```

`status` is either `success` or `failure`. A successful result must use the
`success` category and has no remediation codes. Failed results use one of the
`input`, `validation`, `configuration`, `runtime`, or `integrity` categories.
The category is a code, not a human-written explanation.

`counters` is an object of at most 128 entries whose keys are lowercase logical
identifiers and whose values are non-negative integers. Keys are sorted during
serialization. An envelope can include at most 64 artifact fingerprints. The
envelope does not accept arbitrary `data`, `message`, `details`, or other
free-text fields.

`remediation_codes` is a sorted, de-duplicated list of at most three values
from this finite set:

| Code | Meaning |
| --- | --- |
| `check_input` | Validate the local input contract. |
| `check_configuration` | Review local configuration or options. |
| `verify_artifact` | Re-check a local artifact fingerprint. |
| `retry_command` | Retry the local command. |
| `contact_operator` | Escalate to the responsible operator. |

## Artifact fingerprints

Artifacts contain only a logical name, a lowercase SHA-256 digest, and a byte
count:

```json
{
  "name": "report",
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "size_bytes": 128
}
```

Use `ArtifactFingerprint.from_bytes()` or `.from_file()` to calculate a
fingerprint. File paths and artifact bytes are never included in the envelope;
the name must be a lowercase logical identifier rather than a path.

## Canonical serialization

```python
from openmed.cli.result_envelope import (
    ArtifactFingerprint,
    ResultCategory,
    RemediationCode,
    create_failure_envelope,
)

result = create_failure_envelope(
    ResultCategory.VALIDATION,
    counters={"processed": 4},
    artifacts=[ArtifactFingerprint.from_bytes("report", b"local bytes")],
    remediation_codes=[RemediationCode.CHECK_INPUT],
)
print(result.to_json())
```

`to_json()` uses compact UTF-8-safe JSON with sorted keys and no locale,
terminal-width, timestamp, host, or path fields. `write_json(stream)` adds one
newline for a CLI stream. The module uses only the standard library and does
not make network calls; `.from_file()` reads only the caller-provided local
file. `from_json()` rejects duplicate keys, non-standard numeric constants, and
documents larger than 1,048,576 characters.
