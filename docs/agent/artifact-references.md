# Content-free artifact references

`openmed.agent.ArtifactReference` lets an agent run point to a produced artifact
without copying reports, clinical content, filenames, local paths, or remote
URLs into its trace.

## Reference contract

Each version 1 reference contains only:

- an opaque `art_` identifier followed by 32 lowercase hexadecimal characters;
- one closed kind: `evidence`, `preview`, `fhir`, `omop`, or `evaluation`;
- a lowercase, versioned schema identifier such as
  `openmed.agent.evidence.v1`;
- a 64-character lowercase SHA-256 digest; and
- a positive byte size no larger than a signed 64-bit integer.

```python
from openmed.agent import ArtifactReference

reference = ArtifactReference.from_dict(
    {
        "artifact_id": "art_" + "1" * 32,
        "kind": "evidence",
        "schema_id": "openmed.agent.evidence.v1",
        "sha256": "a" * 64,
        "byte_size": 128,
    }
)

payload = reference.to_json()
```

JSON parsing rejects duplicate or unknown fields. Use
`validate_artifact_references()` when attaching several references to reject a
repeated artifact identifier while preserving their order.

## Privacy and I/O boundary

Generate artifact identifiers from random bytes; never derive them from patient
data, prompts, filenames, operator identity, or tool output. Validation errors
contain only a public field name and a stable error code, never the rejected
value.

The reference records a caller-supplied digest and size but does not verify
either one. Creating, parsing, and serializing references never opens local
files, fetches remote resources, stores artifacts, or validates their clinical
content.
