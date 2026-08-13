# Evidence-bundle integrity

OpenMed can verify a local, counts-only evidence bundle before review with
`openmed.risk.check_evidence_bundle`. Verification is deterministic and does
not make network calls. It reads evidence bytes only to calculate SHA-256
digests; the returned report contains counts and stable failure categories,
never file paths, manifest values, or evidence contents.

## Bundle contract

A bundle is a directory containing `manifest.json` and the files referenced by
its `files` entries. Paths must be relative POSIX paths inside the bundle, and
each entry names a section and a canonical `sha256:<64 lowercase hex>` digest.
The default required sections are `summary`, `metrics`, and `provenance`.
Manifests may also include `manifest_hash`, a canonical SHA-256 digest over the
manifest with that field omitted.

```json
{
  "schema_version": "openmed.evidence_bundle.v1",
  "policy_fingerprint": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "required_sections": ["summary", "metrics", "provenance"],
  "provenance": {
    "source_fingerprint": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "generator": "openmed-evaluator",
    "created_at": "2026-08-08T12:00:00Z"
  },
  "files": [
    {
      "path": "evidence/metrics.json",
      "section": "metrics",
      "sha256": "sha256:<digest>"
    }
  ]
}
```

The policy fingerprint is a caller-verifiable binding. Pass the expected
fingerprint when the review context has one:

```python
from openmed.risk import check_evidence_bundle

result = check_evidence_bundle(
    "./evidence-bundle",
    expected_policy_fingerprint="sha256:<policy-digest>",
)
if not result.passed:
    print(result.to_dict())
```

Provenance is complete only when it contains a source fingerprint, generator
identifier, and timezone-qualified creation timestamp. A provenance entry may
also repeat `policy_fingerprint`; if present, it must match the manifest's
fingerprint.

## Failure categories

Reports contain only these stable category names and aggregate counts:

| Category | Meaning |
| --- | --- |
| `manifest_unreadable` | The manifest or bundle root could not be read locally. |
| `invalid_manifest` | The manifest shape, digest, or path contract is invalid. |
| `schema_mismatch` | The manifest schema is not supported by this verifier. |
| `policy_mismatch` | The policy fingerprint is missing, malformed, or unexpected. |
| `missing_section` | A required section has no manifest entry. |
| `incomplete_provenance` | Required safe provenance metadata is absent or malformed. |
| `missing_file` | A manifest-referenced evidence file is absent. |
| `hash_mismatch` | A file or optional manifest hash differs from its content. |
| `unsafe_path` | A file reference escapes the bundle or uses a symlink. |
| `unreadable_file` | A referenced path is not a readable regular file. |

The checker is an integrity gate, not a compliance certification or clinical
decision guarantee. Evidence must remain aggregate and synthetic in committed
fixtures; raw identifiers and clinical text do not belong in manifests,
reports, logs, or documentation.
