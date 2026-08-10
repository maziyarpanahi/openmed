# Export filename policy

OpenMed export filenames are operational metadata. They must identify the
artifact kind, serialization format, schema version, and a short provenance
fingerprint without copying source identifiers into a filesystem path.

The local policy lives in `openmed.core.export_naming` and has no model,
filesystem, clock, or network dependency. A typical export name is:

```text
audit-report-json-schema-v1-aaaaaaaaaaaa.json
```

The components are, in order:

1. a general artifact type;
2. the output format;
3. the literal `schema` marker and schema version;
4. the first 12 hexadecimal characters of a SHA-256 provenance fingerprint;
5. the format extension.

Use `ExportArtifactMetadata` with `build_export_filename`:

```python
from openmed.core.export_naming import (
    ExportArtifactMetadata,
    build_export_filename,
    fingerprint_for,
)

metadata = ExportArtifactMetadata(
    artifact_type="audit-report",
    format="json",
    schema_version="v1",
    fingerprint=fingerprint_for({"schema": "v1", "format": "json"}),
)
filename = build_export_filename(metadata)
```

If provenance is not already a digest, `fingerprint_for` hashes it in memory
using canonical JSON. The source value is not retained by the metadata object,
returned in a report, or included in error messages. Callers should use
synthetic or already-deidentified provenance when building that input.

The policy rejects POSIX and Windows path separators, control characters,
traversal components, unsupported filename characters, and values that look
like direct identifiers. It also rejects arbitrary metadata fields such as
`subject_id`; only the typed fields above are accepted. A fingerprint must be
a short or full hexadecimal SHA-256 digest, so a raw identifier cannot be
silently placed in the fingerprint slot.

Names do not contain a generated timestamp. An explicit ISO date or datetime
may be passed as `explicit_timestamp` when a caller has a documented need for
one. The value is normalized to a punctuation-free UTC-aware token, and the
same explicit value always produces the same name.

This is a filename safety and reproducibility policy. It is not a compliance
certification, a de-identification guarantee, or a clinical decision rule.
