# Reproducible release provenance

`openmed.compliance.reproducible_release` verifies that a release was built
from the declared source revision, normalized build inputs, dependency lock,
and artifact bytes. It is a local-only evidence check: it does not contact a
registry, source host, attestation service, or other network endpoint.

The verifier is a technical control, not a compliance certification or a
clinical decision guarantee.

## Safe evidence record

Build a record from local inputs or from already computed digests:

```python
from openmed.compliance.reproducible_release import (
    build_release_provenance,
    verify_release_provenance,
)

declared = build_release_provenance(
    source_revision="a" * 40,
    build_inputs={
        "python": "3.10",
        "build_backend": "hatchling",
        "flags": ["-O2"],
    },
    dependency_lock_digest="sha256:" + "b" * 64,
    artifact_hashes={
        "sdist": "sha256:" + "c" * 64,
        "wheel": "sha256:" + "d" * 64,
    },
)

report = verify_release_provenance(declared, built)
if not report:
    report.raise_if_invalid()
```

`built` is created with the same function from the locally built artifact and
its observed inputs. The result of `record.to_dict()` contains only the
following evidence fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Version of the OpenMed record schema. |
| `source_revision` | Lowercase immutable source revision. |
| `build_inputs_digest` | Digest of canonicalized toolchain and build inputs. |
| `dependency_lock_digest` | Digest of the declared dependency lock. |
| `artifact_hashes` | Path-free artifact identifiers and content digests. |
| `provenance_schema` | Schema identifier for the attached provenance claim. |

Raw build-input values are consumed only to derive a digest and are not stored
in the record. Path-like values are represented by a path marker during input
normalization. Artifact identifiers cannot contain path separators. Exceptions
and mismatch reports likewise contain only stable categories, safe identifiers,
and digests; credentials and source payloads are never copied into evidence.

## Normalization and hashing

Build-input mappings are sorted recursively by key. Lists and tuples preserve
order because command order can affect a build. Sets are sorted by canonical
JSON. Finite numeric values, booleans, strings, nulls, bytes, and nested
JSON-like mappings are supported. Bytes are replaced by their content digest;
path-like values are replaced by a marker. The canonical representation uses
sorted keys, compact separators, ASCII escaping, and rejects non-finite
numbers.

Dependency locks and artifacts can be supplied as local bytes or local paths
through `compute_dependency_lock_digest`, `compute_artifact_digest`, and
`compute_artifact_hashes`. These helpers read only the supplied local content
and return SHA-256 digests.

## Stable mismatch categories

`ReleaseVerificationReport.mismatch_categories` always uses this order:

1. `schema`
2. `source_revision`
3. `build_inputs`
4. `dependency_lock`
5. `artifact_hash`
6. `provenance_schema`

The report includes one safe field-level entry for each differing claim. A
missing or extra artifact is reported under `artifact_hash` without exposing a
filesystem path. The report is deterministic for equivalent mappings and can
be serialized with `report.to_dict()` for audit storage.

## Focused validation

Run the focused regression test for this verifier:

```bash
.venv/bin/python -m pytest tests/unit/compliance/test_reproducible_release.py -q
```
