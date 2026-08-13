# Surrogate-map referential-integrity audit

`openmed.risk.surrogate_audit` performs a local, deterministic post-run check
for linked-table surrogate maps. It consumes only keyed or hashed source keys,
their generated surrogates, and optional map metadata. It never resolves a
hash, reads an original identifier, contacts a service, or writes an audit
artifact.

## Input contract

Each map is named and contains bindings with a `key_hash` and `surrogate`.
`cardinality` is the expected number of distinct key hashes. A parent/child
relationship is required when orphan detection is needed:

```python
from openmed.risk import audit_surrogate_maps

report = audit_surrogate_maps(
    {
        "subjects": {
            "entries": [
                {
                    "key_hash": "hmac-sha256:00000001",
                    "surrogate": "[SUBJECT_001]",
                },
                {
                    "key_hash": "hmac-sha256:00000002",
                    "surrogate": "[SUBJECT_002]",
                },
            ],
            "cardinality": 2,
        },
        "visits": {
            "entries": [
                {
                    "key_hash": "hmac-sha256:00000001",
                    "surrogate": "[SUBJECT_001]",
                }
            ],
            "cardinality": 1,
        },
    },
    relationships=[{"parent": "subjects", "child": "visits"}],
)

assert report.passed
```

The values in this example are synthetic placeholders. Production callers
must supply a keyed hash produced by their local privacy boundary; they must
not pass source identifiers to the auditor. Parallel `key_hashes` and
`surrogates` sequences, direct hash-to-surrogate mappings, and row-oriented
metadata are also accepted. `entry_count` and `key_hashes` may be supplied as
additional metadata when a release manifest declares them.

## Counts-only result

`report.to_dict()` is JSON-compatible and contains only aggregate fields. The
stable `failure_categories` mapping always includes these four categories:

| Category | Counted condition |
|---|---|
| `cardinality` | A declared distinct-key or entry count does not match, or an entry is malformed. |
| `collision` | One hash maps to multiple surrogates, or one surrogate is reused for multiple hashes within a map. |
| `orphan` | A child-map hash is absent from its declared parent map. |
| `cross_table_consistency` | A hash shared by related maps resolves to different surrogate values. |

Counts identify affected maps, key groups, or relationships—not the values
that caused a finding. `report.failures` contains only non-zero category/count
pairs and is safe for release-review summaries. When no relationships are
provided, shared hashes are compared across map pairs for consistency, but no
implicit orphan finding is emitted.

## Security boundary

The auditor is an integrity check, not a compliance certification, clinical
decision, or guarantee that a surrogate is unlinkable. Keep reversible maps
and any key material in their existing local custody boundary. Do not log the
input maps, include them in exception messages, or add source identifiers to
fixtures. Persist only the counts-only report when release evidence is
required.
