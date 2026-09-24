# SDOH Evidence Deduplication

`deduplicate_sdoh_evidence()` prevents copied-forward social-history text from
inflating an evidence count. It clusters observations by category, status,
temporality, and normalized protected text while retaining every document-local
source reference.

```python
from openmed.clinical.sdoh_deduplicate import (
    SDOHEvidenceObservation,
    SDOHSourceReference,
    deduplicate_sdoh_evidence,
)

observations = [
    SDOHEvidenceObservation(
        observation_id="observation-1",
        category="housing",
        status="current",
        temporality="recent",
        source=SDOHSourceReference(
            source_id="document-local-1",
            version_id="v1",
            start=12,
            end=28,
        ),
        protected_text="synthetic housing concern",
    )
]

result = deduplicate_sdoh_evidence(observations)
assert result.independent_evidence_count == 1
```

Normalization applies Unicode NFKC, case folding, punctuation separation, and
whitespace collapse. Category, status, and temporality remain part of the
cluster identity, so conflicting observations are not collapsed. A cluster is
classified as `unique`, `exact`, or `normalized`.

Protected text is excluded from object representations and output. Reports
contain only controlled fields, one-way SHA-256 fingerprints, opaque source
references, and half-open offsets. Use document-local or otherwise
non-sensitive reference IDs; never pass patient identifiers as `source_id`.

The independent evidence count is the number of clusters, not the number of
source mentions. It is review metadata, not a confidence score or clinical
decision. The implementation is deterministic, standard-library-only, and
performs no network calls.
