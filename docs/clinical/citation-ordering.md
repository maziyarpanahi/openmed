# Deterministic citation ordering

`order_citations()` gives guarded clinical outputs a canonical evidence order
without retaining source text. Citation rows contain only opaque document,
section, and evidence identifiers; half-open source offsets; and a primary
evidence marker.

```python
from openmed.clinical.citation_ordering import Citation, CitationOrdering

artifact = CitationOrdering(
    (
        Citation(
            document_id="doc-02",
            section="assessment",
            source_start=24,
            source_end=31,
            evidence_id="evidence-02",
        ),
        Citation(
            document_id="doc-01",
            section="history",
            source_start=4,
            source_end=13,
            evidence_id="evidence-01",
            primary=True,
        ),
    )
)

payload = artifact.to_json()
```

## Ordering contract

Citations are sorted lexicographically by:

1. `document_id`
2. `section`
3. `source_start`
4. `source_end`
5. `evidence_id`

The primary marker does not change this order. It identifies the primary row
after ordering. A citation collection represents one guarded claim, so two
distinct primary citations are rejected. Duplicate coordinates are also
rejected when their primary markers disagree.

## Privacy boundary

Identifiers must be opaque metadata tokens and source offsets must satisfy
`0 <= source_start < source_end`. Prompts, tool arguments, clinical outputs,
evidence text, bearer values, and filesystem paths are not fields in the
artifact schema. Validation errors are fixed categories and never echo rejected
values.

The implementation is deterministic, uses only the Python standard library,
and performs no network calls. It orders evidence metadata for review; it does
not evaluate clinical claims or choose primary evidence.
