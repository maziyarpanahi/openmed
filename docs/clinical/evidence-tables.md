# Clinical Evidence Tables

Clinical evidence tables provide a compact, deterministic review view without
exporting note text or extracted values. Each row contains only half-open source
offsets, a controlled assertion state, confidence, a review flag, and an
optional SHA-256 digest.

The table is a review aid. It is not a clinical decision, compliance
certification, or guarantee that an extraction is correct.

## Build typed records

```python
from openmed.clinical.evidence_table import (
    AssertionStatus,
    EvidenceRecord,
    EvidenceTable,
)

records = [
    EvidenceRecord.from_extraction(
        source_start=12,
        source_end=29,
        assertion_status=AssertionStatus.AFFIRMED,
        confidence=0.94,
        review_required=False,
    ),
    EvidenceRecord.from_extraction(
        source_start=44,
        source_end=58,
        assertion_status=AssertionStatus.UNCERTAIN,
        confidence=0.61,
        review_required=True,
    ),
]

table = EvidenceTable.from_records(records)
print(table.to_json())
print(table.to_markdown())
```

Records are sorted by source offsets and safe metadata before rendering. JSON
includes deterministic assertion and review counts plus the ordered records.
Markdown uses the same ordering and represents absent value hashes as
`omitted`.

## Protected values are omitted by default

`protected_value` is accepted only by `EvidenceRecord.from_extraction()` and is
never stored on the returned object. Without an explicit opt-in, it is omitted:

```python
record = EvidenceRecord.from_extraction(
    source_start=12,
    source_end=29,
    assertion_status=AssertionStatus.AFFIRMED,
    confidence=0.94,
    review_required=False,
    protected_value="synthetic finding",
)

assert record.value_hash is None
```

Set `include_value_hash=True` when a stable comparison token is needed. The
record then stores only `sha256:<hex>` and discards the raw input after hashing.
SHA-256 does not make a low-entropy value anonymous; do not publish hashes when
dictionary attacks are plausible. Prefer omission unless the review workflow
has a documented need for correlation.

## Validation contract

- Offsets must be non-negative and form a non-empty half-open span.
- Assertion status must be one of `affirmed`, `negated`, `uncertain`,
  `historical`, `hypothetical`, or `unknown`.
- Confidence must be finite and between `0` and `1`.
- `review_required` must be a boolean.
- A supplied hash must match `sha256:` followed by 64 lowercase hex digits.

Validation errors identify the invalid field without including the submitted
value. The module uses only the Python standard library and performs no network
calls.
