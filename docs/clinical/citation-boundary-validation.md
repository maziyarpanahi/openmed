# De-identification-aware citation boundary validation

Citation offsets must be interpreted in the exact post-de-identification
document that produced them. `build_deidentification_offset_map()` binds that
document to a SHA-256 digest and records only source and post-redaction
offsets. It does not retain source text, replacement text, or a
re-identification mapping.

```python
from openmed.clinical.citation_boundaries import (
    build_deidentification_offset_map,
    validate_citation_boundaries,
)

offset_map = build_deidentification_offset_map(
    original_text,
    deidentified_text,
    replacements,  # records expose source start/end and redacted_text
    source_version="local-source-v1",
)

report = validate_citation_boundaries(
    citations,
    offset_map,
    raise_on_error=False,
)
if report.valid:
    use_for_human_review(report.validated_citations)
```

## Validation contract

- Citation offsets are half-open `[start, end)` offsets in the
  post-de-identification text. Existing citation records using `source_start`
  and `source_end` are accepted as that post-de-identification coordinate
  space.
- Every citation must carry the map's post-de-identification document digest.
  A different digest is rejected, even when the numeric offsets happen to be
  in bounds.
- A citation may cover an unchanged region or exactly one complete replacement
  interval. A citation that cuts through a replacement, combines a replacement
  with neighbouring text, or crosses a removal boundary is rejected with the
  fixed `citation_crosses_replacement_boundary` reason code.
- A citation's source version must be present in the supplied map set. Pass a
  mapping of version to maps when more than one locally available source
  version must be checked; unavailable versions fail closed.

The validator returns deterministic counts, opaque digests, offsets, and fixed
reason codes. With the default `raise_on_error=True`, the first canonical
rejection raises `CitationBoundaryError`. Use `raise_on_error=False` to retain
all safe rejection records for a review packet. Neither path makes a network
call or logs source values.

The map and report are assistive provenance artifacts. They do not establish
clinical correctness, compliance certification, or an autonomous clinical
decision; qualified human review remains required.
