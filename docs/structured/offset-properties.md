# Cross-format offset properties

OpenMed adapters normalize PDF, OCR, RTF, ODT, and presentation content into a
text stream with half-open character offsets. The shared property harness in
`openmed.structured.offset_properties` checks the part of that contract that is
independent of a parser:

- offsets use Python code-point indexes and satisfy `0 <= start <= end <= len(text)`;
- source spans are ordered and non-overlapping;
- detector spans project to every overlapping source span;
- zero-width detector spans are valid and do not create a fake source range; and
- failures are reported by a stable category rather than a parser message.

The harness is dependency-free and never opens a path or makes a network call.
It accepts any adapter result exposing `text` and `spans`, so format-specific
parsers can be tested with the same invariant suite without bundling parser
dependencies.

## Running the shared suite

```python
from openmed.structured.offset_properties import (
    build_synthetic_offset_cases,
    run_offset_property_suite,
)

reports = run_offset_property_suite(
    {
        "pdf": pdf_adapter,
        "ocr": ocr_adapter,
        "rtf": rtf_adapter,
        "odt": odt_adapter,
        "presentation": presentation_adapter,
    },
    cases=build_synthetic_offset_cases(),
)
```

Each adapter callable receives a `SyntheticOffsetCase` and returns an object
with `text` and `spans`. The committed test suite uses in-memory adapter-shaped
results only; real parser tests remain responsible for their own optional
dependencies and source fixtures.

`OffsetProjectionReport.to_dict()` contains counts, offsets, and a SHA-256 text
fingerprint. It intentionally does not include the normalized text or any
surface value from a span, making it suitable for deterministic local evidence.

The property suite is structural support tooling. Passing it is not a parser
accuracy guarantee, compliance certification, or clinical decision guarantee.
