# OCR document-routing evaluation

`openmed.eval.ocr_routing` provides a deterministic, offline evaluation harness
for routing clinical documents after OCR. It exercises the local document
classifier and the existing routing profiles with synthetic examples for
radiology, pathology, progress, discharge, operative, consult, and unknown
documents.

Radiology, pathology, and discharge summaries expect their specialized local
profiles. Other document families exercise the generic pass-through fallback.

The harness is an evaluation of the routing pipeline, not a compliance
certification or a clinical decision guarantee. It does not load a model, make
a network request, or call an external OCR service.

## Run the default corpus

```python
from openmed.eval.ocr_routing import assert_ocr_routing_gate

report = assert_ocr_routing_gate()
print(report.metrics.to_dict())
```

`run_ocr_routing_eval()` returns a report without raising when a case fails.
`assert_ocr_routing_gate()` raises an `AssertionError` whose diagnostic names
only fixture IDs, failure categories, and safe structural details.

## What is scored

The report includes three aggregate surfaces:

- `route_selection_accuracy` compares the predicted document type with the
  fixture's expected family.
- `offset_projection_accuracy` compares section labels and canonical
  half-open offsets after projecting detector output from OCR text back to the
  canonical coordinate space. Precision, recall, and F1 are also reported.
- `safe_fallback_rate` checks that unknown, unsupported, or low-confidence
  classifications select the generic pass-through profile and preserve all
  offset-bearing sections and probe entities.

OCR-to-canonical alignment uses Python's standard-library
`difflib.SequenceMatcher`. Equal runs retain exact boundaries; insertions,
deletions, and replacements are mapped monotonically. Callers can build the
map directly with `build_offset_projection(source_text, target_text)` and
project a range with `projection.project_span(start, end)`.

## Privacy and fixture policy

The default corpus is synthetic and lives in memory. Fixture manifests and
reports contain lengths, labels, offsets, counts, confidence values, fixture
IDs, and domain-separated SHA-256 digests. They do not serialize canonical or
OCR text. Custom fixtures should use synthetic offline values and should not
place source text in logs, exception messages, committed golden data, or
evaluation artifacts.

The routing result is an engineering signal. Downstream clinical review,
privacy controls, and application-specific safety gates remain required.

## Limits and interpretation

This is a fixture regression gate, not model-accuracy evidence. Unless supplied
explicitly, gold section boundaries are produced by the same local detector on
canonical text. A complete mismatch scores zero F1; an empty/empty comparison
scores one by convention. Every case must pass, even when aggregate thresholds
are relaxed.

Each text is limited to 4,096 characters and each alignment to 4,000,000
source/target character pairs. A run accepts at most 512 fixtures and each
section collection at most 4,096 entries. Oversized inputs fail closed before
alignment. These limits bound this small offline harness, not the production
document pipeline.

Classifier and detector errors use fixed categories. Unknown document types
map to `unknown`; unknown section labels are domain-separated hashes. Callback
errors do not trigger a second detector invocation. Invalid confidence values
fall back conservatively to zero.

Fixture IDs and language metadata are caller-owned identifiers: use
non-sensitive values. Excluding document text is not anonymization of those
identifiers or proof that a caller-supplied fixture is synthetic.
