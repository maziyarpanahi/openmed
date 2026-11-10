# Document-type calibration reports

`openmed.eval.doctype_calibration` turns synthetic document-type labels and
classifier predictions into deterministic, counts-only calibration evidence.
It complements top-one accuracy with confidence reliability, abstention
behavior, and support for each document type.

The report contains:

- fixed-width confidence bins with support, correct counts, mean confidence,
  accuracy, and absolute calibration gap;
- expected calibration error (ECE) over those bins;
- abstained and retained counts/rates at each requested confidence threshold;
- gold support, prediction count, and correct count for every canonical
  document type; and
- stable SHA-256 fingerprints for the model descriptor and scored fixture set.

`unknown` predictions always count as abstentions. At a requested threshold,
any other prediction with confidence below the threshold also abstains. A
prediction exactly on the threshold is retained.

## Build a report

```python
from openmed.eval.doctype_calibration import (
    build_doctype_calibration_report,
)

samples = [
    {
        "expected_type": "progress_note",
        "predicted_type": "progress_note",
        "confidence": 0.91,
    },
    {
        "expected_type": "radiology_report",
        "predicted_type": "unknown",
        "confidence": 0.0,
    },
]

report = build_doctype_calibration_report(
    samples,
    model={"family": "doctype", "revision": "synthetic-v1"},
    num_bins=10,
    abstention_thresholds=(0.5, 0.7, 0.9),
)

json_text = report.to_json()
markdown_text = report.to_markdown()
report.write_json("artifacts/doctype-calibration.json")
report.write_markdown("artifacts/doctype-calibration.md")
```

Input order does not affect the fixture fingerprint. Model descriptor key
order does not affect the model fingerprint. A precomputed fingerprint in the
form `sha256:<64 lowercase hexadecimal characters>` can be passed as `model`
when model metadata is fingerprinted elsewhere.

## Privacy and reproducibility boundary

Each plain-dictionary sample is projected onto exactly `expected_type`,
`predicted_type`, and `confidence`. Extra fields—including note text, fixture
identifiers, paths, and arbitrary metadata—are ignored and are neither
rendered nor included in exceptions. Expected labels must be one of the
classifier's canonical document types; predictions may additionally be
`unknown`. This prevents arbitrary source-derived strings from becoming report
categories.

Model metadata is bounded, normalized, and hashed in memory. The descriptor
itself is not copied into JSON or Markdown. Fixture fingerprints bind the
normalized scored records, including duplicate records, after a stable sort.
Reports contain canonical type names and aggregate numbers only; they never
contain individual sample records.

The implementation performs no network access, loads no model, and bundles no
dataset. Callers provide synthetic labels and local predictions. The report is
evaluation evidence only: it is not a compliance certification or a clinical
decision guarantee.
