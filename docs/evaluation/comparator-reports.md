# Comparator reports

`openmed.eval.comparator_report` turns a comparator matrix into a reviewable,
counts-only artifact. The renderer accepts a `ComparatorMatrixReport` or a
plain JSON-ready dictionary and writes deterministic JSON or Markdown without copying
fixture text, fixture identifiers, arbitrary metadata, nested benchmark
reports, or exception messages.

```python
from openmed.eval import (
    build_comparator_report,
    render_comparator_report_json,
    render_comparator_report_markdown,
)

report = build_comparator_report(
    matrix,
    environment={
        "python": "3.12",
        "platform": "local",
        "runner_version": "synthetic-v1",
    },
)

json_text = render_comparator_report_json(report)
markdown_text = render_comparator_report_markdown(report)
```

Each system row contains only the fixed comparator metrics, their available
numeric count fields, fixture count, status, and a failure count. The report
also includes metric definitions, a SHA-256 environment fingerprint, and
failure totals grouped into high-level categories such as `dependency`,
`execution`, and `not_available`. Failure messages are never rendered.

Built-in public comparator, model, and device names remain readable. Custom
suite, model, device, and system identifiers are replaced with deterministic
SHA-256 identifiers so a short name-like source value cannot cross the report
boundary. Sanitized report mappings are deeply immutable after construction,
and serialization revalidates the artifact type, schema version, row limit,
numeric bounds, and JSON indentation. Large declared failure counts are added
arithmetically instead of allocating one in-memory event per failure.

The environment mapping is used only as hash input. For a reproducible artifact
across machines, pass an explicit environment mapping or a precomputed
`sha256:<64 lowercase hex characters>` fingerprint. The renderer performs no
network access and does not load model or fixture content.

The report is evaluation evidence only; it is not a compliance certification or
a clinical decision guarantee.
