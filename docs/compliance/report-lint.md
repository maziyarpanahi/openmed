# Privacy report field linting

`openmed.compliance.report_lint` checks aggregate audit and release reports
against a typed, allowlisted schema before they are serialized or shared. It
is a deterministic local check: it performs no network call and does not copy
the input report into its result.

The linter accepts these representations:

- `count`: a non-negative integer, optionally bounded by `maximum`.
- `hash`: a canonical lowercase `sha256:<64 hexadecimal characters>` digest.
- `code`: a bounded safe code, optionally restricted to `allowed_values`.
- `boolean`, `number`, and `ratio`: finite scalar values with optional bounds.
- `object` and `array`: nested typed fields with bounded array length.

`text` and `forbidden` are explicit rejection rules. Do not use free-form
strings for report fields. Store sensitive material outside the report and
bind it with a hash, or emit an aggregate count instead.

## Example

```python
from openmed.compliance.report_lint import ReportFieldSpec, lint_report

schema = {
    "report_hash": ReportFieldSpec("hash", required=True),
    "record_count": ReportFieldSpec("count", maximum=100_000),
    "status": ReportFieldSpec(
        "code",
        allowed_values=("failed", "passed"),
        required=True,
    ),
}

result = lint_report(
    {
        "report_hash": "sha256:" + ("0" * 64),
        "record_count": 12,
        "status": "passed",
    },
    schema,
)

if not result.valid:
    safe_diagnostics = result.to_dict()
```

`ReportLintResult` contains only the validity flag, aggregate counts, stable
reason codes, safe schema paths, coarse value shapes, and digests for unknown
field names. It never includes rejected values. `lint_report(..., strict=True)`
raises `ReportLintError` with the same value-free summary.

The linter is a reporting boundary check, not a compliance certification or a
clinical decision. Validate the surrounding release workflow and review
residual risk locally before sharing any report.
