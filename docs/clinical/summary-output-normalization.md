# Deterministic summary-output normalization

Local summary models can express the same claims with different Markdown
spacing, list markers, heading forms, or inline citation placement. The
standard-library normalizer makes those presentation details reproducible
without rewriting claim words or contacting a network service.

```python
from openmed.clinical import normalize_summary_output

result = normalize_summary_output(
    "##  Assessment ##\n* Finding one[ 1 ]\n+ Finding two [2]"
)

assert result.normalized_text == (
    "## Assessment\n- Finding one [1]\n- Finding two [2]"
)
assert result.operation_codes == (
    "whitespace",
    "heading",
    "list_marker",
    "citation_placement",
)
```

## Canonical format

The normalizer applies only formatting transformations:

- LF line endings, single horizontal spaces, no leading or trailing blank
  lines, and at most one blank line between blocks.
- ATX headings with one space after the marker. Setext headings are converted
  to `#` or `##` according to their original underline.
- Unordered list markers become `-`. Ordered list markers become sequential
  `1.`, `2.`, and so on for each contiguous list at a given indentation.
- Numeric citations such as `[1]`, `[1, 2]`, and `【3】`, plus explicit tokens
  such as `[citation:4]`, are canonicalized to ASCII brackets and placed at
  the end of their logical line. Markdown link labels are not citation tokens.

The implementation never performs case folding, Unicode compatibility
decomposition, sentence rewriting, claim sorting, or model inference. It
returns the normalized text in `result.normalized_text`; a caller that only
needs the text can use `normalize_summary_text()`.

## Value-free audit record

`result.to_dict()` and `result.to_json()` intentionally do not include the
normalized text. They contain only the schema version and fixed operation
codes, so they can be retained in an audit or regression artifact without
copying summary content:

```json
{"operation_codes":["whitespace","list_marker"],"schema_version":1}
```

The `operations` field is immutable and restricted to the published
`SUMMARY_OUTPUT_NORMALIZATION_OPERATION_CODES` tuple. Validation errors use
fixed messages and do not echo rejected values. The normalizer emits no logs,
does not read environment or clock state, and makes no mandatory network
call. It is a formatting aid for human-reviewed clinical workflows, not a
clinical decision or compliance guarantee.
