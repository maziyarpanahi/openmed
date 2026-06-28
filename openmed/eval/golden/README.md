# Golden De-Identification Fixtures

This directory contains synthetic-only golden fixtures for the eval suites. The
fixtures contain no DUA data, no production data, and no real PHI.

Each JSON file has a top-level `fixtures` list; JSONL files use one fixture
object per line. Each fixture uses this shape:

```json
{
  "id": "golden-multilingual-en-ssn",
  "language": "en",
  "text": "Synthetic chart lists SSN 123-45-6789 for test patient.",
  "gold_spans": [
    {
      "start": 26,
      "end": 37,
      "label": "SSN",
      "text": "123-45-6789"
    }
  ],
  "metadata": {
    "category": "multilingual",
    "synthetic": true,
    "expected_output": {
      "method": "mask",
      "text": "Synthetic chart lists SSN [SSN] for test patient."
    }
  }
}
```

Required fields:

- `text`: source fixture text.
- `gold_spans`: canonical-label spans with character offsets into `text`.
- `metadata.category`: one of `nested_overlapping`, `chunk_boundary`,
  `multilingual`, `checksum_ids`, `date_arithmetic`, or
  `policy_profile_actions`.
- `metadata.expected_output`: expected post-action output, including `method`
  and resulting `text`.
- `metadata.synthetic`: must be `true`.

The package loader validates offsets, canonical labels, synthetic markers,
expected output, and language coverage. The JSON and JSONL files are also
compatible with `openmed.eval.harness.load_fixtures`; golden-specific expected
output remains
available through each fixture's metadata.
