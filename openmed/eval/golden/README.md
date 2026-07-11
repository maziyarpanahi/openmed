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
  Synthetic hard-negative fixtures use an empty list because the confusable
  tokens are non-PHI and must remain unredacted.
- `metadata.category`: one of `nested_overlapping`, `chunk_boundary`,
  `multilingual`, `checksum_ids`, `financial_ids`, `date_arithmetic`,
  `policy_profile_actions`, or `hard_negatives`.
- `metadata.expected_output`: expected post-action output, including `method`
  and resulting `text`.
- `metadata.synthetic`: must be `true`.
- `metadata.hard_negative_candidates`: required only for `hard_negatives`;
  each candidate records canonical label, offsets, synthetic marker, and
  aggregate difficulty scores.

The package loader validates offsets, canonical labels, synthetic markers,
expected output, hard-negative candidate metadata, and language coverage. The
JSON and JSONL files are also compatible with
`openmed.eval.harness.load_fixtures`; golden-specific expected output remains
available through each fixture's metadata.

## Relation Gold Fixtures

`fixtures/relation_gold.jsonl` contains synthetic-only relation extraction
fixtures for `openmed.eval.suites.relations`. Each JSONL row uses schema
version `1` and this shape:

```json
{
  "id": "relation-sentence-treatment",
  "schema_version": 1,
  "language": "en",
  "text": "Aspirin treats fever in note one.",
  "entities": [
    {
      "id": "e-medication",
      "start": 0,
      "end": 7,
      "label": "MEDICATION",
      "text": "Aspirin"
    }
  ],
  "relations": [
    {
      "id": "rel-aspirin-fever",
      "type": "treats",
      "head": "e-medication",
      "tail": "e-fever",
      "scope": "sentence"
    }
  ],
  "traps": [
    {
      "id": "trap-assertion-negation",
      "kind": "assertion",
      "relation_ids": ["rel-pneumonia-denied"],
      "zero_tolerance": true
    }
  ],
  "metadata": {
    "synthetic": true,
    "category": "relation_gold",
    "schema_version": 1
  }
}
```

Required fields:

- `entities`: canonical-label spans with unique ids and character offsets into
  `text`. Relation arguments reference these ids.
- `relations`: directed relation records with unique ids, `type`, `head`,
  `tail`, and a `scope` of `sentence` or `document`.
- `traps`: optional zero-tolerance assertion or temporal traps that reference
  relation ids and are carried into the score payload for release-gate wiring.
- `metadata.synthetic`: must be `true`; relation gold must not contain DUA,
  production, or real patient data.

The relation loader validates schema version, unique fixture/entity/relation
ids, argument references, offsets, canonical entity labels, relation scopes,
and trap metadata.
