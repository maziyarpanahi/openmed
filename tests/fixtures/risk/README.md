# Risk Regression Fixture Format

The JSONL files in this directory are synthetic-only release-gate fixtures.
Each file begins with one `meta` row:

```json
{"kind":"meta","version":1,"suite":"negation_traps"}
```

Case rows for `negation_traps.jsonl` contain:

- `kind`: `case`
- `case_id`: stable fixture identifier
- `synthetic`: must be `true`
- `text`: synthetic clinical note text
- `spans`: detector spans represented as `{ "label": "...", "text": "..." }`
- `expected`: committed risk expectations

Record rows for `quasi_identifier_uniqueness.jsonl` contain:

- `kind`: `record`
- `record_id`: stable fixture identifier
- quasi-identifier fields consumed by `openmed.risk.risk_report`

Note rows for `text_quasi_identifiers_v1.jsonl` contain:

- `kind`: `note`
- `note_id`: stable fixture identifier
- `synthetic`: must be `true`
- `text`: synthetic free-text note. Span offsets are Python `str` indexes,
  so `text[start:end]` is the span surface.
- `spans`: one object per quasi-identifier, with `start`, `end`, `label`,
  `text`, and `generalized`
- `expected`: the generalized tabular row those spans should produce

The meta row is marked `"synthetic": true` and names the generalization
rules. Each note has one span for `AGE`, `DATE`, `ZIPCODE`, `LOCATION`, and
`OCCUPATION`, in that order. `label` is the canonical name from
`openmed.core.labels` and must pass `is_recognized_label`. `expected` uses
`age`, `date`, `zipcode`, `location`, and `occupation`, and each field equals
that span's `generalized` value.

- `AGE` (`ten_year_band_or_90_plus`): span text is an integer age. Ages below
  90 become a ten-year band such as `70-79`. Ages 90 and above become `90+`.
- `DATE` (`year`): span text is `YYYY-MM-DD`. The generalized value is the
  four-digit year.
- `ZIPCODE` (`zip3_star_star`): span text is five digits. The generalized
  value keeps the first three digits and appends `**`. Prefixes listed in
  meta `small_zip3` (committed value: `000`) become `000**`.
- `LOCATION` (`region_after_comma`): span text is `Place, REGION`. The
  generalized value is the trimmed token after the comma.
- `OCCUPATION` (`broad_category`): span text is a fictional job title. The
  generalized value is the matching `occupation_categories` entry on the meta
  row, compared with `str.casefold`.

Loading this fixture records those rows for a future note-to-table bridge.

The suites intentionally avoid gated corpus rows and controlled-vocabulary
payloads. Future grounding/context work can extend negation-trap rows with
condition assertions while keeping the existing leakage fields stable.
