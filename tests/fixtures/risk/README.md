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

Record rows for `successive_releases_v1.jsonl` contain:

- `kind`: `record`
- `record_id`: stable fixture identifier
- `synthetic`: must be `true`
- `release`: `release_1` or `release_2`
- `privacy_unit_id`: opaque key naming the same synthetic unit in both
  releases; it is not a quasi-identifier
- `age_band` and `area`: the quasi-identifiers, generalized differently in
  each release (`release_1` uses ten-year age bands and a town, `release_2`
  uses five-year age bands and a district)

The meta row names `quasi_identifiers` and `privacy_unit_column`, and commits
the expectations under `expected`. Each release has 24 rows. Running
`openmed.risk.kanon_report` on one release alone gives `k` 3 for `release_1`
and `k` 2 for `release_2`.

Expected intersection result for later composition tests: joining the two
releases on `privacy_unit_id` and combining both releases' quasi-identifiers
gives `k` 1. Units `pu-03`, `pu-04` and `pu-21` each end up alone in their
equivalence class, even though neither release isolates them by itself.
`successive_releases_v1.jsonl` does not score composition; it only fixes the
known example.

The suites intentionally avoid gated corpus rows and controlled-vocabulary
payloads. Future grounding/context work can extend negation-trap rows with
condition assertions while keeping the existing leakage fields stable.
