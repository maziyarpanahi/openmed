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

The suites intentionally avoid gated corpus rows and controlled-vocabulary
payloads. Future grounding/context work can extend negation-trap rows with
condition assertions while keeping the existing leakage fields stable.
