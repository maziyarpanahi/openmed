# Language-route health matrix

`openmed.clinical.language_health` provides an offline diagnostic matrix for
language support claims. It joins the local language-pack registry, the
committed model manifest, local i18n fixtures, and bundled policy profiles so
missing or contradictory entries are visible in one reviewable result.

```python
from openmed.clinical.language_health import build_language_health_matrix

matrix = build_language_health_matrix()
for row in matrix["languages"]:
    print(row["language"], row["status"], row["issues"])
```

The default inputs are local only:

- language-pack routes from the process-local registry;
- model declarations from `models.jsonl`;
- JSON and JSONL fixtures under `tests/fixtures/i18n/` and
  `openmed/eval/golden/fixtures/i18n/`;
- policy profiles packaged under `openmed/core/policies/`.

The report is deterministic and JSON serializable. Rows are sorted by primary
language code, and findings are sorted by language and component. A component
can be `filled`, `missing`, `contradictory`, `fallback`, `limited`,
`user_supplied`, `not_applicable`, or `unverified`. The row status is
`healthy`, `degraded`, `missing`, or `contradictory`.

Fixture evidence is metadata-only: the matrix records file names, record
counts, language codes, and synthetic-safety status, but never fixture text,
predictions, spans, or expected outputs. A fixture is `filled` only when its
metadata certifies every observed record as synthetic. This is an evidence
index, not a clinical validation or compliance certification.

Named model fallbacks and user-supplied model routes remain explicit in the
matrix. They are not silently promoted to trained model coverage. The
`check_language_health()` helper returns the number of findings, while
`require_language_health()` raises `LanguageHealthError` when review is
required.
