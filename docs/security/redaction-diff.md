# Policy-aware redaction diffs

`openmed.risk.redaction_diff` compares two aggregate redaction-result
summaries for a privacy review. It reports changes in action counts, category
counts, and explicitly supplied aggregate counts, alongside the policy
fingerprint for each run.

The input is a mapping (or a local JSON object) containing aggregate fields:

```python
from openmed.risk import diff_redaction_summaries, render_redaction_diff

before = {
    "policy": "clinical_minimal_redaction",
    "action_counts": {"keep": 1, "mask": 2},
    "category_counts": {"PERSON": 1, "LOCATION": 2},
    "counts": {"total": 3, "redacted": 2},
}
after = {
    "policy": "strict_no_leak",
    "action_counts": {"mask": 3, "redact": 1},
    "category_counts": {"LOCATION": 1, "PHONE": 3},
    "counts": {"total": 4, "redacted": 4},
}

diff = diff_redaction_summaries(before, after)
print(diff.to_json())
print(render_redaction_diff(diff))
```

Each changed key has its `before` and `after` count, numeric `delta`, and one
of `added`, `removed`, `increased`, or `decreased` classifications. Ordering
is lexical and stable, so the same summaries always produce the same report.

The diff never accepts or emits document text, replacements, span surfaces, or
other per-record values. Canonical OpenMed categories and actions remain
readable; unknown category and metric keys are represented by stable SHA-256
fingerprints. Policy names are resolved from the local bundled profiles when
possible, and no network call is required.

This is an aggregate review aid, not a compliance certification or a clinical
decision guarantee.
