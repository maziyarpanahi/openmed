# Mixed-script span integrity

OpenMed includes a deterministic, offline regression suite for Unicode span
boundaries that mix Latin, Indic, CJK, right-to-left, and emoji text. The suite
is an engineering check for offset and replacement consistency; it is not a
compliance certification or a clinical decision guarantee.

## What the suite checks

The built-in fixtures are synthetic and cover:

- decomposed Latin combining marks;
- Devanagari virama conjuncts;
- Han and Latin script transitions;
- Arabic bidirectional text next to ASCII identifiers; and
- emoji modifier and zero-width-joiner graphemes next to Latin text.

Every span uses half-open Python Unicode code-point offsets. The evaluator
requires each boundary to coincide with an extended grapheme boundary, requires
the source slice to match its expected digest, preserves the declared span
order, rejects overlap, and verifies that repeated entity keys keep one stable
surrogate across the document and repeated evaluation runs.

## Privacy-safe reports

Fixture source text and surrogate values stay in memory. Serialized reports
contain only fixture identifiers, coverage tags, offsets, labels, counts,
booleans, stable domain-separated hashes, and bounded failure codes. Runner
exceptions are converted to `runner-error` without retaining exception text, so
a failing integration cannot copy a source value into CI output.

```python
from openmed.eval.mixed_script_integrity import (
    evaluate_mixed_script_integrity,
)

report = evaluate_mixed_script_integrity(iterations=5)
assert report.passed
payload = report.to_dict()
```

Callers can pass a `runner(fixture)` function to evaluate another span producer.
The runner must return `MixedScriptSpan` values in source order. No model,
network service, telemetry client, or restricted dataset is used by the default
suite.

## Validation

Run the focused regression test with:

```bash
.venv/bin/python -m pytest tests/unit/eval/test_mixed_script_integrity.py -q
```
