# Language coverage diffs

`openmed.core.language_coverage_diff` compares declared coverage metadata offline;
it does not run model evaluation or decide whether a locale is clinically fit.

Use `diff_language_coverage(before, after)` with versioned manifests:

```python
from openmed.core.language_coverage_diff import diff_language_coverage

before = {"schema_version": "1.0.0", "packs": []}
after = {"schema_version": "1.0.0", "packs": [{
    "locale": "en-US", "scripts": ["Latin"],
    "identifier_classes": ["PERSON"], "surrogate_support": "approximated",
    "evidence_digest": None,
}]}
assert diff_language_coverage(before, after)["added"] == ["en-US"]
```

Locales canonicalize language case, script case and region case, including
underscore aliases. Duplicate canonical locales reject. Capabilities are unique
controlled tokens; surrogate states are `missing`, `approximated`, `supported`.
An evidence digest is absent (`None`) or 64 lowercase hexadecimal characters.
Manifests use schema version `1.0.0`, with at most 1,000 packs and 256 entries in
each capability list. Unknown metadata is not copied to output.

Added/removed locales are ordered. Shared locales may improve and regress
simultaneously: gaining one script cannot conceal losing an identifier class.
Replacing an evidence digest is a change, never inferred scientific improvement.
Adding/removing evidence reports evidence availability only. Use
`render_language_coverage_diff` for deterministic JSON. Outputs contain only
locale codes and controlled change categories; no examples or model paths.
