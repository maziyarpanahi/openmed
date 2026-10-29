# Clinical language-pack readiness matrix

`build_language_readiness_matrix` projects language-pack readiness records into
a bounded JSON document, and `render_language_readiness_json` /
`render_language_readiness_markdown` render that same snapshot as byte-stable
JSON or Markdown. `readiness_entries_from_packs` derives records from the
registered language packs through the existing coherence report.

The matrix is a pure projection: it never loads models, downloads fixtures,
detects a document's language, scores language quality or declares clinical
fitness. Every cell carries a closed capability name, a closed state, and the
`sha256:` evidence digest that justifies a `complete` state.

```python
from openmed.core.language_readiness_matrix import (
    COMPLETE,
    LanguageReadinessEntry,
    build_language_readiness_matrix,
    readiness_entries_from_packs,
    render_language_readiness_markdown,
)

entries = [
    LanguageReadinessEntry("en", "detection", COMPLETE, "sha256:" + "a" * 64),
    LanguageReadinessEntry.from_tag(
        "iw", "surrogates", "partial", aliases={"iw": "he"}
    ),
]
matrix = build_language_readiness_matrix(entries, version="2026.09")
assert matrix["locales"] == ["en", "he"]
assert matrix["totals"]["complete"] == 1

registered = readiness_entries_from_packs()  # uses LANGUAGE_PACK_REGISTRY
print(render_language_readiness_markdown(registered, version="2026.09"))
```

## Closed capabilities and states

Capabilities are `detection`, `surrogates`, `validation`, `fixtures` and
`release_evidence`, always reported in that order. States are `complete`,
`partial`, `blocked` and `missing_evidence`. A locale/capability pair with no
record renders as the `not_reported` placeholder in Markdown only; the JSON
matrix carries reported cells, so `not_reported` never enters the closed state
vocabulary.

`complete` requires an evidence digest of the form `sha256:` followed by 64
lowercase hexadecimal characters. States that cannot cite evidence must omit it,
so a cell is either justified by a digest or explicitly unscored.

## Determinism

Locales are normalized and sorted ordinally, capabilities use the fixed order
above, and states are a closed tuple. JSON is written with sorted keys and
compact separators with no trailing newline; reordering the input records
therefore produces identical bytes. The unit tests pin the SHA-256 digests of a
fixed JSON render and a fixed Markdown render, so any change in the wire shape
fails the suite instead of silently changing released artifacts.

## Deriving from registered packs

`readiness_entries_from_packs(registry=...)` reads the coherence report for the
given `LanguagePackRegistry` (the process-local registry by default) and maps it
onto the readiness vocabulary:

- `detection` combines the `script`, `segmenter` and `recognizers` slots:
  all filled is `complete`, some filled is `partial`, otherwise
  `missing_evidence`;
- `surrogates` maps the `surrogate_locale` slot: filled is `complete`,
  approximated is `partial`, missing is `missing_evidence`;
- `validation` maps the national-ID status: filled is `complete`, `absent` is
  `missing_evidence`, and anything else is `blocked`;
- `fixtures` and `release_evidence` stay `missing_evidence`, because the pack
  registry does not record them.

A `complete` derived cell carries the SHA-256 of the canonical JSON of the pack
declaration that produced it, so the digest changes exactly when that
declaration changes. Derivation is bounded by the registry: one row of five
cells per registered pack, with no per-locale scanning and no network access.

## Legacy spellings

`LanguageReadinessEntry.from_tag` resolves caller-supplied aliases before the
record exists, delegating to `normalize_locale_tag`. Alias keys may use hyphens
or underscores, lookup is case-insensitive, and the caller mapping is never
mutated. Underscores are never converted implicitly: without an explicit alias,
`en_US` fails as `locale_tag_invalid`, matching the shared locale-tag helper.
The direct constructor stays strict and accepts normalized tags only, so
`to_dict()` output is always canonical.

## Failures

`LanguageReadinessError` extends `ValueError` with constant messages:
`unknown readiness capability`, `unknown readiness state`, `complete readiness
requires an evidence digest`, the evidence-digest shape message,
`duplicate readiness entry for locale ... capability ...`,
`locale must be a non-empty string`, `version must be a non-empty string`,
`entries must be an iterable of LanguageReadinessEntry values`, and
`aliases must be a mapping of locale tags` / `alias keys and values must be
strings`. Alias grammar, canonical targets, colliding keys and alias chains stay
with `normalize_locale_tag` and surface as `LanguageReadinessError` wrapping the
shared `LocaleTagError` category. No locale, digest or registry value is
embedded in the alias shape failures.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/core/test_language_readiness_matrix.py -q
```

The focused suite is offline and needs no model weights. It covers the closed
vocabularies, golden JSON and Markdown payloads, byte digests, reordered and
duplicated records, alias folding (including underscore spellings), malformed
aliases, an empty matrix, and derivation from both a custom registry and the
built-in catalog. Related grammar reference:
[Offline structural locale tags](locale-tags.md).
