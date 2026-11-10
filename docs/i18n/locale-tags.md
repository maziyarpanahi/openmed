# Offline structural locale tags

`normalize_locale_tag` normalizes the common BCP 47
`language[-Script][-REGION][-variant...]` structure. `normalize_locale_tags`
validates a sequence as a registry and rejects duplicate canonical entries.
The helpers use only the standard library. They do not change built-in language
registrations, aliases, quality scores, model selection or fallback behavior.

```python
from openmed.core.locale_tag import normalize_locale_tag, normalize_locale_tags

assert normalize_locale_tag("ZH-hant-tw") == "zh-Hant-TW"
assert normalize_locale_tags(["sr-latn-rs", "es-419"]) == ("sr-Latn-RS", "es-419")

aliases = {"en_US": "en-US", "iw": "he"}
assert normalize_locale_tag("EN_us", aliases=aliases) == "en-US"
assert aliases == {"en_US": "en-US", "iw": "he"}  # caller-owned mapping unchanged
```

## Deliberately limited grammar

Accepted components are ASCII language subtags of two to eight letters, an
optional four-letter script, an optional two-letter or three-digit region, and
variants of five to eight alphanumeric characters or four characters beginning
with a digit. Language/variants become lowercase, script title case, region
uppercase; numeric regions are unchanged. Repeated variants are rejected
case-insensitively. Tags are bounded to 255 characters.

This is structural validation, **not complete IANA registry conformance**.
It does not verify that a language, script, region or variant is registered or
that a variant has an appropriate language prefix. Extlangs, extensions,
private-use forms and grandfathered tags are outside this subset unless an
explicit alias maps one to an accepted structural form. No IANA registry is
bundled or downloaded. Whitespace is not silently stripped, and underscores
are not silently converted. Explicit aliases can cover legacy spellings.

## Explicit aliases and duplicate detection

Alias keys are ASCII alphanumeric subtags separated by hyphens or underscores;
lookup is case-insensitive. Targets must already normalize as supported tags.
Case-colliding keys, malformed entries and alias chains/cycles are rejected;
a target can also be an alias key only when that alias is an identity. The
caller mapping is validated before use and never mutated. No implicit legacy
mapping is applied: `iw` remains `iw` without an explicit alias.

`normalize_locale_tags` preserves input order and returns an immutable tuple.
Two entries mapping to the same canonical tag cause `locale_tag_duplicate`;
this includes different casing or aliases. Distinct regional tags remain distinct.
A single string or bytes object is not treated as an iterable of registry items.
The caller controls registry size; only each individual tag is length-bounded.

## Failures and privacy

`LocaleTagError` extends `ValueError`; `.category` and the string are identical
constant categories: `locale_tag_invalid`, `locale_variant_duplicate`,
`locale_tag_duplicate`, `locale_alias_invalid`, `locale_alias_duplicate`, or
`locale_alias_chain_unsupported`. Rejected tag and alias values are not embedded
in these errors. Wrong API types raise constant-message `TypeError`s.
These helpers neither detect a text's language nor evaluate translation quality
or clinical fitness. They validate caller-supplied registry identifiers only.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/core/test_locale_tag.py -q
```

Tests cover script/region/variant combinations, casing, aliases, malformed
entries, duplicate canonicals, order stability and a real JSON registry-file
read-normalize-write round trip. No model weights or network calls are needed.
The helper module itself is import-light; normal package imports still follow
OpenMed's existing parent-package initialization.

Grammar reference: [RFC 5646, section 2.1](https://www.rfc-editor.org/rfc/rfc5646.html#section-2.1).
