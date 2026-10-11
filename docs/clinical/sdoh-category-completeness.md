# SDOH Category Completeness

`audit_sdoh_completeness()` records what happened to every configured SDOH
category. The controlled states are `processed`, `skipped`, `unsupported`, and
`failed`.

```python
from openmed.clinical.sdoh_completeness import (
    SDOHCategoryResult,
    SDOHCategoryState,
    audit_sdoh_completeness,
)

audit = audit_sdoh_completeness(
    configured_categories=("food", "housing"),
    results=(
        SDOHCategoryResult(
            category="housing",
            state=SDOHCategoryState.PROCESSED,
            finding_count=0,
        ),
    ),
)
```

The example reports `housing` as processed with
`unmentioned_not_negative`. It reports the missing `food` processing record as
failed with `missing_processing_result`. This makes pipeline absence visible
without manufacturing a negative social-history finding.

Skipped, unsupported, and failed categories require a controlled reason code
and cannot claim findings. The audit exposes state and reason counts plus one
record per configured category. It rejects duplicate results and results for
unconfigured categories.

Category and reason codes must use a controlled lowercase format. Do not place
note text, patient identifiers, or free-text exception details in these fields.
The audit is deterministic, value-free, and performs no network calls.

## Explicit extraction language

Use `extract_sdoh_with_language()` when recording whether each category was
actually supported. The existing `extract_sdoh()` dispatcher remains available
for current English callers with the same findings and section behavior.

```python
from openmed.clinical.sdoh import (
    available_determinant_extractors,
    extract_sdoh_with_language,
)

result = extract_sdoh_with_language(
    "Social History: lives alone.", language="en",
)
audit = audit_sdoh_completeness(
    available_determinant_extractors(), result.category_results,
)
assert all(item.state is SDOHCategoryState.PROCESSED for item in audit.categories)
assert any(item.category == "living_status" for item in result.findings)

unsupported = extract_sdoh_with_language(
    "Social History: vive solo.", language="es",
)
assert not unsupported.findings
assert all(
    item.state is SDOHCategoryState.UNSUPPORTED
    for item in unsupported.category_results
)
```

The packaged social and substance cue tables explicitly declare English, as do
their registered extractors. Support requires both declarations. Spanish,
German and other unsupported languages yield `unsupported_language`, with zero
findings for every unsupported category. Support is never inferred from the
text or from an English cue that happens to appear in another language.

The `language` keyword is required. Caller-supplied `None` or an empty string
records `language_undeclared`. Tags are normalized for case and `_`/`-`; a
declared base tag covers regional tags such as `en-US`. Declaring a regional tag
does not confer support for every other region. Malformed tags raise a fixed
`invalid_sdoh_language` error without echoing the value.

Trusted custom extractors can declare support explicitly:

```python
from openmed.clinical.sdoh import (
    register_determinant_extractor,
    unregister_determinant_extractor,
)

def synthetic_custom_extractor(text, spans):
    return []  # A test double, not a qualified extractor.

register_determinant_extractor(
    "custom", synthetic_custom_extractor, languages=("es",),
)
try:
    result = extract_sdoh_with_language("Synthetic input.", language="es")
    custom = next(item for item in result.category_results if item.category == "custom")
    assert custom.state is SDOHCategoryState.PROCESSED
    assert custom.absence_interpretation == "unmentioned_not_negative"
finally:
    unregister_determinant_extractor("custom")
```

Omitting `languages` leaves support undeclared for the new entry point; the
legacy dispatcher still runs that callback. Replacing an extractor resets its
language declaration, and unregistering removes it. Registry category keys must
be controlled codes; callbacks must emit findings in their registered category
with valid source offsets. Language declarations describe caller-owned support,
not model qualification or clinical validation. Callbacks are trusted local
application code and are not sandboxed by this API.

`result.findings` retains the original typed findings for downstream handling.
Values and extents may be sensitive: do not log them. The result's representation
contains counts only, and `result.to_dict()` emits category states, counts,
controlled reason codes and source offsets. It omits note text, finding values,
extents, source language strings and callback exceptions.

| Outcome | Meaning |
| --- | --- |
| `processed` | Supported extractor completed; zero findings remains unmentioned. |
| `unsupported` / `language_undeclared` | Caller or extractor support was not declared. |
| `unsupported` / `unsupported_language` | Requested tag is not covered by the extractor or its built-in cue table. |
| `unsupported` / `cue_language_undeclared` | Built-in cue table lacks a language declaration. |
| `skipped` / `section_not_selected` | Section scope selected no Social History window; no callback ran. |
| `failed` / `cue_language_invalid` | Built-in cue-language metadata could not be validated. |
| `failed` / `scope_invalid` | Candidate or section scope could not be read or validated. |
| `failed` / `extractor_failed` | Callback failed or emitted invalid findings; that category's partial findings were discarded. |

Successful categories remain available when another callback fails. These
synthetic checks establish API behavior, not multilingual recall or clinical
performance. Non-English cue packs and models are outside this feature.
