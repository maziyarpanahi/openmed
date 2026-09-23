# Language pack plugins

A language pack declares all of the runtime metadata OpenMed needs to route,
detect, de-identify, and validate one language. Registration is process-local:
it does not discover plugins over the network, download model weights, or send
clinical text anywhere.

The built-in Chinese, Hindi, and Telugu packs are reference implementations in
`openmed/core/language_packs/`. Each is one inert definition module. The
built-in catalog imports and registers those definitions once; the modules do
not mutate global state when imported on their own.

## The one-call path

Create one module that defines a complete `LanguagePack`, then import that
definition from your application's startup code and call
`register_language_pack(pack)` once:

```python
# my_esperanto_pack.py
from openmed.core import LanguagePack

pack = LanguagePack(
    code="eo",
    scripts=("Latin",),
    default_model="/models/esperanto-pii",
    segmenter_id="unicode-sentence",
    recognizers=("builtin-patterns", "model"),
    surrogate_locale="en_US",
    surrogate_locale_approximation=(
        "Faker has no native Esperanto locale; generic fields use en_US"
    ),
    policy_overrides={"profile": "strict_no_leak"},
)
```

```python
# application startup
from openmed.core import (
    pack_coherence_report,
    register_language_pack,
    require_language_pack_coherence,
)

from my_esperanto_pack import pack

register_language_pack(pack)
require_language_pack_coherence()

coverage = next(
    row for row in pack_coherence_report() if row["language"] == pack.code
)
```

That single registration refreshes the live supported-language, default-model,
surrogate-locale, national-ID-provider, and script-routing adapters. No edit to
`pii_i18n.py`, `anonymizer/locales.py`, or `script_detect.py` is required.

Register during deterministic application startup, before constructing a
pipeline or serving requests. A duplicate code raises an error. Use
`register_language_pack(pack, replace=True)` only when your application
deliberately replaces an earlier process-local declaration.

## Capability contract

Every definition supplies these required values:

- a lowercase ISO 639-1 `code` and one or more Unicode `scripts`;
- a local path or model identifier in `default_model`;
- a registered `segmenter_id` (`jieba`, `pysbd`, or `unicode-sentence`);
- one or more `recognizers`;
- a resolvable `surrogate_locale`; and
- policy metadata through `policy_overrides` or `recall_floor_overrides`.

If Faker has no native locale, use a real installed backend and explain the
choice with `surrogate_locale_approximation`. The coverage report then marks
the slot `approximated` rather than silently presenting it as native support.
Script-aware surrogate providers can still preserve names in the source script,
as the Telugu reference pack does while generic fields use `en_IN`.

`national_id_providers` is optional. Declare it only when the matching
deterministic provider dispatch and checksum validator already exist. The
coherence gate generates synthetic values and requires every declared provider
to round-trip its validator.

## Validate before use

`pack_coherence_report()` returns deterministic, JSON-serializable rows. Each
row lists the five coverage slots—script, segmenter, recognizers, surrogate
locale, and policy—as `filled`, `approximated`, or `missing`. The `populated`
count includes both filled and explicitly approximated slots; it never includes
missing capabilities.

Call `require_language_pack_coherence()` during startup or CI to raise on an
unresolved segmenter, locale, provider round-trip, policy profile, or recall
floor. For command-style integrations,
`check_language_pack_coherence()` returns the number of incoherent packs, so a
non-zero value can be used directly as a failing gate.

Registration validates wiring, not recognizer accuracy. Evaluate recall,
critical leakage, span integrity, and script-specific behavior with synthetic
fixtures before using a new pack with clinical data. Do not log raw PHI in
coverage or validation artifacts.
