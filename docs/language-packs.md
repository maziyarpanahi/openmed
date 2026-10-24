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

## Activate an offline language bundle

`openmed.core.language_pack_activation` connects an already-installed offline
language bundle to the registry. It does not package bundles, discover them,
download models, or contact a network service. The caller supplies three
local inputs and a trust anchor:

- the bundle manifest as plain data (for example from `json.load`);
- `LanguageReadinessEvidence` for each exact language tag in the bundle;
- a `LocalModelHandle` for each model artifact the installer already placed
  on disk, keyed by the manifest's artifact alias; and
- a `BundleSignatureVerifier`, an object with a `verify()` method bound to the
  keys your release process trusts. OpenMed ships no signing keys.

```python
from datetime import timedelta

from openmed.core.language_pack_activation import (
    LanguageBundleActivator,
    LanguageReadinessEvidence,
    LocalModelHandle,
)

activator = LanguageBundleActivator(
    verifier=my_release_verifier,
    max_readiness_age=timedelta(days=30),
)
record = activator.activate(
    manifest,
    [LanguageReadinessEvidence.from_dict(item) for item in readiness_rows],
    {"lt-pii": LocalModelHandle(reference=installed_model_dir, sha256=digest)},
)

activator.qualified_tasks("lt")  # ("deidentification", "ner")
activator.clinical_claim("lt")  # "supported" or "preview"
```

### Manifest contract

The schema `1.0` manifest is a JSON object with `schema_version`, `bundle_id`,
`bundle_version` (`MAJOR.MINOR.PATCH`), `artifacts`, `packs`, and `signature`.
`artifacts` maps an alias to `{"kind": ..., "sha256": "sha256:<hex>"}`, where
`kind` is `model`, `lexicon`, `policy`, `self_test`, or `evidence`. Each entry
in `packs` declares a canonical BCP 47 `language` tag, the `model` artifact
alias, the `tasks` the bundle claims, and the usual `LanguagePack` fields
(`scripts`, `segmenter_id`, `recognizers`, `surrogate_locale`, and the
optional overrides). The registry code is the tag's primary language subtag.

The signature covers `language_bundle_signing_payload(manifest)`: the
manifest without `signature`, serialized as UTF-8 JSON with sorted keys and
compact separators. Packaging tools sign exactly those bytes, and
`language_bundle_digest(manifest)` identifies them. After the signature is
verified, activation reads the manifest from the verified bytes only.

Every field in schema `1.0` is critical. An unknown top-level, pack, artifact,
or signature field refuses the bundle instead of being ignored. Optional
`extensions` are signed but informational, unless their names appear in
`critical`; this version understands no critical extensions, so any entry
there refuses the bundle.

### What activation checks

Activation validates everything before it commits anything:

1. the schema version and the signature, through the injected verifier;
2. unknown critical fields and the manifest structure;
3. that each model handle is an absolute local path whose measured digest
   equals the signed artifact digest (URLs and hub identifiers are refused);
4. readiness evidence for the pack's exact language tag: not older than
   `max_readiness_age`, not issued in the future beyond `clock_skew`, in the
   `preview` or `supported` state, and, when the evidence names a
   `model_digest`, measured on the same model;
5. that at least one declared task is qualified by the evidence;
6. pack coherence, using the same checks as `pack_coherence_report()`; and
7. pack identity and capability conflicts with every pack the activator does
   not own: an existing code, a positive `candidate_priority` on a script an
   existing pack serves, or a routing marker an existing pack already uses.

Only tasks that are both declared by the bundle and qualified by the evidence
are activated. Declared tasks without evidence are listed as `withheld_tasks`
in the record; evidence for tasks the bundle does not declare is ignored.

All packs in a bundle are committed as one registry generation, so live
adapters see either the previous state or the complete bundle. A refusal
raises `LanguageBundleActivationError` with a controlled `reason`, such as
`signature_invalid`, `readiness_stale`, `unknown_critical_field`, or
`pack_identity_conflict`, and leaves the previous registry state unchanged.
Messages never contain manifest values, signatures, model paths, or evidence
content.

### Clinical-language claims

Registration wires routing and models; it is not a clinical-language claim.
`clinical_claim(tag)` and `qualified_tasks(tag)` match the exact canonical
tag: evidence for `pt-BR` never qualifies `pt`, and evidence for `pt` never
qualifies `pt-BR`. A pack whose surrogate locale is approximated, either
declared with `surrogate_locale_approximation` or detected by the coherence
report, is always `preview`, even when its evidence is `supported`. A surrogate
locale whose language differs from the pack code must declare the
approximation, or activation is refused with
`locale_approximation_undeclared`.

### Rollback, upgrades, and reload

Packs registered by other code, including built-in packs and packs registered
with `register_language_pack()`, are never replaced or removed. A bundle can
supersede only its own earlier activation, and only with a strictly newer
`bundle_version`. `activator.rollback(record.activation_id)` removes the
bundle's packs and restores the superseded version exactly, as one generation.
If other code replaced one of those packs or took a code that would be
restored, rollback is refused with `rollback_conflict` instead of guessing.

`LanguageBundleActivation.to_json()` holds only codes, versions, timestamps,
and digests. `LanguageBundleActivation.from_dict()` rejects a record whose
`activation_id` no longer matches its content. To reload an activation in a
new process, activate the same manifest and evidence with
`expected_activation_id=record.activation_id`; any drift refuses the
activation with `activation_id_mismatch`.
