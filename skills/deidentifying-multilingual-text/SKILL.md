---
name: deidentifying-multilingual-text
description: "De-identify non-English clinical text on-device with OpenMed by passing lang= and locale= to deidentify(). Use when the user has Spanish, German, French, Italian, Portuguese, Dutch, Hindi, Telugu, Arabic, Japanese, or Turkish medical notes, needs locale-aware fake surrogates, must handle language-specific national IDs (DNI, NIR, Steuer-ID, codice fiscale, BSN, CPF, TCKN, Aadhaar), or asks which languages OpenMed PII supports. Covers SUPPORTED_LANGUAGES, get_pii_models_by_language, get_patterns_for_language, LANG_TO_LOCALE, and accent normalization. Pairs with OpenMed deidentifying-clinical-text and generating-synthetic-surrogates."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: adjacent
  version: "1.0"
---

# De-identifying multilingual text

OpenMed de-identifies clinical text in many languages, each with a dedicated
PII model, language-specific regex patterns (national IDs, phone formats), and a
locale-aware surrogate generator. Pass `lang=` to `deidentify` / `extract_pii`
and the right model, patterns, and fake-data tables are selected automatically.
Everything runs **on-device**.

## When to use this skill

Use it whenever the source text is not English, or when surrogates must look
native to the locale (a German note should get German-looking fake names and a
valid-format Steuer-ID surrogate, not a US SSN).

## Discover supported languages at runtime — don't hardcode

```python
import openmed
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES, get_patterns_for_language

print(sorted(SUPPORTED_LANGUAGES))   # query it; the set is the source of truth
# Language-appropriate default model for a code:
models = openmed.get_pii_models_by_language("es")
# Language-specific regex patterns (national IDs, phones, etc.):
patterns = get_patterns_for_language("de")
```

The set currently spans English plus European, South Asian, Middle Eastern, and
East Asian languages — but **always read `SUPPORTED_LANGUAGES`** rather than
trusting a number, since it changes as models ship. MCP exposes the same list
via `openmed_list_pii_languages`.

## Quick start (Spanish)

```python
import openmed

nota = (
    "El paciente Carlos Hernández (DNI 12345678Z), nacido el 11/04/1979, "
    "vive en Calle Mayor 5, Madrid. Teléfono 612 345 678."
)

result = openmed.deidentify(
    nota,
    lang="es",                # selects the Spanish PII model + ES patterns
    method="replace",         # locale-native fake values
    locale="es_ES",           # Faker locale (defaults from lang via LANG_TO_LOCALE)
)
print(result.deidentified_text)
# El paciente [surrogate name] (DNI [surrogate]), nacido el [date], ...
```

For German, just switch the code:

```python
befund = "Patientin Anna Müller, geb. 11.04.1979, Steuer-ID 12 345 678 901."
result = openmed.deidentify(befund, lang="de", method="replace")
```

## Workflow

1. **Confirm the language is supported** by checking `SUPPORTED_LANGUAGES`.
2. **Pass `lang=`** to `deidentify`/`extract_pii`. This selects the
   language-specific model (via `get_pii_models_by_language`) *and* the regex
   pattern set (via `get_patterns_for_language`) for national IDs and formats.
3. **Set `locale=` for surrogates** when `method="replace"`. If omitted, the
   locale is derived from `lang` through `LANG_TO_LOCALE` (e.g. `pt`→`pt_PT`).
   Override for regional variants (`pt_BR`, `en_GB`, Gulf/Levant Arabic).
4. **Let accent normalization happen.** For models trained on accent-free text
   (Spanish), `deidentify` auto-strips diacritics before inference and maps
   spans back to the *original* accented text. You normally do not set
   `normalize_accents` yourself.
5. **Keep surrogates stable** across a document with `consistent=True, seed=...`.

## Language-specific national IDs

The pattern sets encode and the validators check real national identifier
formats and checksums, so structured IDs are caught even when the model is
unsure. Examples available in `openmed.core.pii_i18n`:

| Language | Identifier | Validator |
| --- | --- | --- |
| French | NIR / INSEE | `validate_french_nir` |
| German | Steuer-ID | `validate_german_steuer_id` |
| Italian | Codice Fiscale | `validate_italian_codice_fiscale` |
| Spanish | DNI / NIE | `validate_spanish_dni`, `validate_spanish_nie` |
| Dutch | BSN | `validate_dutch_bsn` |
| Hindi | Aadhaar | `validate_aadhaar` |
| Portuguese | CPF / CNPJ | `validate_portuguese_cpf`, `validate_portuguese_cnpj` |
| Turkish | TCKN | `validate_turkish_tckn` |

These map to OpenMed `CANONICAL_LABELS` (`ID_NUM`, `SSN`) and are redacted by
the same policy actions as any other identifier.

## Hand-off to / from OpenMed

- **Core de-id:** `deidentifying-clinical-text` — methods, thresholds,
  `keep_mapping`, policies (all accept `lang`/`locale`).
- **Surrogates:** `generating-synthetic-surrogates` — locale-native fakes and
  custom providers per language.
- **Policies:** `configuring-privacy-policies` — `policy=` works with any `lang`.
- **Audit:** `auditing-deidentification-runs` records the model and language in
  the no-PHI report.
- **Other surfaces:** MCP `openmed_deidentify` / `openmed_list_pii_languages`;
  REST `POST /pii/deidentify` (both take a language parameter).

## Edge cases & gotchas

- **Never run the English model on other languages.** Recall collapses. Always
  pass `lang=`; the default model is English only.
- **`lang` ≠ `locale`.** `lang` picks the *detection* model and patterns;
  `locale` shapes the *replacement* fakes. Set both when surrogate realism
  matters (e.g. `lang="pt", locale="pt_BR"`).
- **Some locales are approximations.** Faker has no Telugu locale, so OpenMed
  maps `te`→`en_IN` and warns once. Override `locale=` if you need closer
  regional surrogates.
- **Date order varies.** Day-first languages (fr, de, it, es, nl, pt, …) parse
  `11/04/1979` as 11 April; the date logic is language-aware — keep `lang` set
  when shifting dates.
- **Mixed-language notes** (e.g. English headers in a Spanish chart) may lower
  recall; verify residual risk with `audit=True` and consider a second pass.
- **No raw PHI in logs** regardless of language — offsets, labels, hashes only.

## Standards & references

- HIPAA de-identification, 45 CFR 164.514(b) (US) and GDPR / national DPAs for
  EU subjects: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- National ID format references are encoded in OpenMed validators (no external
  registry bundled).
- OpenMed source: `openmed/core/pii_i18n.py` (`SUPPORTED_LANGUAGES`,
  `get_patterns_for_language`, validators), `openmed/core/model_registry.py`
  (`get_pii_models_by_language`), `openmed/core/anonymizer/locales.py`
  (`LANG_TO_LOCALE`).
