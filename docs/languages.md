# Per-Language PII De-identification

OpenMed's PII detection and de-identification are multilingual. Built-in
language packs live in
[`openmed.core.pii_i18n.SUPPORTED_LANGUAGES`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/pii_i18n.py).
Terminology maintainers and translators should also use the
[localization glossary](i18n/glossary.md) for stable product and privacy terms.
The optional Indic family can serve the built-in Assamese, Bengali, Gujarati,
Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu codes.
Every code documented here wires
up:

- a **default PII model** from `DEFAULT_PII_MODELS`, used when you pass `lang=`
  without an explicit `model_name=` (an `env:OPENMED_INDIC_NER_MODEL` entry
  means weights are optional and must be configured by the caller), and
- a **Faker locale** from `LANG_TO_LOCALE`
  ([`openmed/core/anonymizer/locales.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/anonymizer/locales.py)),
  used by `method="replace"` to generate locale-aware surrogates.

Pick a language by passing its ISO 639-1 code to `extract_pii()` or
`deidentify()`:

```python
from openmed import deidentify

deidentify("Paciente Pedro Almeida, CPF 123.456.789-09", lang="pt", method="mask")
```

For the redaction methods (`mask`, `remove`, `replace`, `hash`, `shift_dates`),
locale resolution, determinism, and cross-document surrogate vaults, see
[PII Anonymization](anonymization.md).

## Automatic language and script routing

The staged privacy pipeline can choose a document pack automatically while
preserving exact language decisions for every script run:

```python
from openmed.core.pipeline import Pipeline

route = Pipeline(lang="auto").stage2_language_script(
    "Patient stable. 患者发热。 रोगी स्थिर है।"
)
print(route.lang, route.model_name)
print(route.metadata["runs"])
```

The core fallback is deterministic and dependency-free. It combines Unicode
script runs with each `LanguagePack`'s candidate priority and context hints;
for example, adjacent kana selects Japanese for Han runs, while standalone Han
prefers Chinese and Marathi lexical markers disambiguate Marathi from the
otherwise Hindi-first Devanagari route. Install
`openmed[lid]` to enable the lazy, on-device `pycld2` adapter for ambiguous
runs. The adapter and its CLD2 implementation are Apache-2.0, import only when
routing is first requested, and do not download or bundle model weights.

Thai and Hebrew script runs select `th` and `he`. Sinhala script runs select
the experimental [`si` user-supplied-model route](#sinhala-si) instead of
English; it ships no weights, so automatic routing still needs an explicit
`model_name`. Myanmar, Khmer, Lao, Thaana, Tibetan and other unregistered
scripts are reported with
`source="stdlib:unknown-script"`, `reason="unsupported_script"` and low
confidence. `Pipeline(lang="auto")` rejects letter-bearing unsupported runs
before model inference. If a caller has independently qualified a fallback,
it can opt in with `LanguageRouter(fallback_pack="en")` passed as
`language_router=`; this is not a claim of language support. Numeric-only and
punctuation-only runs do not trigger that rejection. Routing diagnostics expose
only scripts, codes, counts and offsets, never note text.

Deterministic detection in both `deidentify()` and staged `Pipeline` folds all
Unicode decimal digits to ASCII and removes invisible bidi formatting controls
in an offset-mapped matching view. Validators see the folded values; redaction
spans still cover the original characters. Persian and Urdu matching also
folds declared Arabic keyboard kaf/yeh variants (and Urdu heh variants), without
rewriting the output or changing Arabic-language normalization. Diagnostics
contain normalization counts, not original or folded identifiers.

Numeric Gregorian dates accept `.`, `/` and `-` separators and use the shared
`LOCALE_DATE_ORDER` table for shifting and date surrogates. Valid year-first
forms remain year-first; ambiguous numeric dates without a known locale are
masked with `[DATE_SHIFTED]` rather than guessed. Hebrew attached prefixes are
not included in the redacted date span. This does not add calendar conversion
or model-quality qualification. The robustness operators `bidi_in_value` and
`native_digit_substitution` can exercise these matching boundaries with a
repeatable seed.

Synthetic training notes can carry native date forms too. `LocalePhiGenerator`
keeps ISO Gregorian ASCII dates by default; `date_profiles=` opts a language in
to Solar Hijri (`fa`), Thai Buddhist-era (`th`), Ethiopian (`am`), Lunar Hijri
with an era marker (`ar`, `ur`), or Bikram Sambat (`ne`, once its note route is
generated) dates, in numeric or month-name form and in native digits. The
profiles live in `openmed.training.synthetic.native_dates.NATIVE_DATE_PROFILES`;
`DEFAULT_NATIVE_DATE_PROFILES` picks one native-calendar profile per language.

```python
from openmed.training.synthetic.locale_phi import LocalePhiGenerator

example = LocalePhiGenerator(
    seed=7, date_profiles={"fa": "fa_solar_hijri_month_name"}
).generate("fa")
example.metadata["date_profile"]         # "fa_solar_hijri_month_name"
example.metadata["date_detection_gaps"]  # {}: the Persian patterns match it
```

Dates are sampled directly in their calendar, with no conversion. Months whose
length depends on an astronomical or published table (Esfand, every Lunar
Hijri month, every Bikram Sambat month) only use days 1-29, with days 1-28
for the archived 28-day Shaban 1364 AH in Umm al-Qura. Each date span
records value-free `calendar`, `numeral_system`, `form`, `date_profile`, and
`detection_gap` metadata, never the rendered date. `detection_gap` is `None`
when the language's deterministic date patterns return exactly the span, and
otherwise names the miss. The Ethiopian, Lunar Hijri, and Bikram Sambat
month-name profiles now have exact-span deterministic detection. Remaining
built-in gaps are `era_marker_not_matched` (numeric Hijri `هـ` and `ہجری`
markers) and `gregorian_fallback_only` (unmarked Bikram Sambat numerals).
Languages left out of `date_profiles` produce byte-identical examples.

!!! warning "Language-ID license boundary"
    This router deliberately excludes CLD3, which is outside this roadmap
    task's approved dependency scope. Non-commercial language-ID assets remain
    prohibited; only permissively licensed implementations may be bundled or
    referenced by the router.

!!! note "Kept in sync with the code"
    The table below lists every built-in code in `SUPPORTED_LANGUAGES`, plus
    the optional `INDIC_NER_LANGUAGES` and the `USER_SUPPLIED_MODEL_LANGUAGES`
    routes that ship no bundled weights, together with each code's
    `DEFAULT_PII_MODELS` entry and `LANG_TO_LOCALE` mapping.
    `tests/unit/test_docs_language_coherence.py` asserts this page matches the
    constants exactly, so a newly wired language fails the suite until it is
    documented here.

## Built-in and optional languages

| Code   | Language   | Default PII model                                          | Faker locale | Notes                                                        |
| ------ | ---------- | ---------------------------------------------------------- | ------------ | ----------------------------------------------------------- |
| `am`   | Amharic    | `OpenMed/privacy-filter-multilingual`                      | `am_ET`      | Ethiopic patterns and curated Ethiopic name/place surrogates; `en_KE` backs script-free fields and warns once. |
| `ar`   | Arabic     | `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1`    | `ar_EG`      | Egypt is the most-populous Arabic locale; override per call. Arabic-script name/place surrogates. |
| `as`   | Assamese   | `OpenMed/privacy-filter-multilingual`                       | `as_IN`      | Bengali Faker backend; curated Assamese names; Aadhaar.      |
| `bn`   | Bengali    | `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | `bn_BD`      | Dedicated Bengali PII registry entry.                        |
| `da`   | Danish     | `OpenMed/privacy-filter-multilingual`                       | `da_DK`      | CPR-aware Nordic language pack.                              |
| `cs`   | Czech      | `OpenMed/privacy-filter-multilingual`                       | `cs_CZ`      | Rodné číslo checksum-aware surrogates.                       |
| `de`   | German     | `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1`    | `de_DE`      | Steuer-ID surrogates via `GermanSteuerIdProvider`.           |
| `el`   | Greek      | `OpenMed/privacy-filter-multilingual`                       | `el_GR`      | AMKA Luhn-aware surrogates.                                  |
| `en`   | English    | `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`           | `en_US`      | Default model splits names into `first_name`/`last_name`.    |
| `es`   | Spanish    | `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1`   | `es_ES`      | DNI/NIE checksum-aware surrogates.                           |
| `fa`   | Persian    | `OpenMed/privacy-filter-multilingual`                      | `fa_IR`      | Iranian NID, RTL, Eastern digits, and Jalali date patterns. |
| `fr`   | French     | `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1`    | `fr_FR`      | NIR / INSEE; `fr_SN`, `fr_CI`, and `fr_CM` locale overlays.  |
| `gu`   | Gujarati   | `OpenMed/privacy-filter-multilingual`                       | `gu_IN`      | Native Gujarati digits, Aadhaar/PIN patterns, and ભાઈ/બેન name suffixes. |
| `he`   | Hebrew     | `OpenMed/privacy-filter-multilingual`                      | `he_IL`      | Served by the multilingual privacy filter.                   |
| `hi`   | Hindi      | `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1`    | `hi_IN`      | Aadhaar (Verhoeff) surrogates.                               |
| `id`   | Indonesian | `OpenMed/privacy-filter-multilingual`                      | `id_ID`      | Served by the multilingual privacy filter; NIK-aware.        |
| `it`   | Italian    | `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1`   | `it_IT`      | Codice Fiscale surrogates via `it_IT.ssn`.                   |
| `ja`   | Japanese   | `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1`        | `ja_JP`      | Family-name-first `PERSON` spans.                            |
| `kn`   | Kannada    | `OpenMed/privacy-filter-multilingual`                       | `kn_IN`      | Karnataka PIN/Aadhaar patterns; `en_IN` approximation warns once; initial-led names keep `ಅವರು` outside the span. |
| `ko`   | Korean     | `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1`        | `ko_KR`      | Resident Registration Number (RRN) surrogates.               |
| `ml`   | Malayalam  | `OpenMed/privacy-filter-multilingual`                      | `ml_IN`      | Chillu-safe patterns, Aadhaar; `en_IN` Faker approximation. |
| `mr`   | Marathi    | `OpenMed/privacy-filter-multilingual`                       | `mr_IN`      | Three-part names; `hi_IN` Faker approximation warns once.    |
| `ne`   | Nepali     | `user-supplied`                                             | `ne_NP`      | Native Faker locale; no bundled weights — pass `model_name`.  |
| `nl`   | Dutch      | `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1`    | `nl_NL`      | BSN (Elfproef) surrogates via `nl_NL.ssn`.                   |
| `no`   | Norwegian  | `OpenMed/privacy-filter-multilingual`                       | `no_NO`      | Fødselsnummer double modulus-11 validation.                  |
| `or`   | Odia       | `OpenMed/privacy-filter-multilingual`                       | `or_IN`      | Native Odia surrogates; Aadhaar and Odisha PIN patterns.     |
| `pa`   | Punjabi    | `OpenMed/privacy-filter-multilingual`                      | `pa_IN`      | Gurmukhi patterns; `en_IN` Faker approximation. |
| `pl`   | Polish     | `OpenMed/privacy-filter-multilingual`                      | `pl_PL`      | Native Faker locale, PESEL surrogates, and Polish dates, phones, postcodes, and addresses. |
| `pt`   | Portuguese | `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | `pt_PT`     | `pt_BR` IDs; `pt_MZ` and `pt_AO` locale overlays.            |
| `ro`   | Romanian   | `OpenMed/privacy-filter-multilingual`                      | `ro_RO`      | Served by the multilingual privacy filter; CNP-aware.        |
| `ru`   | Russian    | `OpenMed/privacy-filter-multilingual`                      | `ru_RU`      | Default-model placeholder; SNILS-aware. Dedicated weights are not bundled. |
| `si`   | Sinhala    | `user-supplied`                                             | `si_LK`      | Experimental; no bundled weights — pass `model_name`. Sri Lanka NIC and `+94` patterns; synthetic Sinhala surrogates. |
| `sv`   | Swedish    | `OpenMed/privacy-filter-multilingual`                       | `sv_SE`      | Personnummer Luhn validation and surrogates.                 |
| `sw`   | Swahili    | `OpenMed/privacy-filter-multilingual`                      | `sw`         | Bilingual patterns with Kenya ID and Maisha-aware surrogates. |
| `ta`   | Tamil      | `OpenMed/privacy-filter-multilingual`                       | `ta_IN`      | Public placeholder; dedicated weights are not bundled.       |
| `te`   | Telugu     | `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1`   | `en_IN`      | No Faker Telugu locale — `en_IN` approximation (warns once). |
| `th`   | Thai       | `OpenMed/privacy-filter-multilingual`                      | `th_TH`      | Served by the multilingual privacy filter; Thai NID-aware.   |
| `tr`   | Turkish    | `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1`   | `tr_TR`      | TCKN surrogates.                                             |
| `ur`   | Urdu       | `OpenMed/privacy-filter-multilingual`                      | `ur_IN`      | Indian ID patterns and Aadhaar surrogates; `en_IN` Faker backend warns once. |
| `xh`   | isiXhosa   | `OpenMed/privacy-filter-multilingual`                      | `xh_ZA`      | Nguni patterns; `zu_ZA` Faker approximation warns once.      |
| `zh`   | Chinese    | `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1`         | `zh_CN`      | Dedicated Chinese PII registry entry.                        |
| `zu`   | isiZulu    | `OpenMed/privacy-filter-multilingual`                      | `zu_ZA`      | Nguni patterns with checksum-valid South African ID support.  |
| `uk`   | Ukrainian  | `OpenMed/privacy-filter-multilingual`                       | `uk_UA`      | RNOKPP checksum-aware surrogates.                            |
| `vi`   | Vietnamese | `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | `vi_VN`      | Native `vi_VN` Faker locale; structural CCCD surrogates.     |

Chinese segmentation and Han-script routing use the dedicated `zh` registry
entry. Being listed above does **not** by itself mean a code is model-backed:
the rows whose model column reads `env:OPENMED_INDIC_NER_MODEL` or
`user-supplied` ship no bundled weights and require a caller-supplied model.
Malayalam, Punjabi, Polish, Russian, Tamil, and Urdu retain explicit public placeholder routes for
compatibility, but those routes are not claims of dedicated trained weights.
Codes absent from the table entirely (for
example `lv`, `sk`, `ms`, `tl`, and `fi`) are not model-backed either.
Several of them still have
validator-backed national-ID coverage
(`openmed.core.pii_i18n.NATIONAL_ID_ONLY_LANGUAGES`); see
[PII Anonymization](anonymization.md#clinical-id-checksums) for the ID providers.
Urdu defaults to `ur_IN` and Faker's installed `en_IN` backend for general
surrogate data, with a one-time approximation warning. Callers can still select
`ur_PK` explicitly for CNIC validation.

Arabic and Amharic name and place surrogates are documented approximations,
recorded in the `surrogate_approximation` field of
`openmed.core.anonymizer.locales.locale_coherence_report()`. Both follow the
source span's script: an Arabic-script or Ethiopic span gets a surrogate in
that script, and a Latin span in the same note keeps the Latin Faker fallback.

- `ar`: Faker's `ar_EG` locale ships no Arabic-script person or address data.
  `PERSON`, first-, middle-, and last-name surrogates use Faker's generic
  `ar_AA` name inventory. `LOCATION` (including `CITY`) and `STREET_ADDRESS`
  surrogates use synthetic curated Egyptian city names, generic street names,
  and street templates. A regional override whose Faker locale ships
  Arabic-script data, such as `locale="ar-DZ"`, keeps its own regional names
  and places. `ar_EG` stays the native default and emits no warning.
- `am`: Ethiopic name, city, and street surrogates recombine the synthetic
  curated Amharic values (given name plus father's name; city, kebele, and
  number). The `en_KE` backend supplies only script-free fields, and the
  one-time `en_KE` approximation warning applies only to them.

Phone and national-ID surrogates for both languages are unchanged and keep
their existing format and validator checks.

When a source exhausts a curated name or place inventory, a deterministic
same-script synthetic letter sequence replaces it without reusing source
tokens. Exhausting an address-number range raises a value-free error.

The optional Indic NER adapter never downloads a checkpoint automatically.
Set `OPENMED_INDIC_NER_MODEL` to a user-supplied local path or model repo, or
pass an explicit model. When it is unset, registry lookup returns no optional
model and the Naamapadam-style suite reports a structured skip reason.

`openmed.core.pii_i18n.USER_SUPPLIED_MODEL_LANGUAGES` holds the two codes that
are registered for script routing, surrogate locales, deterministic patterns,
and the public REST/MCP enums while claiming no bundled default model. The
Malayalam route has a different fallback contract:

- `ml` now has a built-in multilingual fallback. The optional Indic NER adapter
  remains available when the caller supplies its weights explicitly.
- `ne` and `si` read `user-supplied`. They have no adapter and no env-var
  path, so omitting `model_name` raises `ValueError: Language '<code>' has no
  bundled OpenMed PII model; pass an explicit model_name (user-supplied-model
  languages: ne, si)`.

The `user-supplied` column value is a registry placeholder rather than a
loadable repository. Passing it back as `model_name` raises the same error as
omitting it, so a value copied from `openmed_list_pii_languages` never becomes a
download attempt.
`SUPPORTED_LANGUAGES` includes registered fallback packs such as `pa` but excludes
the route that requires user-supplied weights.

## Indian-English and code-mixed clinical notes

Latin-script Hinglish routing remains explicit. Set `code_mixed=True` to add
the Roman-Hindi rule bank while keeping the English model route. When token
tags are omitted, OpenMed's stdlib-only fallback derives exact offset/label
records locally; no model weights or network access are required:

```python
from openmed import deidentify

result = deidentify(
    "Patient Ravi ka aadhaar 246778325484 hai.",
    lang="en",
    code_mixed=True,
)
```

Callers with a locally hosted token classifier can pass `lid_model=`. The hook
receives the input and offset-only token spans and must return one of `hi`,
`en`, `ne`, `univ`, or `other` for every span. Explicit
`token_language_tags=` still take precedence. Audit metadata retains offsets,
labels, and hashes rather than token surfaces.

For `lang="hi"`, `lang="ml"`, or `lang="te"`, a note containing both Latin and
the language's Indic script automatically activates the India clinical route.
OpenMed segments the note into offset-preserving script runs, adds bounded
context to each run so PERSON and LOCATION spans can cross a script boundary,
sends Latin windows to the registered English clinical model, and sends Indic
windows to the language's registered model. Caller-supplied model IDs or local
model paths are used for every window instead; OpenMed does not select an
unregistered third-party model automatically.

The documented first-party fallback is
`OpenMed/privacy-filter-multilingual`. Applications that choose that fallback
must supply it explicitly as `model_name`; the route does not silently switch
models after an inference error. Telugu replacement still uses the documented
`en_IN` Faker locale approximation and can emit its existing one-time warning.

Indian-English prescription abbreviations such as `Tab.`, `Cap.`, `OD`, `BD`,
`TDS`, `HS`, and `SOS` are normalized locally before entity merging. Source
text and offsets remain unchanged. All shipped fixtures are synthetic; any
restricted clinical corpus or separately trained weights remain user-supplied
or out of process.

!!! warning "Assistive output"
    India clinical NER output assists review and does not make clinical or
    disclosure decisions.

Hausa (`ha`) is available through that deterministic pattern-pack path with a
native `ha_NG` surrogate locale. Boko context cues cover dates, ages, Nigerian
NINs, and Nigerian/Nigerien phones. Ajami coverage is intentionally limited to
numeric patterns—phones, NINs, and dates written with Western or Arabic-Indic
digits—and does not claim Ajami lexical recognition or transliteration.

Yoruba (`yo`) is available through that deterministic pattern-pack path with a
native `yo_NG` surrogate locale. It covers Nigeria NINs and `+234` phones with
tone-marked, decomposed, or unmarked context cues, and replacement spans expand
to whole base-plus-diacritic clusters so dot-below and tone marks cannot be
orphaned.

Igbo (`ig`) is available through that deterministic pattern-pack path with a
native `ig_NG` surrogate locale. It recognizes Nigeria NINs, `+234` phones,
dates, and ages in Igbo and English-Igbo clinical text, including decomposed or
unmarked context cues. Replacement spans expand to whole base-plus-diacritic
clusters so dot-below vowels cannot be split or left behind.

Kinyarwanda (`rw`) is available through the same deterministic path with an
`rw_RW` surrogate locale. It recognizes Rwanda national IDs next to an
`indangamuntu` or national-ID cue, `+250` phones, and numeric dates.

### Month-name dates in Hausa, Yoruba, Igbo, and Kinyarwanda

Because these four routes ship no bundled model, deterministic patterns also
recognize Gregorian month-name dates in day-month-year (`15 Satumba 1980`) and
month-day-year (`Satumba 15, 1980`) order, with an optional comma before the
year. The vocabulary follows the CLDR month names plus common spellings:

| Code | Month names (January to December) | Also accepted |
|------|-----------------------------------|---------------|
| `ha` | Janairu, Faburairu, Maris, Afirilu, Mayu, Yuni, Yuli, Agusta, Satumba, Oktoba, Nuwamba, Disamba | Fabrairu, Afrilu; `15 ga watan Satumba 1980` |
| `yo` | Ṣẹ́rẹ́, Èrèlè, Ẹrẹ̀nà, Ìgbé, Ẹ̀bibi, Òkúdu, Agẹmọ, Ògún, Owewe, Ọ̀wàrà, Bélú, Ọ̀pẹ̀ | dot-below forms without tone marks; `15 oṣù Ṣẹ́rẹ́ ọdún 1980` |
| `ig` | Jenụwarị, Febrụwarị, Maachị, Epreel, Mee, Jun, Julaị, Ọgọọst, Septemba, Ọktoba, Novemba, Disemba | Febụwarị, Machị, Mach, Eprel, Juun, Ọgọstụ, Nọvemba |
| `rw` | Mutarama, Gashyantare, Werurwe, Mata, Gicurasi, Kamena, Nyakanga, Kanama, Nzeri, Ukwakira, Ugushyingo, Ukuboza | Nzeli |

Yoruba and Igbo names match in precomposed (NFC), decomposed (NFD), and fully
unmarked spellings, and spans never split a base letter from its tone or
dot-below marks. A month word alone, such as a clinic schedule naming
`Mutarama` or the Kinyarwanda word `mata` (milk), is not flagged: a day and a
four-digit year from 1900 to 2099 must be attached. Hausa Ajami month names,
Igbo ordinal month names (`ọnwa mbụ`), and native or lunar calendars are not
covered.

## Worked examples

Each example de-identifies synthetic, non-PHI text with `method="mask"`. The
exact placeholder tokens come from the chosen model's own entity labels, so
they can vary by model (see
[choosing a method](anonymization.md#quickstart-choosing-a-method)); the
canonical labels below are illustrative.

### Amharic — `am`

- Model: `OpenMed/privacy-filter-multilingual` · locale `am_ET`
  (`en_KE` is the documented Faker approximation for script-free fields;
  `method="replace"` keeps Ethiopic names and places in Ethiopic)

```text
Before: ስም፡ ሰላም ተስፋዬ። ስልክ፡ +251 911 234 567።
After:  ስም፡ [NAME]። ስልክ፡ [PHONE]።
```

### Arabic — `ar`

- Model: `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` · locale `ar_EG`
  (`method="replace"` keeps Arabic-script names and places in Arabic script)

```text
Before: المريضة ليلى حسن، الهاتف +20 10 1234 5678
After:  المريضة [NAME]، الهاتف [PHONE]
```

### Assamese — `as`

- Model: `OpenMed/privacy-filter-multilingual` · locale `as_IN`

```text
Before: অৰুণ গুৱাহাটীত জীৱন চিকিৎসালয়লৈ গ'ল।
After:  [PERSON] [LOCATION] [ORGANIZATION] গ'ল।
```

### Bengali — `bn`

- Model: `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` · locale `bn_BD`

```text
Before: অরুণ কলকাতায় আনন্দ হাসপাতালে গেলেন।
After:  [PERSON] [LOCATION] [ORGANIZATION] গেলেন।
```

### Danish — `da`

- Model: `OpenMed/privacy-filter-multilingual` · locale `da_DK`

```text
Before: Patient Anna Nielsen, CPR 170885-1234
After:  Patient [NAME], CPR [ID]
### Czech — `cs`

- Model: `OpenMed/privacy-filter-multilingual` · locale `cs_CZ`

```text
Before: Pacient Jan Novák, rodné číslo 751116/0008
After:  Pacient [NAME], rodné číslo [ID]
```

### German — `de`

- Model: `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` · locale `de_DE`

```text
Before: Patientin Anna Müller, Steuer-ID 86095742719
After:  Patientin [NAME], Steuer-ID [ID]
```

### Greek — `el`

- Model: `OpenMed/privacy-filter-multilingual` · locale `el_GR`

```text
Before: Ασθενής Γιώργος Παπαδόπουλος, ΑΜΚΑ 16117508024
After:  Ασθενής [NAME], ΑΜΚΑ [ID]
```

### English — `en`

- Model: `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` · locale `en_US`

```text
Before: Patient John Doe was seen on 03/14/2025; call 555-0142
After:  Patient [NAME] was seen on [DATE]; call [PHONE]
```

Pass `locale="en_GB"` for UK clinical text; with `lang="en"`, the `gb` and `uk`
aliases select the same detection overlay. The UK overlay adds deterministic NHS
Number, National Insurance number, and postcode rules. Postcodes are matched
without context, in every Royal Mail outward-code shape plus `GIR 0AA`, with a
single space, no space, or lowercase, and are labelled `ZIPCODE`. A full UK
postcode covers about 15 addresses, so the rule does not wait for the model.
Tokens whose inward code reads as a data size or an ordinal, such as `8GB` or
`2ND`, count only near address words, including common street types such as
Close, Drive and Way. Other locales, including plain `en`, keep
the US ZIP rule alone.

### Spanish — `es`

- Model: `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` · locale `es_ES`

```text
Before: Paciente Maria Garcia, DNI 12345678Z
After:  Paciente [NAME], DNI [ID]
```

### Persian — `fa`

- Model: `OpenMed/privacy-filter-multilingual` · locale `fa_IR`

The multilingual checkpoint is a compatibility placeholder until dedicated
Persian weights are published and evaluated. The built-in pack adds RTL
Persian patterns, Eastern-digit normalization, Jalali dates, and checksum-valid
Iranian national-ID surrogates.

```text
Before: بیمار با کد ملی ۱۲۳۴۵۶۷۸۹۱ در تاریخ ۱۴۰۵/۰۵/۲۹ مراجعه کرد.
After:  بیمار با کد ملی [ID] در تاریخ [DATE] مراجعه کرد.
```

### French — `fr`

- Model: `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` · locale `fr_FR`

```text
Before: Patient Jean Dupont, NIR 1 84 12 76 451 089 46
After:  Patient [NAME], NIR [ID]
```

### Gujarati — `gu`

- Model: `OpenMed/privacy-filter-multilingual` · locale `gu_IN`

```text
Before: રોગી શ્રી નરેશભાઈ. પિન કોડ ૩૮૦૦૦૧. આધાર ૨૪૬૭ ૭૮૩૨ ૫૪૮૪.
After:  રોગી શ્રી [PERSON]. પિન કોડ [ZIPCODE]. આધાર [ID_NUM].
```

### Kannada — `kn`

- Model: `OpenMed/privacy-filter-multilingual` · locale `kn_IN` (`en_IN` Faker backend)

```text
Before: ಕೃತಕ ಟಿಪ್ಪಣಿ: ಶ್ರೀ ಕೆ. ಎಸ್. ರವಿ ಅವರು, ಆಧಾರ್ ೨೪೬೭ ೭೮೩೨ ೫೪೮೪.
After:  ಕೃತಕ ಟಿಪ್ಪಣಿ: ಶ್ರೀ [PERSON] ಅವರು, ಆಧಾರ್ [ID_NUM].
```

### Malayalam — `ml`

- Model: `OpenMed/privacy-filter-multilingual` · locale `ml_IN` (`en_IN` Faker backend); an optional local Indic NER model can be supplied explicitly.

```text
Before: രോഗി ശ്രീ പുതുശ്ശേരി രാമൻ, ആധാർ ൨൪൬൭ ൭൮൩൨ ൫൪൮൪.
After:  രോഗി ശ്രീ [PERSON], ആധാർ [ID_NUM].
```

### Marathi — `mr`

- Model: `OpenMed/privacy-filter-multilingual` · locale `mr_IN`

```text
Before: रुग्णाचे नाव सौ. वैशाली सुरेश देशमुख. जन्मतारीख १४ फेब्रुवारी १९८५.
After:  रुग्णाचे नाव [PERSON]. जन्मतारीख [DATE].
```

### Nepali — `ne`

- Model: `user-supplied` · locale `ne_NP`
- Names, places, and organizations need the user-supplied model. The first
  worked example below assumes one is configured.
- Deterministic Nepali patterns run on this route even with an empty model
  detector, in Devanagari and ASCII digits:
  - `+977` and `00977` mobile numbers (96x-98x) and landlines. The country
    code stays inside the masked span. Domestic `0`-prefixed landlines need
    nearby phone context such as `फोन` or `सम्पर्क`.
  - Citizenship-certificate numbers in district-office-year-serial groups
    such as `27-01-75-05678`. Plain or slash-split serials count only directly
    after a number label (`नं.`, `नम्बर`) with `नागरिकता` context nearby.
  - National identity card numbers, either ten digits grouped `3-3-3-1` or
    ten digits after a number label, with `राष्ट्रिय परिचयपत्र` context.
  - Numeric Bikram Sambat dates, year-first or day-first, separated by `-`,
    `/`, or `.`. Day 32 is accepted for years 2000-2199. Earlier years must
    also be valid Gregorian dates.
  - Five-digit postal codes, gated on `हुलाक` or `postal code` context.
- Limits: identifier rules check shape only. They make no checksum or
  issuing-authority claim, and numbers without Nepali identifier context
  (doses, lab values) are not treated as identifiers. Bikram Sambat month
  names, calendar conversion, and Bikram Sambat-aware date shifting are not
  covered. The synthetic `openmed/eval/golden/fixtures/i18n/ne.jsonl` fixture
  checks spans and leakage. It is not a trained Nepali model or a
  clinical-quality claim.

```text
Before: सुमन थापा काठमाडौंको जीवन अस्पताल गए।
After:  [PERSON] [LOCATION]को [ORGANIZATION] गए।
```

```text
Before: जन्म मिति २०४५/०३/३२। मोबाइल +९७७ ९८४१२३४५६७। नागरिकता नं. २७-०१-७५-०५६७८।
After:  जन्म मिति [DATE]। मोबाइल [PHONE]। नागरिकता नं. [ID_NUM]।
```

### Odia — `or`

- Model: `OpenMed/privacy-filter-multilingual` · locale `or_IN`

```text
Before: ଅରୁଣ ଭୁବନେଶ୍ୱରରେ କଳିଙ୍ଗ ହସ୍ପିଟାଲକୁ ଗଲେ।
After:  [PERSON] [LOCATION] [ORGANIZATION] ଗଲେ।
```

### Punjabi — `pa`

- Model: `OpenMed/privacy-filter-multilingual` · locale `pa_IN` (approximated
  with Faker's `en_IN` backend; warns once)
- Built-in Gurmukhi patterns cover Punjabi dates, Indian mobile numbers,
  Aadhaar, Punjab/Chandigarh PIN codes, and honorific-anchored full names.
  Honorifics accept one to eight horizontal spaces, tabs, or non-breaking
  spaces without shifting the name span. These are bounded synthetic regression
  checks, not a trained Punjabi model or clinical-quality claim.
  Replacement names preserve `ਸਿੰਘ` and `ਕੌਰ` gender suffixes. A configured
  Indic adapter may still be selected explicitly.

```text
Before: ਅਰੁਣ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਜੀਵਨ ਹਸਪਤਾਲ ਗਿਆ।
After:  [PERSON] [LOCATION] ਵਿੱਚ [ORGANIZATION] ਗਿਆ।
```

### Tamil — `ta`

- Model: `OpenMed/privacy-filter-multilingual` (placeholder) · locale `ta_IN`

The formerly configured dedicated checkpoint is not in the public model
catalog. Treat the placeholder as compatibility routing, not qualified Tamil
model coverage; configure and evaluate explicit weights for production use.

```text
Before: அருண் சென்னையில் காவேரி மருத்துவமனை சென்றார்.
After:  [PERSON] [LOCATION] [ORGANIZATION] சென்றார்.
```

### Urdu — `ur`

This rules-and-surrogates pack uses a named default-model placeholder, not
verified trained Urdu coverage. Synthetic fixtures test source offsets and
bidi-safe masking, not clinical accuracy. Horizontal non-breaking spaces in
honorific names remain part of the original source span.

- Model: `OpenMed/privacy-filter-multilingual` · locale `ur_IN`

```text
Before: مریض محترمہ فاطمہ خان، آدھار ۲۴۶۷ ۷۸۳۲ ۵۴۸۴۔
After:  مریض [PERSON]، آدھار [ID_NUM]۔
```

### Hebrew — `he`

- Model: `OpenMed/privacy-filter-multilingual` · locale `he_IL`

```text
Before: מטופל דוד לוי, טלפון 054-1234567
After:  מטופל [NAME], טלפון [PHONE]
```

### Hindi — `hi`

- Model: `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` · locale `hi_IN`

```text
Before: रोगी अनीता शर्मा, फोन +91 9876543210
After:  रोगी [NAME], फोन [PHONE]
```

### Indonesian — `id`

- Model: `OpenMed/privacy-filter-multilingual` · locale `id_ID`

```text
Before: Pasien Budi Santoso, NIK 3201234567890123
After:  Pasien [NAME], NIK [ID]
```

### Italian — `it`

- Model: `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` · locale `it_IT`

```text
Before: Paziente Marco Rossi, codice fiscale RSSMRC80A01H501U
After:  Paziente [NAME], codice fiscale [ID]
```

### Japanese — `ja`

- Model: `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` · locale `ja_JP`

```text
Before: 患者 佐藤 花子、電話 +81 90 1234 5678
After:  患者 [NAME]、電話 [PHONE]
```

### Korean — `ko`

- Model: `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` · locale `ko_KR`

```text
Before: 환자 김민수, 전화 010-1234-5678
After:  환자 [NAME], 전화 [PHONE]
```

### Dutch — `nl`

- Model: `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` · locale `nl_NL`

```text
Before: Patiënt Eva de Vries, BSN 123456782
After:  Patiënt [NAME], BSN [ID]
```

### Norwegian — `no`

- Model: `OpenMed/privacy-filter-multilingual` · locale `no_NO`

```text
Before: Pasient Ingrid Hansen, fødselsnummer 12035101460
After:  Pasient [NAME], fødselsnummer [ID]
```

### Polish — `pl`

This is a rules-and-surrogates pack with a named default-model placeholder,
not a claim of trained Polish model coverage. Its synthetic leakage tests do
not establish clinical accuracy.

- Model: `OpenMed/privacy-filter-multilingual` · locale `pl_PL`

```text
Before: Pacjent, PESEL 85031512344, adres ul. Przykładowa 12.
After:  Pacjent, PESEL [ID_NUM], adres [STREET_ADDRESS].
```

### Portuguese — `pt`

- Model: `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` · locale `pt_PT`
  (pass `locale="pt_BR"` for Brazilian CPF/CNPJ surrogates)

```text
Before: Paciente Pedro Almeida, CPF 123.456.789-09
After:  Paciente [NAME], CPF [ID]
```

### Romanian — `ro`

- Model: `OpenMed/privacy-filter-multilingual` · locale `ro_RO`

```text
Before: Pacient Ion Popescu, CNP 1960101221144
After:  Pacient [NAME], CNP [ID]
```

### Russian — `ru`

- Model: `OpenMed/privacy-filter-multilingual` · locale `ru_RU`
- The multilingual model name is a routing placeholder until dedicated
  Russian weights ship; deterministic Cyrillic patterns remain available
  offline.

```text
Before: Пациент Иван Петров, СНИЛС 112-233-445 95
After:  Пациент [NAME], СНИЛС [ID]
```

### Sinhala — `si`

- Model: `user-supplied` · locale `si_LK`. This route is experimental and
  ships no weights: pass `model_name`, including with `lang="auto"`.
- Sinhala script runs route to `si` rather than to English. Names, places,
  and organizations need the user-supplied model. Tamil-script Sri Lankan
  text still routes to `ta`, which does not run the patterns below.
- Deterministic Sinhala patterns run on this route even with an empty model
  detector:
  - `+94` and `0094` numbers, with the country code inside the masked span,
    and domestic `070`-`078` mobiles. Domestic landlines need nearby phone
    context such as `දුරකථන`.
  - Sri Lanka National Identity Card (NIC) numbers in both layouts, nine
    digits plus `V` or `X` and twelve digits, only near NIC context such as
    `ජාතික හැඳුනුම්පත`, `ජා.හැ.අ` or `NIC`. Values whose day field is outside
    001-366 and 501-866, or whose twelve-digit birth year is before 1900 or
    after the current year, are rejected. No check digit is claimed.
  - Numeric dates in the shared day-first order, house-numbered Sinhala
    street addresses ending in a road word such as `පාර` or `මාවත` near
    address context, and five-digit postal codes after `තැපැල් කේතය`.
- The NIC layout adds 500 to the day field for one sex. That offset is a
  structural rule of the identifier. OpenMed accepts both ranges and never
  derives, reports, or carries a sex over to a surrogate.
- `si_LK` is a documented approximation that warns once. Names, cities, and
  street addresses come from a small repository-authored synthetic Sinhala
  inventory, other fields use the `en_IN` Faker backend, and NIC surrogates
  keep only the source layout while every embedded field is drawn at random.
- Limits: Sinhala month names are not covered. `lang="auto"` picks the
  dominant script run, so a mostly English note with a few Sinhala words
  routes to `en` and skips these patterns; pass `lang="si"` for such notes.
  The synthetic `openmed/eval/golden/fixtures/i18n/si.jsonl` fixture checks
  spans and leakage. It is not a trained Sinhala model or a clinical-quality
  claim.

```text
Before: උපන් දිනය 14/03/1985. ජාතික හැඳුනුම්පත් අංකය 657546289V. ජංගම +94 71 234 5678.
After:  උපන් දිනය [DATE]. ජාතික හැඳුනුම්පත් අංකය [ID_NUM]. ජංගම [PHONE].
```

### Swedish — `sv`

- Model: `OpenMed/privacy-filter-multilingual` · locale `sv_SE`

```text
Before: Patient Anna Andersson, personnummer 510312-1140
After:  Patient [NAME], personnummer [ID]
```

### Swahili — `sw`

- Model: `OpenMed/privacy-filter-multilingual` · locale `sw`

```text
Before: Jina: Amina Hassan. Nambari ya kitambulisho 12345678
After:  Jina: [NAME]. Nambari ya kitambulisho [ID]
```

### Telugu — `te`

- Model: `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` · locale `en_IN`
  (Faker has no Telugu locale — `en_IN` is a documented approximation)

```text
Before: రోగి రమేష్ కుమార్, ఫోన్ +91 9876543210
After:  రోగి [NAME], ఫోన్ [PHONE]
```

### Thai — `th`

- Model: `OpenMed/privacy-filter-multilingual` · locale `th_TH`

```text
Before: ผู้ป่วย สมชาย ใจดี โทร 081-234-5678
After:  ผู้ป่วย [NAME] โทร [PHONE]
```

### Turkish — `tr`

- Model: `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` · locale `tr_TR`

```text
Before: Hasta Ayşe Yılmaz, TCKN 10000000146
After:  Hasta [NAME], TCKN [ID]
```

### isiXhosa — `xh`

- Model: `OpenMed/privacy-filter-multilingual` · locale `xh_ZA`
  (`zu_ZA` is the documented Faker approximation)

```text
Before: Igama lesigulane: Xolani Qwabe. Inombolo yesazisi 7903116001080
After:  Igama lesigulane: [NAME]. Inombolo yesazisi [ID]
```

### Chinese — `zh`

- Model: `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` · locale `zh_CN`

```text
Before: 患者王芳，电话 13800138000
After:  患者[NAME]，电话 [PHONE]
```

The dedicated registry entry is used by default for Chinese PII detection,
while segmentation and exact-offset guarantees remain unchanged.

### isiZulu — `zu`

- Model: `OpenMed/privacy-filter-multilingual` · locale `zu_ZA`

```text
Before: Igama lesiguli: Nomcebo Dlamini. Inombolo kamazisi 8001015009087
After:  Igama lesiguli: [NAME]. Inombolo kamazisi [ID]
### Ukrainian — `uk`

- Model: `OpenMed/privacy-filter-multilingual` · locale `uk_UA`

```text
Before: Пацієнтка Олена Коваль, РНОКПП 2974281300
After:  Пацієнтка [NAME], РНОКПП [ID]
```

### Vietnamese — `vi`

- Model: `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` · locale `vi_VN`

Dates parse day-first (`17/08/1985` and `ngày 8 tháng 12 năm 1970`), mobile
numbers follow the `+84`/`0[3,5,7,8,9]` plan, and postal codes are the current
five-digit format. CCCD detection requires an explicit `CCCD` or
`căn cước công dân` cue and is validated structurally — the 12-digit contract
from Article 12 of the 2023 Law on Identification, with no public checksum.

```text
Before: Bệnh nhân Nguyễn Minh Anh, CCCD 001203123456
After:  Bệnh nhân [NAME], CCCD [ID]
```

## Non-Gregorian calendar dates

Deterministic patterns also recognize month-name dates written in four
non-Gregorian calendars. They run in the safety sweep and the semantic merger,
so they protect these dates even with an empty or user-supplied model.

| Calendar | Metadata code | Routes | Example | Orders and ranges |
|----------|---------------|--------|---------|-------------------|
| Lunar Hijri | `hijri` | `ar`, `fa`, `ur` | `١٥ رمضان ١٤٤٧ هـ` | day month year; years 1300-1499 |
| Ethiopian | `ethiopian` | `am` | `መስከረም 15 ቀን 2017 ዓ.ም` | day month year or month day year; 13 months; years 1900-2099 |
| Bikram Sambat | `bikram_sambat` | `ne` | `२०८२ असार १५ गते` | year month day, day month year or month day year; days up to 32; years 1900-2199 |
| Hebrew | `hebrew` | `he` | `ט״ו בשבט תשפ״ו` | day month year; years 5500-5899 or Hebrew-numeral years |

- Days and years accept ASCII digits and the route's native numerals:
  Arabic-Indic and Persian digits for Hijri dates, Devanagari digits for Bikram
  Sambat, Ethiopic numerals such as `፲፭ ጥር ፲፱፻፹፭`, and Hebrew-numeral days
  and years with geresh and gershayim (`ט״ו`, `תשפ״ו`, `ה׳תשפ״ו`). ASCII
  apostrophes and quotation marks are accepted in place of geresh and
  gershayim.
- Era markers stay inside the span: `هـ`, `هجري` and `للهجرة` (Arabic), `ه.ق`
  and `قمری` (Persian), `ہجری` and `ھ` (Urdu), `ዓ.ም` (Amharic), and `वि.सं.`
  (Nepali). Optional words such as Arabic `من شهر`, Persian `ماه`, Amharic
  `ቀን` and Nepali `साल` and `गते` are also accepted.
- Hijri month names match Arabic, Persian and Urdu letter forms, with or
  without hamza, harakat or a zero-width non-joiner (`ربيع الأول`,
  `ربیع‌الاول`, `ربیع الاوّل`), and with honorifics such as `رمضان المبارک`.
  Hebrew one-letter prefixes (`בט״ו`, `ב-15`) and the Amharic prefix `በ` stay
  outside the span, while `ב` on the month name (`בשבט`) stays inside it.
- A month name only matches with a day and a year attached. Month words used
  as ordinary words or names, such as `رمضان` in a sentence about fasting,
  Hebrew `שבט` ("tribe") or Nepali `जेठ` ("brother-in-law"), and partial dates
  such as `15 رمضان` or `رمضان 1447`, are not flagged. Gregorian month-name
  detection is unchanged.

Each matched span carries a value-free calendar code in
`metadata["calendar"]`, which shifting and surrogate stages can use to select
a calendar. Safety-sweep spans also record it under
`metadata["safety_sweep"]["calendar"]`. Spans without the key are Gregorian or
unspecified. The code never includes the date value, and nothing is logged.

```python
from openmed.core.safety_sweep import safety_sweep

spans = safety_sweep("تاريخ الميلاد: ١٥ رمضان ١٤٤٧ هـ", [], lang="ar")
print([(span.start, span.end, span.metadata["calendar"]) for span in spans])
# [(15, 31, 'hijri')]
```

Detection supplies offsets and calendar metadata. `method="shift_dates"`
uses that metadata for supported calendar-aware shifts, with a placeholder
fallback for unsupported values; see [Anonymization](anonymization.md).
Latin transliterations such as `Ramadan`,
`Meskerem` or `Asar`, dates written out in words, and lunisolar Vikram Samvat
dates written with a tithi are not covered. Persian solar-calendar month names
are matched by the Persian pack without a calendar code.

## Localized medical-record-number labels

Record numbers are direct identifiers, so the deterministic layer recognizes
them after localized record, file, patient, hospital and admission-number
labels, not only after the English `MRN`. The labels live in one declarative
table,
[`openmed.core.medical_record_labels.MEDICAL_RECORD_LABELS`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/medical_record_labels.py),
which compiles into context-gated `medical_record_number` patterns shared by
semantic merging and the safety sweep. With an empty model detector, the
number is still masked and its label stays readable:

```text
Before: ผู้ป่วย HN: 6501234 นัดตรวจซ้ำ
After:  ผู้ป่วย HN: [medical_record_number] นัดตรวจซ้ำ

Before: No. RM: 12.34.56, Hb 11,2 g/dL
After:  No. RM: [medical_record_number], Hb 11,2 g/dL
```

The table covers Amharic (`የካርድ ቁጥር`), Arabic (`رقم الملف الطبي`), Chinese
(`病历号` and `住院号` followed by decimal digits), Hausa (`Lambar fayil`),
Hebrew (`מספר תיק`), Hindi (`पंजीकरण संख्या`, `UHID`), Indonesian (`No. RM`),
Persian (`شماره پرونده`), Swahili (`Namba ya faili`), Thai (`HN`, `AN`),
Urdu (`ایم آر نمبر`) and Vietnamese (`Mã bệnh nhân`), plus related spellings.
English keeps its built-in `MRN` rule.

- **Label required.** A number is relabelled only when a listed label sits
  next to it, so unlabelled numbers, doses, lab values and phone numbers keep
  their existing handling. Count phrases such as Hindi `कुल रोगी संख्या` are
  deliberately not labels, and the Thai `HN`/`AN` abbreviations are
  case-sensitive.
- **Tolerated forms.** Any Unicode decimal digits (Thai, Devanagari,
  Arabic-Indic, Persian), 4–20 digits with single `-`, `/` or `.` separators,
  label punctuation `:`, `：`, `፡` and `#`, bidi controls between label and
  number, NFC, NFD and unaccented Latin spellings, and, for right-to-left
  scripts, a visual-order number written before its label (`1234567 :מספר תיק`).
- **Reports.** `medical_record_label_matches(text, lang)` returns offsets and
  the canonical table label for each match, never the number.
  `medical_record_label_coverage().to_dict()` returns language codes, label
  counts and the registered packs that still have no labels.

Adding a language needs one table entry and one synthetic row in
`tests/fixtures/pii/medical_record_labels_synthetic.jsonl` with labelled
numbers and negative controls; no detector code changes.
`tests/unit/core/test_medical_record_labels.py` fails until both exist and
pins the list of registered packs without labels. Institution-specific number
formats and checksums are out of scope.
