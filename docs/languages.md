# Per-Language PII De-identification

OpenMed's PII detection and de-identification are multilingual. Built-in
language packs live in
[`openmed.core.pii_i18n.SUPPORTED_LANGUAGES`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/pii_i18n.py).
The optional Indic family adds nine user-configured routes and can also serve
the built-in Hindi and Telugu codes. Every code documented here wires up:

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
prefers Chinese and Devanagari currently prefers Hindi. Install
`openmed[lid]` to enable the lazy, on-device `pycld2` adapter for ambiguous
runs. The adapter and its CLD2 implementation are Apache-2.0, import only when
routing is first requested, and do not download or bundle model weights.

!!! warning "Language-ID license boundary"
    This router deliberately excludes CLD3, which is outside this roadmap
    task's approved dependency scope. Non-commercial language-ID assets remain
    prohibited; only permissively licensed implementations may be bundled or
    referenced by the router.

!!! note "Kept in sync with the code"
    The table below lists every built-in code in `SUPPORTED_LANGUAGES` plus the
    optional `INDIC_NER_LANGUAGES`, together with its `DEFAULT_PII_MODELS`
    entry and `LANG_TO_LOCALE` mapping.
    `tests/unit/test_docs_language_coherence.py` asserts this page matches the
    constants exactly, so a newly wired language fails the suite until it is
    documented here.

## Built-in and optional languages

| Code   | Language   | Default PII model                                          | Faker locale | Notes                                                        |
| ------ | ---------- | ---------------------------------------------------------- | ------------ | ----------------------------------------------------------- |
| `am`   | Amharic    | `OpenMed/privacy-filter-multilingual`                      | `am_ET`      | Ethiopic patterns; `en_KE` Faker approximation warns once.   |
| `ar`   | Arabic     | `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1`    | `ar_EG`      | Egypt is the most-populous Arabic locale; override per call. |
| `as`   | Assamese   | `env:OPENMED_INDIC_NER_MODEL`                               | `as_IN`      | Optional Indic NER weights; Bengali Faker backend.           |
| `bn`   | Bengali    | `env:OPENMED_INDIC_NER_MODEL`                               | `bn_BD`      | Optional Indic NER weights.                                  |
| `da`   | Danish     | `OpenMed/privacy-filter-multilingual`                       | `da_DK`      | CPR-aware Nordic language pack.                              |
| `de`   | German     | `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1`    | `de_DE`      | Steuer-ID surrogates via `GermanSteuerIdProvider`.           |
| `en`   | English    | `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`           | `en_US`      | Default model splits names into `first_name`/`last_name`.    |
| `es`   | Spanish    | `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1`   | `es_ES`      | DNI/NIE checksum-aware surrogates.                           |
| `fr`   | French     | `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1`    | `fr_FR`      | NIR / INSEE; `fr_SN`, `fr_CI`, and `fr_CM` locale overlays.  |
| `gu`   | Gujarati   | `env:OPENMED_INDIC_NER_MODEL`                               | `gu_IN`      | Optional Indic NER weights.                                  |
| `he`   | Hebrew     | `OpenMed/privacy-filter-multilingual`                      | `he_IL`      | Served by the multilingual privacy filter.                   |
| `hi`   | Hindi      | `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1`    | `hi_IN`      | Aadhaar (Verhoeff) surrogates.                               |
| `id`   | Indonesian | `OpenMed/privacy-filter-multilingual`                      | `id_ID`      | Served by the multilingual privacy filter; NIK-aware.        |
| `it`   | Italian    | `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1`   | `it_IT`      | Codice Fiscale surrogates via `it_IT.ssn`.                   |
| `ja`   | Japanese   | `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1`        | `ja_JP`      | Family-name-first `PERSON` spans.                            |
| `kn`   | Kannada    | `env:OPENMED_INDIC_NER_MODEL`                               | `kn_IN`      | Optional Indic NER weights; Indian Faker fallback.           |
| `ko`   | Korean     | `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1`        | `ko_KR`      | Resident Registration Number (RRN) surrogates.               |
| `ml`   | Malayalam  | `env:OPENMED_INDIC_NER_MODEL`                               | `ml_IN`      | Optional Indic NER weights; Indian Faker fallback.           |
| `mr`   | Marathi    | `env:OPENMED_INDIC_NER_MODEL`                               | `mr_IN`      | Optional Indic NER weights; Hindi Faker backend.             |
| `nl`   | Dutch      | `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1`    | `nl_NL`      | BSN (Elfproef) surrogates via `nl_NL.ssn`.                   |
| `no`   | Norwegian  | `OpenMed/privacy-filter-multilingual`                       | `no_NO`      | Fødselsnummer double modulus-11 validation.                  |
| `or`   | Odia       | `env:OPENMED_INDIC_NER_MODEL`                               | `or_IN`      | Optional Indic NER weights.                                  |
| `pa`   | Punjabi    | `env:OPENMED_INDIC_NER_MODEL`                               | `pa_IN`      | Optional Indic NER weights; Indian Faker fallback.           |
| `pt`   | Portuguese | `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | `pt_PT`     | `pt_BR` IDs; `pt_MZ` and `pt_AO` locale overlays.            |
| `ro`   | Romanian   | `OpenMed/privacy-filter-multilingual`                      | `ro_RO`      | Served by the multilingual privacy filter; CNP-aware.        |
| `ru`   | Russian    | `OpenMed/privacy-filter-multilingual`                      | `ru_RU`      | Served by the multilingual privacy filter; SNILS-aware.      |
| `sv`   | Swedish    | `OpenMed/privacy-filter-multilingual`                       | `sv_SE`      | Personnummer Luhn validation and surrogates.                 |
| `sw`   | Swahili    | `OpenMed/privacy-filter-multilingual`                      | `sw`         | Bilingual patterns with Kenya ID and Maisha-aware surrogates. |
| `ta`   | Tamil      | `env:OPENMED_INDIC_NER_MODEL`                               | `ta_IN`      | Optional Indic NER weights.                                  |
| `te`   | Telugu     | `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1`   | `en_IN`      | No Faker Telugu locale — `en_IN` approximation (warns once). |
| `th`   | Thai       | `OpenMed/privacy-filter-multilingual`                      | `th_TH`      | Served by the multilingual privacy filter; Thai NID-aware.   |
| `tr`   | Turkish    | `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1`   | `tr_TR`      | TCKN surrogates.                                             |
| `xh`   | isiXhosa   | `OpenMed/privacy-filter-multilingual`                      | `xh_ZA`      | Nguni patterns; `zu_ZA` Faker approximation warns once.      |
| `zh`   | Chinese    | `OpenMed/privacy-filter-multilingual`                      | `zh_CN`      | Routing placeholder; no dedicated Chinese PII model yet.     |
| `zu`   | isiZulu    | `OpenMed/privacy-filter-multilingual`                      | `zu_ZA`      | Nguni patterns with checksum-valid South African ID support.  |

Chinese segmentation and Han-script routing are supported, but the `zh`
default remains an explicit multilingual placeholder rather than a claim that
a dedicated Chinese PII model has shipped. Codes outside this list (for example
`pl`, `lv`, `sk`, `ms`, `tl`, `fi`, and `ur`) are **not** model-backed PII languages.
Several of them still have
validator-backed national-ID coverage
(`openmed.core.pii_i18n.NATIONAL_ID_ONLY_LANGUAGES`); see
[PII Anonymization](anonymization.md#clinical-id-checksums) for the ID providers.
Urdu uses the conceptual `ur_PK` locale for CNIC dispatch and Faker's installed
`en_PK` backend for general surrogate data, with a one-time approximation warning.

The nine optional Indic language packs never download a default checkpoint.
Set `OPENMED_INDIC_NER_MODEL` to a user-supplied local path or model repo, or
pass an explicit model. When it is unset, registry lookup returns no optional
model and the Naamapadam-style suite reports a structured skip reason.

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

For `lang="hi"` or `lang="te"`, a note containing both Latin and Devanagari
(or Latin and Telugu) automatically activates the India clinical route. OpenMed
segments the note into offset-preserving script runs, adds bounded context to
each run so PERSON and LOCATION spans can cross a script boundary, sends Latin
windows to the registered English clinical model, and sends Indic windows to
the language's registered Hindi or Telugu model. Caller-supplied model IDs or
local model paths are used for every window instead; OpenMed does not select an
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

## Worked examples

Each example de-identifies synthetic, non-PHI text with `method="mask"`. The
exact placeholder tokens come from the chosen model's own entity labels, so
they can vary by model (see
[choosing a method](anonymization.md#quickstart-choosing-a-method)); the
canonical labels below are illustrative.

### Amharic — `am`

- Model: `OpenMed/privacy-filter-multilingual` · locale `am_ET`
  (`en_KE` is the documented Faker approximation)

```text
Before: ስም፡ ሰላም ተስፋዬ። ስልክ፡ +251 911 234 567።
After:  ስም፡ [NAME]። ስልክ፡ [PHONE]።
```

### Arabic — `ar`

- Model: `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` · locale `ar_EG`

```text
Before: المريضة ليلى حسن، الهاتف +20 10 1234 5678
After:  المريضة [NAME]، الهاتف [PHONE]
```

### Assamese — `as`

- Model: `env:OPENMED_INDIC_NER_MODEL` · locale `as_IN`

```text
Before: অৰুণ গুৱাহাটীত জীৱন চিকিৎসালয়লৈ গ'ল।
After:  [PERSON] [LOCATION] [ORGANIZATION] গ'ল।
```

### Bengali — `bn`

- Model: `env:OPENMED_INDIC_NER_MODEL` · locale `bn_BD`

```text
Before: অরুণ কলকাতায় আনন্দ হাসপাতালে গেলেন।
After:  [PERSON] [LOCATION] [ORGANIZATION] গেলেন।
```

### Danish — `da`

- Model: `OpenMed/privacy-filter-multilingual` · locale `da_DK`

```text
Before: Patient Anna Nielsen, CPR 170885-1234
After:  Patient [NAME], CPR [ID]
```

### German — `de`

- Model: `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` · locale `de_DE`

```text
Before: Patientin Anna Müller, Steuer-ID 86095742719
After:  Patientin [NAME], Steuer-ID [ID]
```

### English — `en`

- Model: `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` · locale `en_US`

```text
Before: Patient John Doe was seen on 03/14/2025; call 555-0142
After:  Patient [NAME] was seen on [DATE]; call [PHONE]
```

### Spanish — `es`

- Model: `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` · locale `es_ES`

```text
Before: Paciente Maria Garcia, DNI 12345678Z
After:  Paciente [NAME], DNI [ID]
```

### French — `fr`

- Model: `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` · locale `fr_FR`

```text
Before: Patient Jean Dupont, NIR 1 84 12 76 451 089 46
After:  Patient [NAME], NIR [ID]
```

### Gujarati — `gu`

- Model: `env:OPENMED_INDIC_NER_MODEL` · locale `gu_IN`

```text
Before: આરવ અમદાવાદમાં જીવન હોસ્પિટલ ગયા.
After:  [PERSON] [LOCATION] [ORGANIZATION] ગયા.
```

### Kannada — `kn`

- Model: `env:OPENMED_INDIC_NER_MODEL` · locale `kn_IN`

```text
Before: ಅರುಣ್ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಕಾವೇರಿ ಆಸ್ಪತ್ರೆಗೆ ಹೋದರು.
After:  [PERSON] [LOCATION] [ORGANIZATION] ಹೋದರು.
```

### Malayalam — `ml`

- Model: `env:OPENMED_INDIC_NER_MODEL` · locale `ml_IN`

```text
Before: അരുൺ കൊച്ചിയിൽ അമൃത ആശുപത്രിയിൽ പോയി.
After:  [PERSON] [LOCATION] [ORGANIZATION] പോയി.
```

### Marathi — `mr`

- Model: `env:OPENMED_INDIC_NER_MODEL` · locale `mr_IN`

```text
Before: आरव पुण्यात सह्याद्री रुग्णालयात गेला.
After:  [PERSON] [LOCATION] [ORGANIZATION] गेला.
```

### Odia — `or`

- Model: `env:OPENMED_INDIC_NER_MODEL` · locale `or_IN`

```text
Before: ଅରୁଣ ଭୁବନେଶ୍ୱରରେ କଳିଙ୍ଗ ହସ୍ପିଟାଲକୁ ଗଲେ।
After:  [PERSON] [LOCATION] [ORGANIZATION] ଗଲେ।
```

### Punjabi — `pa`

- Model: `env:OPENMED_INDIC_NER_MODEL` · locale `pa_IN`

```text
Before: ਅਰੁਣ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਜੀਵਨ ਹਸਪਤਾਲ ਗਿਆ।
After:  [PERSON] [LOCATION] ਵਿੱਚ [ORGANIZATION] ਗਿਆ।
```

### Tamil — `ta`

- Model: `env:OPENMED_INDIC_NER_MODEL` · locale `ta_IN`

```text
Before: அருண் சென்னையில் காவேரி மருத்துவமனை சென்றார்.
After:  [PERSON] [LOCATION] [ORGANIZATION] சென்றார்.
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

```text
Before: Пациент Иван Петров, СНИЛС 112-233-445 95
After:  Пациент [NAME], СНИЛС [ID]
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

- Model placeholder: `OpenMed/privacy-filter-multilingual` · locale `zh_CN`

```text
Before: 患者王芳，电话 13800138000
After:  患者[NAME]，电话 [PHONE]
```

The default entry is an API-compatible fallback. Supply a validated Chinese
PII model explicitly for production detection; the segmentation and exact
offset guarantees do not imply dedicated Chinese model weights.

### isiZulu — `zu`

- Model: `OpenMed/privacy-filter-multilingual` · locale `zu_ZA`

```text
Before: Igama lesiguli: Nomcebo Dlamini. Inombolo kamazisi 8001015009087
After:  Igama lesiguli: [NAME]. Inombolo kamazisi [ID]
```
