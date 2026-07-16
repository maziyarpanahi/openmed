# Per-Language PII De-identification

OpenMed's PII detection and de-identification are multilingual. The list of
supported language codes is the single source of truth in
[`openmed.core.pii_i18n.SUPPORTED_LANGUAGES`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/pii_i18n.py),
and each code wires up:

- a **default PII model** from `DEFAULT_PII_MODELS`, used when you pass `lang=`
  without an explicit `model_name=`, and
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
    The table below lists **every** code in `SUPPORTED_LANGUAGES` together with
    its `DEFAULT_PII_MODELS` entry and its `LANG_TO_LOCALE` mapping.
    `tests/unit/test_docs_language_coherence.py` asserts this page matches the
    constants exactly, so a newly wired language fails the suite until it is
    documented here.

## Supported languages

| Code   | Language   | Default PII model                                          | Faker locale | Notes                                                        |
| ------ | ---------- | ---------------------------------------------------------- | ------------ | ----------------------------------------------------------- |
| `am`   | Amharic    | `OpenMed/privacy-filter-multilingual`                      | `am_ET`      | Ethiopic patterns; `en_KE` Faker approximation warns once.   |
| `ar`   | Arabic     | `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1`    | `ar_EG`      | Egypt is the most-populous Arabic locale; override per call. |
| `de`   | German     | `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1`    | `de_DE`      | Steuer-ID surrogates via `GermanSteuerIdProvider`.           |
| `en`   | English    | `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`           | `en_US`      | Default model splits names into `first_name`/`last_name`.    |
| `es`   | Spanish    | `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1`   | `es_ES`      | DNI/NIE checksum-aware surrogates.                           |
| `fr`   | French     | `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1`    | `fr_FR`      | NIR / INSEE surrogates via `fr_FR.ssn`.                      |
| `he`   | Hebrew     | `OpenMed/privacy-filter-multilingual`                      | `he_IL`      | Served by the multilingual privacy filter.                   |
| `hi`   | Hindi      | `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1`    | `hi_IN`      | Aadhaar (Verhoeff) surrogates.                               |
| `id`   | Indonesian | `OpenMed/privacy-filter-multilingual`                      | `id_ID`      | Served by the multilingual privacy filter; NIK-aware.        |
| `it`   | Italian    | `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1`   | `it_IT`      | Codice Fiscale surrogates via `it_IT.ssn`.                   |
| `ja`   | Japanese   | `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1`        | `ja_JP`      | Family-name-first `PERSON` spans.                            |
| `ko`   | Korean     | `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1`        | `ko_KR`      | Resident Registration Number (RRN) surrogates.               |
| `nl`   | Dutch      | `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1`    | `nl_NL`      | BSN (Elfproef) surrogates via `nl_NL.ssn`.                   |
| `pt`   | Portuguese | `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | `pt_PT`     | Pass `locale="pt_BR"` for CPF/CNPJ surrogates.               |
| `ro`   | Romanian   | `OpenMed/privacy-filter-multilingual`                      | `ro_RO`      | Served by the multilingual privacy filter; CNP-aware.        |
| `sw`   | Swahili    | `OpenMed/privacy-filter-multilingual`                      | `sw`         | Bilingual patterns with Kenya ID and Maisha-aware surrogates. |
| `te`   | Telugu     | `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1`   | `en_IN`      | No Faker Telugu locale — `en_IN` approximation (warns once). |
| `th`   | Thai       | `OpenMed/privacy-filter-multilingual`                      | `th_TH`      | Served by the multilingual privacy filter; Thai NID-aware.   |
| `tr`   | Turkish    | `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1`   | `tr_TR`      | TCKN surrogates.                                             |
| `xh`   | isiXhosa   | `OpenMed/privacy-filter-multilingual`                      | `xh_ZA`      | Nguni patterns; `zu_ZA` Faker approximation warns once.      |
| `zh`   | Chinese    | `OpenMed/privacy-filter-multilingual`                      | `zh_CN`      | Routing placeholder; no dedicated Chinese PII model yet.     |
| `zu`   | isiZulu    | `OpenMed/privacy-filter-multilingual`                      | `zu_ZA`      | Nguni patterns with checksum-valid South African ID support.  |

Chinese segmentation and Han-script routing are supported, but the `zh`
default remains an explicit multilingual placeholder rather than a claim that
a dedicated Chinese PII model has shipped. Codes outside this list (for example
`pl`, `lv`, `sk`, `ms`, `tl`, `da`, and `ur`) are **not** model-backed PII languages.
Several of them still have
validator-backed national-ID coverage
(`openmed.core.pii_i18n.NATIONAL_ID_ONLY_LANGUAGES`); see
[PII Anonymization](anonymization.md#clinical-id-checksums) for the ID providers.
Urdu uses the conceptual `ur_PK` locale for CNIC dispatch and Faker's installed
`en_PK` backend for general surrogate data, with a one-time approximation warning.

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
