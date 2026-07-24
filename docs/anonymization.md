# PII Anonymization

OpenMed's `deidentify()` API supports five redaction methods for detected
PII entities:

| Method        | Output                                  | Use when                                 |
| ------------- | --------------------------------------- | ---------------------------------------- |
| `mask`        | `[NAME]`, `[EMAIL]`, …                  | You want clear placeholders.             |
| `remove`      | `""` (deleted)                          | You don't need positional alignment.     |
| `replace`     | Locale-aware fake surrogates            | You need realistic-looking text.         |
| `hash`        | `NAME_a1b2c3d4` (entity-typed digest)   | You need consistent linking across docs. |
| `shift_dates` | Dates only — shifted by N days          | You want to preserve relative time.      |

This document focuses on `replace`, which was upgraded in v1.3.0 to a full
Faker-backed obfuscation engine. If you just want to compare all five
methods side by side, start with the quickstart below.

## Quickstart: choosing a method

### `mask` — clear placeholders

```python
from openmed import deidentify

result = deidentify(
    "Patient John Doe (DOB: 01/15/1970) called from 555-1234",
    method="mask",
)
print(result.deidentified_text)
# Patient [first_name] [last_name] (DOB: [date]) called from [phone_number]
```

Placeholder names come from the model's own entity labels, so they vary by
model (the default `OpenMed-PII-SuperClinical-Small-44M-v1` model used here
splits names into `first_name`/`last_name` rather than a single `NAME`).

Not reversible by itself — pass `keep_mapping=True` and use `reidentify()`
(see below) if you need to restore the original text later.

### `remove` — delete PII entirely

```python
result = deidentify("Call 555-1234", method="remove")
print(repr(result.deidentified_text))
# 'Call '
```

Use this when you don't need positional alignment with the original text
(e.g. exporting de-identified text for search indexing).

### `replace` — realistic fake surrogates

```python
result = deidentify(
    "Email: test@example.com",
    method="replace",
    consistent=True,
    seed=42,
)
print(result.deidentified_text)
# Email: asnyder@example.com
```

Best for sharing data with downstream tools that expect well-formed values
(e.g. an email field that should still look like an email). See
[The new `replace` engine](#the-new-replace-engine) below for locale and
determinism options.

### `hash` — consistent, irreversible digests

```python
result = deidentify("Patient John Doe", method="hash")
print(result.deidentified_text)
# Patient first_name_a8cfcd74 last_name_fd53ef83
```

The same input always hashes to the same digest, so repeated mentions of
the same value link together across documents — without storing the
original anywhere.

### `shift_dates` — preserve intervals, hide absolute dates

```python
result = deidentify(
    "Admit 01/15/2020, follow-up 01/25/2020",
    method="shift_dates",
    date_shift_days=30,
)
print(result.deidentified_text)
# Admit 02/14/2020, follow-up 02/24/2020
```

The intent is for every date in a document to shift by the same offset, so
durations between dates (e.g. "3 days after admission") stay correct.
`date_shift_days=30` is a fixed offset when no patient key is supplied.

For longitudinal research, pass a stable `patient_key` so every document for
that patient receives the same HMAC-derived offset across sessions:

```python
patient_token = load_patient_key_from_vault()
hmac_key_material = load_date_shift_hmac_key()

shared_kwargs = {
    "method": "shift_dates",
    "patient_key": patient_token,
    "date_shift_max_days": 365,
    "date_shift_secret": hmac_key_material,
}

first = deidentify("Visit 01/15/2020", **shared_kwargs)
second = deidentify("Visit 03/15/2020", **shared_kwargs)
```

Equal patient keys and the same secret yield identical offsets, preserving
intervals across documents. Different patient keys generally produce different
offsets within `date_shift_max_days`. The raw patient key is used only as HMAC
input and is not returned in shifted text, mappings, logs, or audit artifacts.
Patient-keyed offsets require caller-supplied `date_shift_secret`; do not use
PHI as that key material.
If `patient_key` is supplied with the older `date_shift_days` option, that
value is treated as the maximum absolute offset bound; prefer
`date_shift_max_days` for new code.

### Reversing a de-identification: `reidentify()`

Pass `keep_mapping=True` to get back a `mapping` you can hand to
`reidentify()` later:

```python
from openmed import deidentify, reidentify

text = "Dr. Alice Smith met Bob Jones today"
result = deidentify(text, method="mask", keep_mapping=True)
print(result.deidentified_text)
# Dr. [first_name] [last_name] met [first_name_2] [last_name_2] today

restored = reidentify(result.deidentified_text, result.mapping)
assert restored == text
```

Repeated entities of the same type (two `first_name`s above) get a numbered
placeholder (`[first_name]`, `[first_name_2]`, ...) so each one maps back to
its own original value — this was a known limitation (#204) fixed by #222;
`reidentify()` now round-trips correctly even when a type repeats.

## Custom deny-list and allow-list recognizer

Use `custom_recognizer` when your site has identifiers the model does not
know, or benign values that must never be redacted. The argument accepts a
plain mapping, a `CustomRecognizer` instance, or a `.json`/`.yaml` path.

```python
from openmed import deidentify, extract_pii

custom_recognizer = {
    "case_sensitive": False,
    "deny": {
        "terms": [
            {"term": "Ward Phoenix", "label": "LOCATION"},
        ],
        "patterns": [
            {"pattern": r"\bSTUDY-\d+\b", "label": "ID_NUM"},
        ],
    },
    "allow": {
        "terms": ["Mercy Trial"],
        "patterns": [r"\bPUBLIC-\d+\b"],
    },
}

entities = extract_pii(text, custom_recognizer=custom_recognizer)
result = deidentify(text, method="mask", custom_recognizer=custom_recognizer)
```

Deny-list terms are literal strings. Deny-list patterns are regular
expressions. Each deny entry needs a `label`; OpenMed keeps that label on the
returned entity and normalizes it into the canonical label taxonomy for
policy and audit handling. Matches are emitted with `custom:deny`
provenance.

Allow-list terms and patterns suppress any overlapping span from any detector,
including model detections, deterministic rules, and custom deny-list matches.
Allow-list precedence always wins over deny-list and model detections, so an
allowed value is left untouched in `deidentify()` output.

Recognizer metadata stores hashes and rule ids, not raw matched surfaces. In
the staged pipeline, custom matching runs on normalized text and spans are
remapped back to original offsets before redaction.

## The new `replace` engine

`method="replace"` no longer picks from a small hardcoded list. It builds
a per-document `Anonymizer` (see
[`openmed.core.anonymizer`](https://github.com/maziyarpanahi/openmed/tree/master/openmed/core/anonymizer))
that delegates to [Faker](https://faker.readthedocs.io/) with custom providers
for clinical IDs.

```python
from openmed import deidentify

deidentify(
    "Paciente Pedro Almeida, CPF: 123.456.789-09",
    method="replace",
    lang="pt",
    locale="pt_BR",          # default for pt is pt_PT; override per call
    consistent=True,         # same input -> same surrogate within doc
    seed=42,                 # cross-run reproducibility
)
```

### Locale resolution

`lang` (an ISO 639-1 code OpenMed uses everywhere) maps to a Faker
locale via `LANG_TO_LOCALE`:

| OpenMed `lang` | Faker locale | Notes                                                    |
| -------------- | ------------ | -------------------------------------------------------- |
| `en`           | `en_US`      |                                                          |
| `fr`           | `fr_FR`      |                                                          |
| `de`           | `de_DE`      |                                                          |
| `it`           | `it_IT`      |                                                          |
| `es`           | `es_ES`      |                                                          |
| `nl`           | `nl_NL`      |                                                          |
| `hi`           | `hi_IN`      |                                                          |
| `te`           | `en_IN`      | Faker has no Telugu locale — emits a one-time warning.   |
| `pt`           | `pt_PT`      | Override with `locale="pt_BR"` for Brazilian Portuguese. |

Pass `locale=` explicitly to override per call (e.g. `pt_BR` to generate
CPF/CNPJ surrogates instead of Portuguese NIF/VAT).

Country-aware African French and Portuguese surrogates are available through
conceptual locale overrides. `fr_SN`, `fr_CI`, and `fr_CM` use curated names,
cities, addresses, and country-code phone formats while keeping the French PII
model; `pt_MZ` and `pt_AO` do the same with the Portuguese model. Unsupported
Faker methods delegate to `fr_FR` or `pt_PT`, so these overrides do not alter
the default `fr` and `pt` output. Arabic also accepts `ar-DZ` and `ar-MA`; if
the installed Faker release lacks the requested regional backend, OpenMed
falls back to `ar_EG` and emits the same one-time warning used by the other
Arabic regional overrides.

For the full per-language table — every `SUPPORTED_LANGUAGES` code with its
default PII model, Faker locale, and a before/after example — see
[Per-Language De-identification](languages.md).

### Determinism

Three modes:

- **Random** (default). Every call samples fresh surrogates. Good for
  audits or when you want visible variability.
- **`consistent=True`**. Same `(canonical_label, original_value)` pair
  resolves to the same surrogate within the call. "John Doe" appearing
  twice in one document gets one surrogate.
- **`seed=<int>`** (implies `consistent=True`). Same seed across runs
  produces the same surrogate stream — useful for snapshot tests and
  regression fixtures.

Determinism uses `hashlib.blake2b` over `(seed, canonical_label, original)`,
so different originals always get different surrogates.

### Cross-document surrogate vaults

Use a `SurrogateVault` when separate `deidentify(..., method="replace")`
calls need stable pseudonyms for the same identifier:

```python
from openmed import SurrogateVault, deidentify

vault = SurrogateVault.from_file(
    "surrogate-vault.json",
    hmac_secret="rotate-and-store-this-secret-outside-the-vault",
)

first = deidentify(
    "Patient John Doe was admitted.",
    method="replace",
    surrogate_vault=vault,
)
second = deidentify(
    "John Doe returned for follow-up.",
    method="replace",
    surrogate_vault=vault,
)
```

The vault file stores `(canonical_label, lang, HMAC text_hash) -> surrogate`
entries plus `schema_version` and `hmac_scheme`; it does not store raw source
surfaces or the HMAC secret. Treat the file as sensitive pseudonymous linkage
data anyway: it can connect records across documents even without plaintext.

For person names, the vault derives that HMAC input from a deterministic ISO
15919 pivot. Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu,
Kannada, Malayalam, and ISO 15919 inputs can therefore share one in-memory join
key and one surrogate without storing the romanized name. The same engine can
normalize explicit ITRANS and Harvard-Kyoto input before conversion:

```python
from openmed.processing import from_latin, to_latin, transliteration_key

result = to_latin("राम ராம rāma")
assert result.text == "rāma rāma rāma"
assert result.remap_span(0, 4) == (0, 3)
assert transliteration_key("राम") == transliteration_key("ராம")
assert transliteration_key("rAma", "ITRANS") == transliteration_key("rāma")
assert from_latin("lakShmI", "Devanagari", scheme="itrans") == "लक्ष्मी"
```

The default `schwa_policy="preserve"` and `anusvara_policy="marker"` form the
round-trip-safe subset. `schwa_policy="source"` applies northern word-final
schwa deletion, while `anusvara_policy="homorganic"` expands a nasal according
to the following consonant; both are intentionally lossy. The public
`LOSSY_CASES` tuple also lists nukta and extended letters, Tamil distinctions
that its orthography does not encode, Gurmukhi addak, and Malayalam chillu
letters. The stdlib-only tables are a clean-room Unicode/ISO implementation
interoperable with the conventions of Aksharamukha (AGPL-3.0) and the Indic NLP
Library (MIT). No code, data, copyleft component, neural weights, or third-party
mapping bundle from either project is included.

Perso-Arabic Urdu is intentionally an unsupported stub. The built-in API fails
closed with `ValueError`; deployments that need it must supply a separately
licensed, out-of-process adapter.

### Format preservation

Phone numbers, dates, and emails preserve the structure of the original:

```python
deidentify("Call (415) 555-1234", method="replace", consistent=True, seed=1)
#  -> "Call (XXX) XXX-XXXX"  (digit groups, separators, country code position)

deidentify("Born 01/15/1970", method="replace", consistent=True, seed=1)
#  -> "Born MM/DD/YYYY"      (separator and ordering kept)

deidentify("Email: john@hospital.org", method="replace", lang="en")
#  -> "Email: alice@hospital.org"  (domain kept, local part faked)
```

### Clinical ID checksums

OpenMed reuses the existing checksum validators in
[`openmed.core.pii_i18n`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/pii_i18n.py)
so every surrogate ID passes the same validator that detection uses:

| Locale  | ID type             | Provider                                               |
| ------- | ------------------- | ------------------------------------------------------ |
| `pt_BR` | CPF                 | Faker built-in (`pt_BR.cpf`)                           |
| `pt_BR` | CNPJ                | Faker built-in (`pt_BR.cnpj`)                          |
| `nl_NL` | BSN (Elfproef)      | Faker built-in (`nl_NL.ssn`)                           |
| `fr_FR` | NIR                 | Faker built-in (`fr_FR.ssn`)                           |
| `it_IT` | Codice Fiscale      | Faker built-in (`it_IT.ssn`)                           |
| `es_ES` | NIE                 | Faker built-in (`es_ES.nie`)                           |
| `en_IN` | Aadhaar (Verhoeff)  | OpenMed `AadhaarProvider` (Faker's built-in is invalid) |
| `de_DE` | Steuer-ID           | OpenMed `GermanSteuerIdProvider` (Faker's `de_DE.ssn` is US-style) |
| any     | NPI (Luhn over 80840) | OpenMed `NPIProvider`                                 |
| any     | Generic MRN         | OpenMed `MedicalRecordNumberProvider`                  |

### Extending

```python
from openmed import register_clinical_provider, register_label_generator
from faker.providers import BaseProvider

# Add your own checksum-bearing ID format
class HospitalAccountProvider(BaseProvider):
    def hospital_account(self):
        return f"HACC-{self.numerify('########')}"

register_clinical_provider(HospitalAccountProvider)

# Override the generator for a canonical label
def my_first_name(faker, original, *, locale):
    return faker.first_name() + "-test"

register_label_generator("FIRST_NAME", my_first_name)
```

## Chinese word segmentation

Chinese routing uses a pluggable word segmenter that preserves exact Python
string offsets. The default `jieba` backend and the small synthetic clinical
dictionary ship with OpenMed; no model download is required:

```python
from openmed.core.config import OpenMedConfig
from openmed.processing import create_chinese_segmenter_from_config

config = OpenMedConfig(
    chinese_segmentation_backend="jieba",
    chinese_user_dict_path="/srv/openmed/zh_terms.txt",
)
segmenter = create_chinese_segmenter_from_config(config)
tokens = segmenter.segment("患者王芳因心房颤动入院")

assert all(token.text == "患者王芳因心房颤动入院"[token.start:token.end]
           for token in tokens)
```

An additional dictionary is a UTF-8 text file with one entry per line in
`term`, `term frequency`, or `term frequency POS` form. For example:

```text
心脏超声 90000 nz
临床路径
```

Keep organization-specific dictionaries outside the package and point
`chinese_user_dict_path` or `OPENMED_CHINESE_USER_DICT` at the local file. This
avoids bundling private terminology or restricted lexicons.

The optional `pkuseg` (MIT) and HanLP (Apache-2.0) adapters are installed with
`openmed[zh-pkuseg]` and `openmed[zh-hanlp]`, respectively. Set
`chinese_segmentation_backend` or `OPENMED_CHINESE_SEGMENTATION_BACKEND` to
select one. `pkuseg` defaults to the `medicine` domain, configurable through
`chinese_pkuseg_domain`; HanLP accepts a preloaded tokenizer or a local model
path through `create_chinese_segmenter(..., hanlp_model=...)`. OpenMed never
downloads optional model files implicitly, so provision those assets before
selecting either backend.

## Privacy-filter family

OpenMed ships three privacy-filter families, all **the same OpenAI
Privacy Filter architecture** (gpt-oss-style sparse-MoE transformer with
local attention, sink tokens, RoPE+YaRN, tiktoken `o200k_base`), differing
only in their training data:

The per-language PII API uses `openmed.core.pii_i18n.SUPPORTED_LANGUAGES`
as its source of truth and supports **30 supported PII language codes**:
`am`, `ar`, `cs`, `da`, `de`, `el`, `en`, `es`, `fr`, `he`, `hi`, `id`, `it`,
`ja`, `ko`, `nl`, `no`, `pt`, `ro`, `ru`, `sv`, `sw`, `ta`, `te`, `th`, `tr`,
`uk`, `xh`, `zh`, and `zu`.
Russian and Chinese routing currently use documented multilingual
default-model placeholders; dedicated model weights are not bundled.
The optional Indic NER adapter adds eight user-configured routes (`as`, `bn`,
`gu`, `kn`, `ml`, `mr`, `or`, and `pa`) and can also serve Hindi, Tamil, and
Telugu. It loads only an explicit path or repository from
`OPENMED_INDIC_NER_MODEL` and has no bundled default checkpoint.
Additional validator-backed national-ID providers cover ID-only locales such as
Polish, Latvian, Slovak, Malay, Filipino, Finnish, and Urdu without adding
default PII models for those language codes. Urdu's conceptual `ur_PK` locale
uses Faker's installed `en_PK` backend for general surrogate data while CNIC
generation remains provider-backed and format-valid.
The multilingual privacy-filter family is a checkpoint family; it does not
expand the per-language API allow-list.

| Variant                              | Trained on                                      | PyTorch artifact                         | MLX (full)                                      | MLX (8-bit)                                           |
| ------------------------------------ | ----------------------------------------------- | ---------------------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| OpenAI Privacy Filter                | OpenAI's PII training set                       | `openai/privacy-filter`                  | `OpenMed/privacy-filter-mlx`                    | `OpenMed/privacy-filter-mlx-8bit`                     |
| OpenAI Nemotron Privacy Filter       | Nemotron PII dataset                            | `OpenMed/privacy-filter-nemotron`        | `OpenMed/privacy-filter-nemotron-mlx`           | `OpenMed/privacy-filter-nemotron-mlx-8bit`            |
| OpenMed Multilingual Privacy Filter  | OpenMed multilingual PII corpus; 28 model-backed codes plus the documented `ru`/`zh` routing placeholders | `OpenMed/privacy-filter-multilingual`    | `OpenMed/privacy-filter-multilingual-mlx`       | `OpenMed/privacy-filter-multilingual-mlx-8bit`        |

All run through the same `extract_pii()` / `deidentify()` API — only the
weights differ:

```python
extract_pii(text, model_name="OpenMed/privacy-filter-mlx-8bit")
extract_pii(text, model_name="OpenMed/privacy-filter-nemotron-mlx-8bit")
extract_pii(text, model_name="OpenMed/privacy-filter-multilingual-mlx-8bit")

deidentify(text, model_name="OpenMed/privacy-filter-nemotron",
           method="replace", consistent=True, seed=42)
deidentify(text, model_name="OpenMed/privacy-filter-multilingual",
           method="replace", consistent=True, seed=42)
```

**Backend selection.** On Apple Silicon with MLX importable, the MLX
artifact runs natively via `PrivacyFilterMLXPipeline`. Elsewhere, the
call substitutes the corresponding PyTorch model via `transformers` and
emits a one-time `UserWarning` explaining the swap. The fallback is
**family-aware** — an MLX-only Nemotron request on Linux substitutes
`OpenMed/privacy-filter-nemotron`, and an MLX-only multilingual request
substitutes `OpenMed/privacy-filter-multilingual`, so the user gets the same
training distribution they asked for.

Either way the output entity dicts have the same shape so the rest of
the pipeline behaves identically. Smart-merging (regex-based span
construction) is skipped for this family — the model already does
Viterbi-constrained BIOES decoding internally.

The dispatch lives in
[`openmed.core.backends.create_privacy_filter_pipeline`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/backends.py).
To register a new fine-tune that should fall back to its own PyTorch repo
on non-Mac hosts, add a row to `_TORCH_FALLBACK_BY_FAMILY` in that module.
If a fine-tune introduces a genuinely different *architecture* (not just
new weights), it would also need a new MLX model class and family branch
in `openmed.mlx.models.build_model` — but a same-architecture fine-tune
needs neither.

## Backwards compatibility

`replace` outputs are no longer drawn from the prior hardcoded list
(`["Jane Smith", "John Doe", "Alex Johnson", "Sam Taylor"]`, etc.).
Any test that asserted on those exact strings must either:

- pass `consistent=True, seed=<value>` and update the expected output, or
- assert non-equality with the original instead of equality with a
  hardcoded surrogate.

Other methods (`mask`, `remove`, `hash`, `shift_dates`) are unchanged.
