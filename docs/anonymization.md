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
    "DOB 01/15/2020",
    method="shift_dates",
    date_shift_days=30,
)
print(result.deidentified_text)
# DOB [date]
```

The intent is for every date in a document to shift by the same offset, so
durations between dates (e.g. "3 days after admission") stay correct. With
the default English model, however, dates currently get masked instead of
shifted — the model's raw label for dates is lowercase `date`, but the
redaction code only shifts entities labeled exactly `DATE`. Tracked in
#408.

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

## Privacy-filter family

OpenMed ships three privacy-filter families, all **the same OpenAI
Privacy Filter architecture** (gpt-oss-style sparse-MoE transformer with
local attention, sink tokens, RoPE+YaRN, tiktoken `o200k_base`), differing
only in their training data:

The per-language PII API uses `openmed.core.pii_i18n.SUPPORTED_LANGUAGES`
as its source of truth and supports **13 supported PII language codes**:
`ar`, `de`, `en`, `es`, `fr`, `he`, `hi`, `it`, `ja`, `nl`, `pt`, `te`, and `tr`.
The multilingual privacy-filter family is a checkpoint family; it does not
expand the per-language API allow-list.

| Variant                              | Trained on                                      | PyTorch artifact                         | MLX (full)                                      | MLX (8-bit)                                           |
| ------------------------------------ | ----------------------------------------------- | ---------------------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| OpenAI Privacy Filter                | OpenAI's PII training set                       | `openai/privacy-filter`                  | `OpenMed/privacy-filter-mlx`                    | `OpenMed/privacy-filter-mlx-8bit`                     |
| OpenAI Nemotron Privacy Filter       | Nemotron PII dataset                            | `OpenMed/privacy-filter-nemotron`        | `OpenMed/privacy-filter-nemotron-mlx`           | `OpenMed/privacy-filter-nemotron-mlx-8bit`            |
| OpenMed Multilingual Privacy Filter  | OpenMed multilingual PII corpus; same 13-code API allow-list | `OpenMed/privacy-filter-multilingual`    | `OpenMed/privacy-filter-multilingual-mlx`       | `OpenMed/privacy-filter-multilingual-mlx-8bit`        |

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
