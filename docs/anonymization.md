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

This document focuses on `replace`, which was upgraded in the
`feature/obfuscation-pii` line to a full Faker-backed obfuscation engine.

## The new `replace` engine

`method="replace"` no longer picks from a small hardcoded list. It builds
a per-document `Anonymizer` (see [`openmed.core.anonymizer`](../openmed/core/anonymizer))
that delegates to [Faker](https://faker.readthedocs.io/) with custom
providers for clinical IDs.

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
[`openmed.core.pii_i18n`](../openmed/core/pii_i18n.py) so every surrogate
ID passes the same validator that detection uses:

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

The MLX-only artifacts `OpenMed/privacy-filter-mlx` and
`OpenMed/privacy-filter-mlx-8bit` route through `extract_pii()` /
`deidentify()` like any other model:

```python
extract_pii(text, model_name="OpenMed/privacy-filter-mlx-8bit")
deidentify(text, model_name="OpenMed/privacy-filter-mlx-8bit",
           method="replace", consistent=True, seed=42)
```

On Apple Silicon with MLX importable, this runs the MLX pipeline.
Elsewhere, the call substitutes `openai/privacy-filter` via
`transformers` and emits a one-time `UserWarning` explaining the swap.
Either way the output entity dicts have the same shape so the rest of
the pipeline behaves identically. Smart-merging (regex-based span
construction) is skipped for this family — the model already does
Viterbi-constrained BIOES decoding internally.

The dispatch lives in
[`openmed.core.backends.create_privacy_filter_pipeline`](../openmed/core/backends.py).

## Backwards compatibility

`replace` outputs are no longer drawn from the prior hardcoded list
(`["Jane Smith", "John Doe", "Alex Johnson", "Sam Taylor"]`, etc.).
Any test that asserted on those exact strings must either:

- pass `consistent=True, seed=<value>` and update the expected output, or
- assert non-equality with the original instead of equality with a
  hardcoded surrogate.

Other methods (`mask`, `remove`, `hash`, `shift_dates`) are unchanged.
