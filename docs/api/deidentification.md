# De-identification API

The `deidentify()` helper returns a typed `DeidentificationResult` with the
redacted text, detected `PIIEntity` spans, result metadata, and a `to_dict()`
representation for JSON-style workflows.

## Entity Table Export

Use `DeidentificationResult.to_dataframe()` when you want to inspect the
entities detected in a single de-identification result as a table:

```python
from openmed import deidentify

result = deidentify("Patient John Doe called 555-1234", method="mask")
entities = result.to_dataframe()
```

The method imports pandas lazily, so importing `openmed` or
`openmed.core.pii` does not import pandas. If pandas is not installed, calling
`to_dataframe()` raises an actionable `ImportError` with the install command.

The returned DataFrame has one row per detected entity and always uses this
column order:

| Column | Description |
| ------ | ----------- |
| `text` | Entity text span captured by the detector. |
| `label` | Detector label for the span. |
| `entity_type` | Normalized PII entity type stored on the result entity. |
| `start` | Character start offset in the original text. |
| `end` | Character end offset in the original text. |
| `confidence` | Detector confidence score. |
| `action` | De-identification policy action applied to the entity, when available. |
| `result_id` | Stable hash-derived identifier shared by every row from the result. |

When no PII entities are present, `to_dataframe()` returns an empty DataFrame
with the same columns.

## India ABDM and ABHA identifiers

Enable the ABDM recognizer bundle for Indian clinical records to detect and
replace ABHA numbers and addresses, Aadhaar numbers, PAN values, and contextual
HPR/HFR registry identifiers:

```python
from openmed import deidentify

result = deidentify(
    note,
    method="replace",
    lang="en",
    locale="en_IN",
    abdm=True,
)
```

The bundle also activates automatically for Hindi or Telugu, an India locale,
or the `india_dpdp_act` policy profile. Pass `abdm=False` to opt out explicitly.
It is inactive for other languages and locales by default. Recognized source
labels (`ABHA_NUMBER`, `ABHA_ADDRESS`, `AADHAAR`, `PAN`, `ABDM_HPR_ID`, and
`ABDM_HFR_ID`) normalize to `ID_NUM` and the `DIRECT_IDENTIFIER` policy class.
Replacement mode produces synthetic values only; it does not call ABDM, verify
an identifier, or store or resolve a real ABHA number.
ABHA numbers are validated by their publicly documented 14-digit shape; the
public NHA materials do not specify an offline checksum algorithm.

ABHA-linked record sharing requires the individual's informed consent. This
mode de-identifies recognized identifiers, but it does not collect or record
consent and does not decide whether a disclosure is permitted. Applications
must enforce their own consent and disclosure workflow before sharing records.
See the [official ABDM FAQ](https://abdm.gov.in/FAQ) for the ABHA identity and
consent model.
