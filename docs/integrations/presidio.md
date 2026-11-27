# Presidio RecognizerResult Adapter

OpenMed's Presidio adapter converts Presidio `RecognizerResult` objects to
canonical `PIIEntity` objects, converts entities back, and merges both sets of
spans. The adapter keeps the Presidio dependency optional; mapping results to
OpenMed does not require installing Presidio.

Install the extra only when you want `from_canonical` to create Presidio's
default `RecognizerResult` class:

```bash
pip install "openmed[presidio]"
```

## Functions and configuration

- `to_canonical(result, *, text=None, config=None)` accepts one result or an
  iterable of results. Each result needs `entity_type`, `start`, and `end`;
  it may also supply `score`, `text`, `recognition_metadata`, and
  `recognizer_name`. If result text is absent, `text` supplies the source
  string for the span surface.
- `from_canonical(entities, *, result_cls=None, config=None)` converts a
  sequence of `PIIEntity` values back to result objects. Pass `result_cls` to
  use a compatible class without the optional dependency; otherwise the
  installed Presidio extra provides `RecognizerResult`.
- `merge_with_openmed(openmed_entities, presidio_results, *, text,
  config=None, use_semantic_patterns=True)` converts the Presidio results,
  combines them with OpenMed entities, and resolves overlaps. The original
  source string is required for span alignment and merging.

`PresidioAdapterConfig` has three fields: `source="presidio"` labels adapter
metadata, `preserve_presidio_labels=True` retains an input Presidio label when
converting back, and `allow_semantic_only_matches=False` controls semantic-only
matches during merging. `use_semantic_patterns` on `merge_with_openmed`
controls semantic pattern matching for that call.

## Label mappings

`to_canonical` uses this Presidio-to-OpenMed mapping. An unlisted Presidio
label is normalized and retained as its canonical label.

| Presidio label | OpenMed label |
|---|---|
| `PERSON` | `PERSON` |
| `PHONE_NUMBER` | `PHONE` |
| `EMAIL_ADDRESS` | `EMAIL` |
| `LOCATION` | `LOCATION` |
| `DATE_TIME` | `DATE` |
| `IP_ADDRESS` | `IP_ADDRESS` |
| `URL` | `URL` |
| `US_SSN` | `SSN` |
| `CREDIT_CARD` | `CREDIT_CARD` |
| `CRYPTO` | `BITCOIN_ADDRESS` |
| `IBAN_CODE` | `IBAN` |
| `US_DRIVER_LICENSE` | `ID_NUM` |
| `US_PASSPORT` | `ID_NUM` |
| `US_BANK_NUMBER` | `ACCOUNT_NUMBER` |
| `MEDICAL_LICENSE` | `ID_NUM` |

When converting back, OpenMed uses this canonical-to-Presidio mapping unless
the entity metadata contains an original Presidio label and
`preserve_presidio_labels=True`.

| OpenMed label | Presidio label |
|---|---|
| `PERSON` | `PERSON` |
| `FIRST_NAME` | `PERSON` |
| `LAST_NAME` | `PERSON` |
| `MIDDLE_NAME` | `PERSON` |
| `EMAIL` | `EMAIL_ADDRESS` |
| `PHONE` | `PHONE_NUMBER` |
| `LOCATION` | `LOCATION` |
| `STREET_ADDRESS` | `LOCATION` |
| `ZIPCODE` | `LOCATION` |
| `DATE` | `DATE_TIME` |
| `DATE_OF_BIRTH` | `DATE_TIME` |
| `TIME` | `DATE_TIME` |
| `SSN` | `US_SSN` |
| `CREDIT_CARD` | `CREDIT_CARD` |
| `IBAN` | `IBAN_CODE` |
| `IP_ADDRESS` | `IP_ADDRESS` |
| `URL` | `URL` |
| `ACCOUNT_NUMBER` | `US_BANK_NUMBER` |
| `ID_NUM` | `MEDICAL_LICENSE` |

## Synthetic example

The example uses a reserved `.test` address and an in-memory result mapping.
It makes no network calls and does not need Presidio installed.

```python
from openmed.interop.presidio import to_canonical

text = "Contact synthetic@example.test today."
start = text.index("synthetic@example.test")
result = {
    "entity_type": "EMAIL_ADDRESS",
    "start": start,
    "end": start + len("synthetic@example.test"),
    "score": 0.99,
}

entity = to_canonical(result, text=text)[0]
print(entity.label)  # EMAIL
print(entity.metadata["presidio_entity_type"])  # EMAIL_ADDRESS
```

The adapter records the source label in metadata, along with the configured
adapter and source names. `from_canonical` can use that original label on a
round trip when `preserve_presidio_labels` remains enabled.

## Merge behavior and privacy

`merge_with_openmed` copies the supplied OpenMed entities, converts the
Presidio results, then passes both sets through OpenMed's entity merger. The
merger uses text offsets, scores, labels, and optional semantic patterns to
select non-overlapping spans. The function returns canonical `PIIEntity`
objects; input entities are not mutated.

Use synthetic text for examples and tests. For real clinical text, keep source
text and results in your approved local environment and follow your
organization's data policy. The adapter does not send text to a service.

Offline adapter tests use synthetic result objects and cover label conversion,
round trips, and merging in `tests/unit/interop/test_presidio_adapter.py`.
