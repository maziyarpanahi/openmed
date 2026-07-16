# Per-language i18n golden de-identification fixtures

Each `*.jsonl` file in this directory is a **synthetic, no-DUA** clinical note
that exercises one language pack's own date / phone / national-ID / address /
postcode patterns and surrogate locale. A new language pack can pass generic
English checks while silently mis-detecting a native ID or shifting an offset;
these fixtures fail loudly when that happens.

`tests/unit/eval/test_i18n_golden_fixtures.py` runs
`get_patterns_for_language()` over every fixture and asserts each gold span is
recovered at its exact offset with the correct canonical label, and that each
national-ID passes its language's checksum validator where one exists.

## Schema

One JSON object per line (currently one note per language). This is the same
shape the shared golden loader (`openmed/eval/golden/loader.py`) validates, so
fixtures here are also picked up by the harness.

```json
{
  "id": "golden-i18n-<lang>-clinical-pii",
  "language": "<ISO 639-1 code>",
  "text": "<synthetic clinical note>",
  "gold_spans": [
    {
      "start": 0,
      "end": 10,
      "label": "DATE",
      "text": "<exact substring text[start:end]>"
    },
    {
      "start": 20,
      "end": 33,
      "label": "ID_NUM",
      "text": "<national id>",
      "metadata": {
        "checksum_status": "valid",
        "identifier_type": "<cnp|cpf|nir|...>"
      }
    }
  ],
  "metadata": {
    "category": "multilingual",
    "expected_output": {
      "method": "mask",
      "text": "<note with each span replaced by [LABEL]>"
    },
    "identifier_type": "<national id type>",
    "locale": "<expected surrogate Faker locale, e.g. fr_FR>",
    "synthetic": true
  }
}
```

### Field notes

- **`gold_spans[].label`** must be a canonical label. The five exercised
  families use `DATE`, `PHONE`, `ID_NUM`, `STREET_ADDRESS`, `ZIPCODE`.
- **`gold_spans[].start`/`end`** are half-open character offsets into `text`;
  `text[start:end]` must equal `text`.
- **`ID_NUM` `metadata.checksum_status`** is `"valid"` when the language has a
  national-ID checksum validator and the value passes it, or `"unvalidated"`
  for languages whose national-ID pattern is regex-only (e.g. `ar`, `ja`).
- **`metadata.locale`** records the expected surrogate Faker locale.
- **`metadata.synthetic`** must be `true`. Never use real patient data.

## Authoring checklist for a new language pack

When you add a language pack, add `openmed/eval/golden/fixtures/i18n/<lang>.jsonl`
in the same shape:

1. Write one synthetic note that contains **at least one** span of each family:
   a date, a phone number, a national ID, a street address, and a postcode,
   using invented values only.
2. Make the national-ID a **checksum-valid** value — generate it with the
   language's surrogate provider / validator rather than typing digits by hand,
   and set `checksum_status` accordingly (`unvalidated` only if the pack has no
   national-ID validator).
3. Compute every span offset from the text programmatically; do not hand-count.
   `text[start:end]` must equal the span `text`.
4. Confirm every gold span is recovered by `get_patterns_for_language(<lang>)`
   at its exact offset — a preceding capitalized cue word can be absorbed into a
   street match, and some phone regexes exclude the `+CC` prefix, so verify the
   recovered value equals the gold value.
5. Keep `metadata.category = "multilingual"`, `metadata.synthetic = true`, and a
   masked `expected_output.text`.
6. Run `.venv/bin/python -m pytest tests/unit/eval/test_i18n_golden_fixtures.py`.
