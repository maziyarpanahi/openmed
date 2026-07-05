# Medication Sig Parser

Medication sigs pack dose, dose-form, route, frequency, duration, and a PRN
condition into terse strings ("1 tab PO BID x7 days", "take 2 puffs q4h PRN").
`openmed.clinical.sig_parser` turns them into a structured `Sig`, the value
layer beneath medication grounding, reconciliation, and FHIR Dosage export.

Frequency and duration are not re-implemented here: the parser isolates those
candidates and delegates to the shipped `medication_sig.normalize_frequency` and
`normalize_duration` helpers, which own that lexicon. This module adds what they
deliberately exclude -- dose, dose-form, route, and the PRN condition.

## Parsing a sig

```python
from openmed.clinical import parse_sig

parse_sig("1 tab PO BID x7 days")
# {'raw': '1 tab PO BID x7 days', 'dose': 1.0, 'unit': 'tablet',
#  'form': 'tablet', 'route': 'oral', 'frequency_per_day': 2.0,
#  'as_needed': False, 'condition': None, 'duration_days': 7, 'missing': []}
```

The `Sig` mapping has these fields:

| Field | Meaning |
|---|---|
| `raw` | The input sig text. |
| `dose` | The numeric dose amount, or `None`. |
| `unit` | The dose unit (`mg`, `mcg`, `ml`, ...) or the form when the count is a form word. |
| `form` | The controlled dose-form (`tablet`, `capsule`, `puff`, `drop`, ...) or `None`. |
| `route` | The controlled route (`oral`, `intravenous`, `subcutaneous`, ...) or `None`. |
| `frequency_per_day` | Scheduled rate per day, from `normalize_frequency`, or `None`. |
| `as_needed` | `True` when the sig is PRN. |
| `condition` | The PRN reason ("PRN pain" -> `pain`), or `None`. |
| `duration_days` | Duration in days, from `normalize_duration`, or `None`. |
| `missing` | Components not found (`dose` / `route` / `frequency`). |

## Route and form controlled sets

Routes normalize from common abbreviations and phrases: `PO` / `by mouth` ->
`oral`, `IV` -> `intravenous`, `IM` -> `intramuscular`, `SC` / `SQ` ->
`subcutaneous`, `SL` -> `sublingual`, `PR` -> `rectal`, `INH` / `nebulized` ->
`inhaled`, and so on. Dose-form words (`tab`, `cap`, `puff`, `gtt`, ...) map to a
controlled form.

## Partial sigs

Malformed or partial sigs parse what is present and flag the rest:

```python
parse_sig("PO daily")["missing"]   # ['dose']
```

## Span-attached variant

`parse_sigs(text, spans)` parses the sig covered by each medication span and
returns the span offset alongside the parsed `Sig`, so results map back to the
source document.

```python
from openmed.clinical import parse_sigs

text = "ibuprofen 1 tab PO q6h PRN pain"
spans = [{"start": 10, "end": len(text), "label": "MEDICATION"}]
parse_sigs(text, spans)
# [{'span': (10, 31), 'sig': {... 'as_needed': True, 'condition': 'pain' ...}}]
```

## Notes

Parsing is deterministic and offline. Sig parsing is support tooling and is not
a substitute for clinician review; validate before any clinical use.
