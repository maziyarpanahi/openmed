# Serial measurement trends

`openmed.clinical.trend` groups repeated measurements of the same entity, such
as a tumor diameter or a laboratory analyte, orders them by resolved timepoint,
converts them to a common unit and derives a descriptive direction:
`increasing`, `decreasing`, `stable`, `mixed` or `unknown`. A direction is a
summary for clinician review, not a clinical judgment; every trend carries
`TREND_ADVISORY`.

## Building trends

```python
from openmed.clinical import extract_measurement_trends

trends = extract_measurement_trends(
    [
        {"entity": "Tumor diameter", "value": "1.2 cm", "timepoint": "2026-01-05"},
        {"entity": "Tumor diameter", "value": 15, "unit": "mm", "timepoint": "2026-02-02"},
        {"entity": "tumor diameter", "value": 18, "unit": "mm", "timepoint": "2026-03-01"},
    ]
)
trends[0]["direction"]  # "increasing"
```

Entities are grouped case- and whitespace-insensitively. Points whose unit
cannot be compared with the rest of the group are listed in
`incomparable_points` rather than dropped. A series that cannot be totally
ordered by timepoint is `unknown`, with `first_value`, `last_value` and `delta`
set to `None`.

## Stable JSON for review artifacts

`serialize_measurement_trends` turns trends into compact JSON that does not
depend on the order the measurements arrived in, so it can be committed as a
regression fixture or attached to a review.

```python
from openmed.clinical import serialize_measurement_trends

text = serialize_measurement_trends(trends)
```

The same measurements in any order produce byte-identical output:

- trends are sorted by their case- and whitespace-insensitive entity key;
- a trend's `entity` label is the lowest spelling found among its points;
- ordered `points` keep their chronological order, while unordered `points`
  and all `incomparable_points` are sorted by their public fields;
- object keys are sorted and separators are compact (`,` and `:`).

Points that share one timepoint and normalize to the same value are ordered by
that value and their own fields rather than by input position. Without this,
`14 mm` and `1.4 cm` on one date (`0.014` and `0.013999999999999999` in metres)
would move `last_value` and `delta` with the input order.

Serialization changes no classification and no value. It raises `ValueError`
when a derived number (`delta`, `first_value`, `last_value` or a point's
`canonical_magnitude`) is NaN or infinite; the message names the field, not the
value.

Golden outputs for ordered, mixed, unknown and incomparable trends are kept in
`tests/fixtures/clinical/measurement_trend_serialized.jsonl`.
