# Temporality-aware clinical NLI pairs

Clinical natural-language inference can make a historical statement look like
evidence for a present claim when temporal qualifiers are dropped. The
temporality-aware pair builder keeps a normalized event interval and an
explicit temporal status on both the premise and hypothesis, then applies a
conservative compatibility gate before an NLI label is used.

## Build a pair

```python
from openmed.clinical import build_temporal_nli_pair

pair = build_temporal_nli_pair(
    {
        "text": "synthetic history of cough",
        "offset": [0, 27],
        "interval": "2024-01-01/2024-01-03",
        "temporality": "historical",
    },
    {
        "text": "synthetic current cough",
        "offset": [28, 52],
        "interval": "2026-01-01",
        "temporality": "recent",
    },
    predicted_label="entailment",
)

pair.temporal_compatibility  # "incompatible"
pair.label                    # "review_required"
pair.review_required         # True
```

`recent`, `historical`, `hypothetical`, `future`, and `unknown` are accepted
status values. Common aliases such as `current`, `past`, and `conditional`
normalize to the same controlled vocabulary. An interval may be a normalized
ISO date, an inclusive ISO date range, an existing timeline interval, or a
relative expression such as `3 days ago`. Relative expressions are resolved
only when `reference_time` is supplied:

```python
pair = build_temporal_nli_pair(
    "synthetic prior event",
    "synthetic prior event",
    premise_time="3 days ago",
    hypothesis_time="3 days ago",
    premise_temporality="historical",
    hypothesis_temporality="historical",
    reference_time="2026-06-15",
    predicted_label="entailment",
)
assert pair.premise_interval.value == "2026-06-12"
assert pair.label == "entailment"
```

When a side has an event mapping with no explicit status, the builder reuses
the local ConText temporality resolver for historical or hypothetical cues.
Absolute intervals can also infer `historical`, `recent`, or `future` when a
reference date is supplied. Otherwise `TemporalMetadata.from_value()` leaves
the status `unknown`; an unqualified event surface follows ConText's existing
`recent` default, and an unanchored interval remains review-required.

## Compatibility and review

The compatibility state is one of:

| State | Meaning | Effective entailment label |
| --- | --- | --- |
| `compatible` | Same non-conditional status and certain nominal interval overlap | The supplied label is retained |
| `incompatible` | Statuses differ, a side is hypothetical, or intervals are disjoint | `review_required` |
| `unresolved` | A status, endpoint, or relative anchor is missing; uncertainty bounds overlap without nominal overlap | `review_required` |

An `entailment` returned by a local NLI backend is therefore never accepted
when the temporal evidence is incompatible or unresolved. The original model
label remains available as `predicted_label`, so a reviewer can see what was
gated without losing the model result. Contradiction, neutral, and abstention
labels are preserved in `predicted_label`; the temporal review flag still
applies.

`compare_temporal_metadata` returns the controlled state and a value-free
reason (`interval_overlap`, `status_mismatch`, `intervals_disjoint`, and so
on). `TemporalComparison` also exposes `nominal_overlap` and
`possible_overlap` for uncertainty-aware review.

## Privacy and local-first behavior

`to_model_input()` and `to_text_pair()` are the explicit hand-off to a caller's
local NLI backend and retain the source strings in memory. `to_dict()`,
`to_audit_dict()`, `to_json()`, and `repr(pair)` contain hashes, lengths,
offsets, normalized temporal metadata, and controlled review reasons; they do
not contain premise or hypothesis text. Pair identifiers are stable SHA-256
fingerprints. Invalid input errors intentionally omit supplied values.

Construction is deterministic and has no mandatory network, model, clock, or
filesystem call. The output is an assistive review artifact and does not make
diagnoses, treatment recommendations, or autonomous clinical decisions.
