# Social-history sections

`generate_social_history_examples` builds fictional "Social History" sections
with gold social-determinant (SDOH) events whose offsets point at the
evidence clause in the text. Templates are written in the repository;
restricted SHAC records are never read, bundled, or used. See the
[overview](index.md) for shared conventions.

## Generate sections

```python
from openmed.training.synthetic import generate_social_history_examples

sections = generate_social_history_examples(2, seed=1)
section = sections[0]
print(section.text)
for event in section.events:
    start, end = event.span
    print(event.category, event.value, event.status, "->", section.text[start:end])
```

Each section covers all five categories in `SOCIAL_HISTORY_CATEGORIES`
(`alcohol`, `drug`, `tobacco`, `employment`, `living_status`) in a seeded
order. Every three consecutive sections rotate through current, historical,
and negative or absent states for each category, so a batch of three or more
is balanced. `count` must be a non-negative integer and `seed` an integer;
booleans and other types raise `ValueError`.

## Events and records

An `SDOHEvent` has `category`, `value`, `status`, optional `amount`,
`temporality` (`recent` or `historical`), and a half-open `span`:

```python
print(section.events[0].to_dict())
# {'category': 'alcohol', 'value': 'alcohol', 'status': 'current',
#  'attributes': {'extent': '2 drinks/week', 'temporality': 'recent'},
#  'span': [16, 45]}
```

- `SDOHEvent.to_finding()` converts an event to the runtime `SDOHFinding`
  used by the SDOH extractor, so gold events and model output can be compared
  with the same type.
- `SyntheticSocialHistory.to_dict()` returns a JSON-compatible record with
  `text`, `events`, and a `metadata` block:

```python
print(section.to_dict()["metadata"])
# {'synthetic': True, 'source': 'openmed.synthetic.social_history', 'restricted_data': False}
```

`SYNTHETIC_SOCIAL_HISTORY_SOURCE` is that `source` string; use it to filter
these records out of mixed data.

## Seeds and determinism

The seed controls template choice and clause order, with no other input:

```python
assert generate_social_history_examples(5, seed=1) == generate_social_history_examples(5, seed=1)
```

The default is `seed=0`.

## Known limits

- A small, fixed template set: a few phrasings per category and state.
  Models trained only on it will not see free-form social histories.
- One clause per category per section, always all five, always one per line.
- English only.
- Events are gold *by construction*; they are not clinician-adjudicated and
  not clinical ground truth.
