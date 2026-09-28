# SDOH Negated-Need Resolution

SDOH extractors can surface the same determinant in an asserted need and in a
screening denial.  `openmed.clinical.sdoh_negated_need` resolves each caller
supplied finding in its own local sentence and clause so a denial for one
determinant does not bleed into a neighboring finding.

This is a deterministic, offline review aid.  It does not infer a need from a
missing mention, assign a Z-code, or make a clinical decision.

## Resolve findings

Pass the source text and findings with half-open `start`/`end` offsets. Existing
`SDOHFinding` values and mappings are accepted. The returned records are safe
metadata: they retain offsets and controlled labels, but not the finding value
or cue text.

```python
from openmed.clinical import resolve_sdoh_negated_needs

text = "No food insecurity, but reports a transportation barrier."
findings = [
    {
        "category": "food_insecurity",
        "start": text.index("food insecurity"),
        "end": text.index("food insecurity") + len("food insecurity"),
    },
    {
        "category": "transportation_barrier",
        "start": text.index("transportation barrier"),
        "end": text.index("transportation barrier")
        + len("transportation barrier"),
    },
]

results = resolve_sdoh_negated_needs(text, findings)
[(item.assertion, item.need_status) for item in results]
# [('negated', 'absent'), ('affirmed', 'present')]
```

Results are sorted by source offsets. `input_index` preserves the original
finding position when a caller needs to join the safe metadata to a protected
in-memory collection.

## Assertion contract

| `assertion` | `need_status` | Meaning | Review |
| --- | --- | --- | --- |
| `affirmed` | `present` | The candidate is an asserted need, or has no local polarity cue. | No, unless another cue is ambiguous. |
| `negated` | `absent` | An explicit local denial applies to this determinant. | No automatic clinical action. |
| `unknown` | `unknown` | The resolver abstained because local evidence is ambiguous or uncertain. | Yes |

The resolver keeps negation determinant-specific. Coordinated denials such as
`denies food insecurity and transportation barriers` can cover both findings;
an adversative or newly asserted clause such as `but reports transportation`
starts a separate scope. Backward cues such as `not present`, `absent`, and
`ruled out` are supported when they follow the finding.

Double negation is never reduced with even/odd parity. For example, `not no`
is returned as `assertion="unknown"`, `double_negation=True`, and
`review_required=True`. Contradictory local cues such as a denied finding that
is also explicitly present likewise become `unknown` and carry the controlled
`"contradictory_cues"` review reason.

Pseudo-negation and incomplete documentation are not treated as a denial.
Phrases such as `cannot be excluded`, `not ruled out`, `not documented`, and
`unknown` produce an unresolved record for review.

## Privacy and provenance

`SDOHNegatedNeedEvidence.to_dict()` and `.to_json()` contain only:

- source and cue offsets;
- canonical determinant and assertion labels;
- controlled review reasons and flags; and
- the caller input index.

They do not include `text`, `value`, raw cues, logs, or exception payloads.
The original finding remains caller-owned and should stay in the authorized
review workflow. The module uses only local regular-expression rules, performs
no mandatory network call, and does not load restricted SDOH datasets.

All bundled examples and tests use synthetic wording. The output is not a
compliance certification or an autonomous clinical decision guarantee; review
the source span before downstream coding or referral action.
