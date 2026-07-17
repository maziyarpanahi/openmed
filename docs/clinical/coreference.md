# Clinical Coreference Resolution

`resolve_coreference()` groups `OpenMedSpan` mentions that refer to the same
clinical entity in one document. It handles repeated named mentions, nominal
references such as "the lesion" or "that medication", personal and neutral
pronouns, and patient anchors such as "the patient".

!!! warning "Assistive annotations only"
    Coreference chains are deterministic heuristics for review and downstream
    organization. They must not automatically trigger a diagnosis, treatment,
    medication change, or other clinical decision.

The resolver is rules-first and fully local. It uses section agreement,
sentence distance, head-noun agreement, and entity-type compatibility. It does
not call a model, an LLM, or a network service. Source surfaces are used only
in memory to compare mentions; the returned index contains document ids and
offsets, and the resolver does not log raw mention text.

## Resolving Spans

Pass the source text and its `OpenMedSpan` mentions. Pronoun or generic nominal
mentions can use `canonical_label="OTHER"`; informative nouns such as
"medication" and personal pronouns provide a conservative type hint.

```python
from openmed.clinical import resolve_coreference
from openmed.core.schemas import OpenMedSpan, hmac_text_hash

text = "A left lung lesion was found. The lesion is stable. It is unchanged."


def span(surface: str, label: str, occurrence: int = 0) -> OpenMedSpan:
    start = -1
    cursor = 0
    for _ in range(occurrence + 1):
        start = text.index(surface, cursor)
        cursor = start + len(surface)
    return OpenMedSpan(
        doc_id="example-note",
        start=start,
        end=start + len(surface),
        text_hash=hmac_text_hash(surface, "application-owned-secret"),
        entity_type=label.casefold(),
        canonical_label=label,
        section="Assessment",
    )


spans = [
    span("left lung lesion", "CONDITION"),
    span("The lesion", "CONDITION"),
    span("It", "OTHER"),
]

chains, span_to_chain = resolve_coreference(spans, text)
chain = chains[0]

[(member.start, member.end) for member in chain.members]
# [(2, 18), (30, 40), (52, 54)]

chain.representative.start, chain.representative.end
# (2, 18)

span_to_chain[(spans[2].doc_id, (spans[2].start, spans[2].end))]
# stable chain id
```

Each `CoreferenceChain` contains:

| Field | Meaning |
| --- | --- |
| `chain_id` | Stable document-scoped id derived from member offsets and labels. |
| `members` | Original `OpenMedSpan` objects in document order. |
| `representative` | Most informative non-anaphoric member. |
| `confidence` | Mean deterministic link confidence, from `0.0` to `1.0`. |
| `advisory` | Clinical-review disclaimer. |

The `span_to_chain` index maps `(doc_id, (start, end))` to `chain_id`, so review
interfaces can recover a chain without storing a raw mention surface.

## Resolution Rules

Mentions are processed in document order, and a reference can link only to an
earlier compatible mention. This antecedent-only rule rejects cataphora such as
"It resolved before the rash was documented." A pronoun with no antecedent
remains a singleton chain.

For compatible antecedents, the resolver combines:

1. head-noun or canonical lexical agreement;
2. `canonical_label` and `entity_type` compatibility;
3. matching clinical sections; and
4. sentence and character distance.

Pronouns are limited to nearby antecedents. Nominals such as "the medication"
use their head noun to avoid linking to a nearer but incompatible condition.

## Experiencer Boundary

Patient, family, and other experiencers are hard boundaries: mentions with
different experiencers never share a chain. The resolver reads explicit
`metadata["experiencer"]` first, uses Family History as a section prior, and
otherwise applies the local cue-based experiencer resolver. This keeps a
relative's diabetes separate from a patient's diabetes even when the surface
forms are identical.
