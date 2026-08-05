# Clinical Coreference Resolution

`resolve_coreference()` groups mentions that refer to the same clinical entity
in one document. The established span-native form accepts fully detected
`OpenMedSpan` records. The event-candidate form accepts PROBLEM, TEST, and
TREATMENT seeds plus `document_text` and detects repeated strings, definite noun
phrases, and simple neutral pronouns.

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
from openmed.clinical import link_medication_attributes, resolve_coreference
from openmed.clinical.exporters import to_fhir
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

## Privacy-safe Event Candidate Clustering

Use the keyword-only `document_text` form to detect and cluster clinical event
candidates. This mode is limited to PROBLEM-, TEST-, and TREATMENT-like spans;
it does not add PII coreference. Its sanitized result does not mutate event
frames; the span-native chains described below opt downstream extraction into
representative rewriting.

```python
from openmed.clinical import resolve_coreference

text = "Imaging:\nA synthetic lesion was noted.\nAssessment:\nThe lesion was stable."
seeds = [
    {
        "document_id": "synthetic-note",
        "text": "synthetic lesion",
        "start": 11,
        "end": 27,
        "label": "PROBLEM",
        "negation": "affirmed",
    }
]

result = resolve_coreference(
    seeds,
    document_text=text,
    document_id="synthetic-note",
    hash_secret="application-owned-secret",
)
cluster_ids = result.cluster_ids_by_offset()
cluster_ids[("synthetic-note", (11, 27))]
# "synthetic-note:entity:..."

result.to_dict()["clusters"][0]
# {
#   "cluster_id": "synthetic-note:entity:...",
#   "document_id": "synthetic-note",
#   "semantic_type": "problem",
#   "member_offsets": [[11, 27], [51, 61]],
#   "member_hashes": ["hmac-sha256:...", "hmac-sha256:..."],
#   "canonical_hash": "hmac-sha256:...",
#   "mention_count": 2,
#   "advisory": "...",
# }
```

Event cluster payloads contain cluster ids, document offsets, content hashes,
safe type metadata, counts, and the assistive-use advisory. They do not contain
raw or canonical mention text. Supply `hash_secret` to make the content hashes
HMAC-SHA256 values under an application-owned key. Without a key, the resolver
returns deterministic SHA-256 fingerprints.

Assertion polarity is a hard clustering boundary: an affirmed mention and a
negated mention cannot share a cluster, even when their normalized strings are
identical. Definite-NP and pronoun candidates inherit their antecedent identity
only for scoring; their local assertion context is evaluated separately.

### Synthetic Gold Metric

The committed `event_coref.jsonl` fixture is scored with B-cubed, a documented
proxy for the CoNLL clustering average. For each mention, precision is the
fraction of its predicted cluster that belongs to its gold cluster, while recall
is the fraction of its gold cluster recovered in the prediction. The metric
averages those per-mention values and reports their harmonic-mean F1. The
acceptance gate is B-cubed F1 >= 0.60. The fixture is wholly synthetic and covers
repeated strings, definite noun phrases, pronouns, cross-sentence chains,
cross-section chains, all three event families, and opposite-polarity mentions.

## Attaching Event Frames to Representatives

Medication-change and lab-trend extraction accept the document-local chains
through `coreference_chains=`. TREATMENT and TEST head roles are emitted at the
canonical representative offset while their selected local evidence remains in
role provenance. The lower-level `attach_coreference_representatives()` helper
applies the same behavior to PROBLEM-like roles in other event frames.

```python
from openmed.clinical import extract_medication_change_events

frames = extract_medication_change_events(
    text,
    event_mentions,
    coreference_chains=chains,
)
drug = frames[0].role_slots("drug")[0]

drug.cluster_id
# stable document-scoped chain id
drug.provenance["coreference"]["source_spans"]
# [{"start": ..., "end": ..., "text_hash": "hmac-sha256:..."}]
```

When repeated head slots in one frame belong to the same chain, they collapse
to one representative slot. Attribute roles remain attached once to that
representative. The added provenance stores only cluster ids, confidence,
offsets, and HMAC hashes; it does not copy mention surfaces into provenance.
Frame `value` fields retain their existing caller-visible behavior.

## Collapsing Downstream Relations and Exports

Medication relation linking and grounded FHIR export accept the same
document-local chains through `coreference_chains=`. Relation candidates still
use each local mention to score nearby dose, route, frequency, and duration
attributes. Emitted relation heads are then rewritten to the representative,
and groups with the same `chain_id` collapse to one medication entity.

```python
groups = link_medication_attributes(
    text,
    medication_and_attribute_spans,
    coreference_chains=chains,
)

bundle = to_fhir(
    grounded_medication_spans,
    document_id="example-note",
    coreference_chains=chains,
)
```

Each collapsed relation record retains `cluster_id`, its representative offset
and HMAC hash, and the offset and HMAC hash of every supporting mention. FHIR
resources carry the same fields in the
`clinical-coreference-evidence` extension. Supporting provenance never copies
mention surfaces; existing relation head values and FHIR CodeableConcept values
keep their caller-visible behavior.

This collapse is intentionally document-level. A `CoreferenceChain` must cover
mentions from one source document, and downstream calls apply it only within
that document's relation or export batch. Cross-document entity linkage is a
separate, later step: document-linking cluster ids must not be supplied as
mention chains, and this path neither infers nor collapses entities across
documents.

## Resolution Rules

Mentions are processed in document order, and a reference can link only to an
earlier compatible mention. This antecedent-only rule rejects cataphora such as
"It resolved before the rash was documented." A pronoun with no antecedent
remains a singleton chain and cannot act as the anchor for another unresolved
pronoun. Overlapping spans are not treated as successive mentions, and mixed
document ids are rejected because each resolution pass is single-document.

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
