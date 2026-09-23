# OMOP vocabulary write gates

Mappings are only valid relative to the vocabulary release that produced
them. `openmed.interop.omop.vocabulary_write_gate` compares mapping provenance
with the local target snapshot before an approved OMOP mutation batch reaches
its committer.

The gate is deterministic and offline. It does not download Athena content,
call a terminology service, inspect patient rows, or bundle a restricted
vocabulary. The deployment supplies the minimum terminology metadata from its
locally governed snapshot.

## Describe the target snapshot

Provide release versions by vocabulary and only the concept state needed for
the proposed mappings:

```python
from openmed.interop.omop import (
    VocabularyConcept,
    VocabularySnapshot,
    VocabularyWriteGate,
)

snapshot = VocabularySnapshot(
    {"SNOMED": "site-release-2026-09"},
    (
        VocabularyConcept(
            concept_id=123456,
            vocabulary_id="SNOMED",
            standard_concept="S",
        ),
    ),
)
gate = VocabularyWriteGate(snapshot)
```

`VocabularyConcept.invalid_reason` marks a concept as retired. An active
concept must still be present in the expected vocabulary and have OMOP
`standard_concept = "S"` to be write-compatible.

Snapshot construction is local policy. Exporting the needed columns from a
site-owned Athena installation is appropriate; committing those vocabulary
assets to OpenMed is not.

## Extract mapping provenance

The gate accepts only target concept, target vocabulary, and release metadata.
It has no fields for source codes, descriptions, patient identifiers, or row
values:

```python
from openmed.interop.omop import VocabularyMappingProvenance

provenance = VocabularyMappingProvenance(
    target_concept_id=123456,
    target_vocabulary_id="SNOMED",
    vocabulary_version="site-release-2026-09",
)
```

For a `SourceToConceptMapping` returned by `VocabularyRouter`, use
`VocabularyMappingProvenance.from_mapping(mapping)`. The extractor deliberately
ignores source codes and descriptions.

## Classify before writing

```python
report = gate.evaluate((provenance,))
if not report.is_compatible:
    send_for_review(report.remapping_queue)
```

Each mapping receives one classification:

| Classification | Meaning |
| --- | --- |
| `compatible` | The target contains the same active standard concept under the exact mapped vocabulary release. |
| `remap_required` | Provenance is missing, a release changed, the concept is absent or non-standard, or its vocabulary changed. |
| `retired` | The target snapshot marks the mapped concept invalid. |

Any `remap_required` or `retired` result is material drift. The report contains
closed reason codes, ordinals, vocabulary identifiers, and digests. It omits
concept identifiers and raw version strings. The remapping queue can therefore
be retained for review without copying clinical source values. Digests are
still sensitive metadata and need the same access controls and retention
policy as other audit artifacts.

## Enforce the gate at commit

Use the gate's `commit` wrapper with an approved staged mutation batch:

```python
result = gate.commit(
    batch,
    local_committer,
    approval=approval,
    mappings=(provenance,),
)
```

The wrapper evaluates the current target snapshot immediately before invoking
`OmopMutationBatch.commit`. Material drift raises
`VocabularyWriteGateError("incompatible_vocabulary_snapshot")`; the committer
is not called. The exception's `report` supplies the value-free queue for
review.

The gate does not remap automatically. A reviewer must resolve the queued
mapping against a governed local vocabulary, create new provenance, preview
and approve the resulting batch, and run the gate again. This control does not
certify the deployment or authorize autonomous clinical decisions.
