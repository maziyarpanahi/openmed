# Terminology resolution cascades

OpenMed provides a deterministic, offline-first terminology cascade for
turning source codings or clinical terms into versioned vocabulary mappings.
The cascade makes uncertainty visible: a query is `mapped`, `ambiguous`,
`unmapped`, or `rejected`. Only `mapped` is returned as a successful
`StoreResult`; every other state carries its reviewable result as a typed
non-success outcome.

This layer complements candidate generation and conflict resolution. Candidate
generation finds possibilities. The cascade establishes which method ran,
which vocabulary snapshot was used, whether a unique mapping exists, and which
records must enter a review queue.

## Resolution order

The resolver stops at the first stage that returns candidates:

1. an exact code already in the target vocabulary;
2. a caller-supplied source-code crosswalk;
3. a normalized preferred-term match;
4. a normalized synonym or language-alias match;
5. a caller-supplied hierarchy or relationship rule;
6. an optional local semantic candidate provider.

This ordering prevents a lower-confidence semantic suggestion from replacing
an explicit coding or exact lexical mapping. Multiple candidates at the active
stage produce `ambiguous` unless an opted-in semantic provider has a unique
leader above the configured confidence margin. No generic success state hides
an empty candidate set.

```python
from openmed.clinical.grounding import VocabConcept, VocabularyIndex
from openmed.clinical.terminology import (
    TerminologyQuery,
    TerminologyRelationshipRule,
    TerminologyResolver,
    TerminologySnapshot,
)

index = VocabularyIndex(
    "loinc",
    [
        VocabConcept(
            system="loinc",
            code="SYN-100",
            preferred_term="Synthetic preferred term",
            synonyms=("Synthetic alias",),
        )
    ],
)
snapshot = TerminologySnapshot(
    vocabulary="local-loinc",
    version="2026.1",
    index=index,
)
resolver = TerminologyResolver(
    snapshot,
    hmac_secret=b"application-private-key-material",
    relationships=(
        TerminologyRelationshipRule(
            source_term="Synthetic child term",
            target_code="SYN-100",
            relationship="narrower",
        ),
    ),
)

outcome = resolver.resolve(TerminologyQuery(source_value="Synthetic alias"))
if outcome.ok:
    selected = outcome.value.selected_candidate
else:
    reviewable_result = outcome.value
```

The HMAC key must be private and stable for the scope in which mapping history
needs to be joined. Changing it deliberately breaks that linkage.

## Public mapping contract

`TerminologyMappingResult` includes:

- target vocabulary, version, system, content hash, and snapshot digest;
- ranked coded candidates with confidence, mapping rule, relationship, and
  candidate identity;
- selected candidate identity only for `mapped` results;
- policy identity, version, digest, and semantic-enabled flag;
- the explicit mapping state, reason code, and review requirement;
- an HMAC digest of the source query and optional source coding.

The source lexical surface, matched alias, and target display are deliberately
absent. `repr`, JSON, SQLite history, queue rows, and coverage summaries do not
contain them. Public JSON Schemas are bundled for mapping results, review
items, and coverage summaries. Version `1.0.0` uses the `same_major`
compatibility policy: additive fields can be introduced within major version
1, while breaking changes require a new major version.

## Visible review queue and append-only history

`SQLiteTerminologyMappingStore` writes immutable mapping results and creates a
queue item for every `ambiguous`, `unmapped`, or `rejected` result. Repeating
the same write is idempotent. Reusing a mapping identity with different bytes
is a conflict. Payload hashes are verified on read, the database is created
with owner-only permissions, and symbolic-link database paths are rejected.

```python
from openmed.clinical.terminology import SQLiteTerminologyMappingStore

with SQLiteTerminologyMappingStore("private/terminology.db") as store:
    stored = store.record(reviewable_result)
    queue = store.review_queue(states={"ambiguous", "unmapped"})
    history = store.history(reviewable_result.source_digest)
```

A vocabulary version or content change produces a different snapshot and
mapping identity. The previous record remains in history; it is never
overwritten. Rejected reviewer decisions use `result.reject(reason_code)` and
are appended as their own mapping and queue records.

## Optional semantic providers

Semantic fallback is disabled by default. Enabling it does not install,
download, or call a model. The caller must supply a provider that implements
`LocalSemanticCandidateProvider` and explicitly declares `local_only = True`.
The provider is invoked only after every deterministic stage misses. Missing
providers, provider failures, low-confidence candidates, and candidate ties
remain explicit outcomes.

This contract can wrap an encoder, a compact reranker, or a local vector index
without coupling the terminology layer to a particular runtime. Networked
retrieval belongs in a separately authorized adapter and must not masquerade
as the local provider.

## Coverage reporting

`TerminologyCoverageSummary.from_results()` returns only aggregate counts and
rates for mapped, ambiguous, unmapped, and rejected states. It is suitable for
evaluation artifacts and release gates because it does not retain source
digests, codes, candidate identities, or raw values. Teams should track both
mapping coverage and the ambiguous/unmapped rates; a high aggregate coverage
number must not erase important review workload.

Terminology mappings are assistive records. They require validation for the
target vocabulary release and intended clinical workflow, and they must not be
treated as autonomous clinical decisions.
