# Terminology conflict resolution

Terminology adapters can return different codes for the same clinical mention.
`openmed.clinical.terminology` provides a local, deterministic reconciliation
step so that a selected code is never an unexplained consequence of adapter
iteration order.

The resolver accepts the existing grounding `Candidate` record, a
`TerminologyCandidate`, or a mapping with equivalent fields. It performs no
catalog lookup and makes no network call.

## Selection policy

Candidates are compared in this explicit order. A later rule is considered
only when all earlier rules tie:

1. configured source priority, with a larger value winning;
2. terminology version, with the newer numeric/lexical version winning;
3. configured exactness, where `exact` outranks `synonym`/`alias`, which
   outrank `fuzzy`/`partial`;
4. candidate score, with a larger finite score winning;
5. a stable SHA-256 candidate identity, used only as the final tie-break.

Unknown sources receive a priority below every configured source. If a source
priority is not supplied, all sources tie at the source rule and the remaining
rules still produce deterministic output. Version priorities may optionally
pin named releases; otherwise versions such as `2025.10` and `2025.2` are
compared numerically by their components.

Candidates with the same `(system, code)` identity are deduplicated first.
The losing records are retained under the `duplicate` category. Other losing
records are classified as `lower_source_priority`, `older_version`,
`less_exact`, `lower_score`, or `stable_tiebreak`.

## Privacy-safe provenance

The resolver does not accept a query string. Displays, synonyms, matched
aliases, and arbitrary metadata remain available on the in-memory candidate
but are excluded from `ConflictResolution.to_dict()`. Serialized selected and
discarded provenance contains only terminology identifiers, source/version
metadata, scoring fields, and a stable candidate identifier. Exceptions use
field-level messages and do not interpolate candidate values.

```python
from openmed.clinical.terminology import (
    TerminologyCandidate,
    TerminologyConflictResolver,
)

resolver = TerminologyConflictResolver(
    {"curated": 30, "local": 20, "legacy": 10},
    version_priority={"2025.02": 3, "2025.01": 2},
)

result = resolver.resolve(
    (
        TerminologyCandidate(
            system="SYNTHETIC",
            code="SYN-100",
            source="curated",
            version="2025.01",
            exactness="exact",
            score=0.91,
            display="Synthetic display",
        ),
        TerminologyCandidate(
            system="SYNTHETIC",
            code="SYN-200",
            source="local",
            version="2025.02",
            exactness="exact",
            score=0.99,
            display="Synthetic alternative display",
        ),
    )
)

safe_report = result.to_dict()
selected_code = result.selected_provenance.code if result.selected_provenance else None
discarded = result.discarded_by_category
```

`selected_code` is `SYN-100` because the configured source priority is applied
before version and score. The result also exposes every discarded category,
including empty buckets, which keeps downstream audit schemas stable. The
output is assistive terminology reconciliation and requires human verification
before clinical use.
