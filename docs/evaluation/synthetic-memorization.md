# Synthetic memorization audit

`openmed.eval.synthetic_memorization` audits synthetic generation output against
protected reference material before a dataset is published. Callers supply the
protected text themselves — for example, a clinician notes template, a targeted
evaluation slice, or a licensed corpus that must not be bundled with the
release. The audit answers one question per candidate: does any fragment of this
candidate reproduce, closely paraphrase, or expose a run of the protected
reference?

The module is local-first and value-free by construction:

- protected text is reduced to SHA-256 and n-gram fingerprints before any audit
  runs, and the raw text is never carried on a report, finding, error message,
  or fixture;
- no network call is performed, and the audit result depends only on its inputs;
- findings are ordered deterministically, so two runs over the same inputs
  produce byte-identical JSON.

## Signals

Each candidate is compared with every protected reference under three signals.

| Signal | What it detects | Detector | Default threshold |
| --- | --- | --- | --- |
| `exact` | verbatim fragment reuse | normalized fragment digest equality | — |
| `fuzzy` | close paraphrase of a fragment | Jaccard similarity of token shingles | `0.72` |
| `exposure` | long character n-gram overlap with the reference as a whole | covered fraction of the reference n-grams | `0.40` |

`exact` and `fuzzy` work at fragment granularity: the reference is split on
sentence and punctuation boundaries, and only fragments that are long enough
(24 characters by default) enter the fingerprint. `exposure` works on the whole
reference, so a candidate that samples many short runs from one protected note
cannot hide behind fragment boundaries.

## Fingerprinting protected references

```python
from openmed.eval.synthetic_memorization import (
    MemorizationSignal,
    SyntheticMemorizationPolicy,
    audit_synthetic_memorization,
    assert_release_allowed,
    fingerprint_reference,
)

policy = SyntheticMemorizationPolicy(
    fuzzy_similarity=0.8,
    exposure_match_ratio=0.35,
    blocking_signals=(
        MemorizationSignal.EXACT,
        MemorizationSignal.EXPOSURE,
        MemorizationSignal.FUZZY,
    ),
)

references = (
    fingerprint_reference(
        "protected-001",
        "Patient reports severe chest pain radiating to the left arm since this morning.",
        policy=policy,
    ),
)

report = audit_synthetic_memorization(
    {"candidate-001": generated_note},
    references,
    policy=policy,
)

if report.blocked:
    # Publishing this candidate would repeat protected clinical material.
    print(report.to_json())
else:
    assert_release_allowed(report)
```

`fingerprint_reference` rejects text shorter than `min_fragment_chars`, so a
caller cannot silently fingerprint a reference that fragments into nothing. A
`SyntheticMemorizationPolicy` is validated on construction: thresholds must sit
inside their documented ranges, `blocking_signals` must be sorted and free of
duplicates, and every bound has a hard ceiling.

## Release gate

`audit_synthetic_memorization` returns a `SyntheticMemorizationReport`. The
verdict is `blocked` when at least one retained finding belongs to a blocking
signal, and `clear` otherwise. `assert_release_allowed` raises
`SyntheticMemorizationError` for a blocked report; the message carries only the
finding count and the blocking signals, never the matched text or its digest.

Blank candidates are skipped rather than blocked and are counted in
`skipped_candidate_count`, so an empty generation slot cannot fail an otherwise
clean dataset.

## Determinism and evidence

The report exposes `signals`, `signal_counts`, `candidate_count`,
`reference_count`, `truncated`, and the ordered `findings`. Each finding carries
the candidate id, reference id, signal, reason code, fragment digest, the
character offset and length of the match inside the normalized candidate, and
the detector score. `to_dict()` and `to_json()` are canonical and round-trip
through `from_dict()` and `from_json()`.

`max_findings` (64 by default) bounds report size; a truncated report sets
`truncated` but keeps the verdict, so truncation never turns a blocked candidate
into a releasable one.

## Bounds

| Bound | Default | Maximum |
| --- | --- | --- |
| `min_fragment_chars` | 24 | 200 000 |
| `fuzzy_similarity` | 0.72 | 1.0 |
| `shingle_size` | 3 | 8 |
| `exposure_ngram_size` | 12 | 64 |
| `exposure_match_ratio` | 0.40 | 1.0 |
| `max_findings` | 64 | 10 000 |
| `max_candidate_fragments` | 256 | 10 000 |
| candidates per audit | — | 2 048 |
| references per audit | — | 512 |
| text length per document | — | 200 000 characters |

## Tests

```bash
python -m pytest tests/unit/eval/test_synthetic_memorization.py -q
```

The suite covers the three signals, fragment and n-gram fingerprints, policy and
payload validation, canonical JSON plus round-trip, report invariants, the
release gate, input bounds, and an offline check that fails the test if the
audit opens a socket.
