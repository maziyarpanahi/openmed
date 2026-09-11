# Social-determinant evidence contract

`openmed.clinical.sdoh_evidence` provides a small, deterministic envelope for
social-determinant extraction results. It preserves provenance and uncertainty
boundaries so an SDOH signal cannot be mistaken for a diagnosis or an
automated eligibility decision.

The contract is local-first and contains no model, corpus, credential, or
network-loading path. The source document remains caller-owned. The envelope
stores only controlled labels and source offsets; it never stores an excerpt,
surface value, document identifier, or raw input text.

## Required fields

```python
from openmed.clinical.sdoh_evidence import SDOHEvidence

finding = SDOHEvidence(
    evidence_type="self_report",
    assertion="unknown",
    source_section="social_history",
    source_span=(12, 27),
    review_status="needs_review",
    determinant="housing_insecurity",
)
```

The required evidence type distinguishes a self report, proxy report, clinician
observation, structured record, inference, unknown provenance, or refused
provenance. `source_span` is a non-empty half-open character-offset pair. No
source excerpt is accepted by the contract.

`assertion` must be one of `present`, `absent`, `uncertain`, `unknown`, or
`refused`. `unknown` means the available evidence does not establish a state;
`refused` means the source explicitly declined to provide one. Neither value
may be treated as an affirmative finding. `review_status` is
`needs_review` by default for records built without an explicit approval state.

```python
assert finding.to_dict() == {
    "evidence_type": "self_report",
    "assertion": "unknown",
    "source_section": "social_history",
    "source_span": [12, 27],
    "review_status": "needs_review",
    "determinant": "housing_insecurity",
}
```

The serialized mapping is safe for an audit report: it has offsets and
controlled labels, but no source value. `to_json()` uses sorted keys and fixed
separators for stable output. `SDOHEvidence.from_dict()` ignores unknown
extension keys, including accidental `text`, `value`, `surface`, and
`excerpt` fields, and revalidates the required fields.

## Safe reports and upstream findings

Use `build_sdoh_evidence_report()` to validate and deterministically order a
batch. Reports carry the assistive-use disclaimer and serialize only validated
records. `evidence_from_sdoh_finding()` adapts the existing SHAC-style finding
shape while retaining only its safe category and span; a missing upstream
status becomes explicit `unknown`.

This contract is an evidence and review boundary, not a clinical decision
engine, compliance certification, diagnosis, or eligibility determination.
Use synthetic or public data in tests and examples. Restricted SHAC data is
not bundled or loaded by this module.
