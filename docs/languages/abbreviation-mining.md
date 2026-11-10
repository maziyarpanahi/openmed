# Local abbreviation mining

`openmed.clinical.languages.abbreviation_mining.mine_abbreviations` mines local
parenthetical candidates such as `blood pressure (BP)` and `BP (blood pressure)`.
Matching initials produces review hypotheses, not validated clinical meanings.
Ambiguous abbreviations retain every distinct expansion for native adjudication.
The implementation bundles no terminology and performs no network calls.

Pass a bounded iterable of local documents and a private high-entropy
`fingerprint_key` of at least 32 bytes. Keep that key local; use different keys for
unrelated review contexts. Raw abbreviations and expansion candidates stay in the
returned objects for authorized local review and are excluded from their repr.
Only `candidate.to_dict()` is intended as the review-artifact export: it contains
keyed candidate/concordance fingerprints, a closed frequency band, and
`unreviewed` status. It never contains snippets or exact occurrence counts.
Do not serialize candidate objects generically or publish private term spellings.

Defaults bound 1,000 documents, one million total characters, 10,000 candidate
pairs and 80 characters of context on each side of a match. Oversize inputs raise
controlled errors, never silently truncate. The caller must supply Unicode
normalization consistently; the miner does not modify source offsets. Terms are
NFC/casefold canonicalized only after extraction. Exact initials, all-uppercase
2–10 letter short forms, and parenthetical syntax are intentionally conservative;
clinical semantics, unparenthesized shorthand and model-assisted expansion are
outside this initial deterministic contract.

```python
from openmed.clinical.languages.abbreviation_mining import mine_abbreviations

rows = mine_abbreviations(
    ["Blood pressure (BP) in a synthetic example."],
    fingerprint_key=b"synthetic-example-key-not-production" * 2,
)
assert rows[0].frequency_band == "once"
assert rows[0].to_dict()["review_status"] == "unreviewed"
```
