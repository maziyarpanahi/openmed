# Transcript revision lineage

`openmed.multimodal.streaming.transcript.RevisionLedger` tracks one stable segment through insert, delete, replace, timestamp-shift and finalize operations. It preserves unaffected token identities and remaps registered citations. If an edit changes cited text or makes a cited token ambiguous, the ledger invalidates that citation instead of presenting stale offsets as current evidence.

The ledger receives no transcript text, audio or clinical values. The caller retains protected text separately and applies the same operations to it. This Python contract supplies lineage for downstream privacy and review gates; it does not perform transcription, prove that the caller applied an edit, certify PHI safety or authorize clinical decisions. OpenMedKit parity is tracked separately in #3679.

## Register and remap a citation

```python
from openmed.multimodal.streaming.transcript import RevisionLedger, TranscriptToken

ledger = RevisionLedger(
    session_ref="a" * 32,
    segment_ref="b" * 32,
    length=8,
    tokens=(
        TranscriptToken(token_ref="c" * 32, start=0, end=3, start_ms=0, end_ms=100),
        TranscriptToken(token_ref="d" * 32, start=4, end=8, start_ms=100, end_ms=200),
    ),
)
reference = ledger.capture_span("e" * 32, 4, 8)

# Insert a two-code-point prefix into the caller's protected text as well.
ledger.insert(0, 2, (), expected_revision=0)
mapped = ledger.remap_span(reference)
assert mapped.valid and (mapped.start, mapped.end) == (6, 10)
assert mapped.reference.token_refs == ("d" * 32,)

ledger.shift_timestamps(("d" * 32,), 20, expected_revision=1)
assert ledger.remap_span(reference).start_ms == 120
ledger.finalize(expected_revision=2)
assert ledger.remap_span(reference).finalized
report = ledger.to_json()
```

The references above are synthetic. Session, segment, token and span references must be independently assigned opaque 128-bit lowercase hexadecimal values, never patient identifiers or encoded clinical text. Segment identity remains unchanged across revisions. Token references are never reused, including after deletion. Each operation advances the revision once; `expected_revision` rejects stale concurrent work. Serialize access to each ledger instance.

## Character and time coordinates

Character ranges are half-open Unicode code-point offsets, matching Python string indexing. They are not UTF-8 byte offsets or grapheme indices. Inserted/replacement token character ranges are relative to the new fragment. Existing unaffected tokens shift by the edit's character-length delta without changing identity. Whitespace or unrecognized fragments may have no tokens; a registered span must intersect at least one token, and gaps do not establish recognized content.

Token times are integer milliseconds relative to the source audio stream. They remain stream-relative even when the token's character offsets are fragment-local. Tokens must have nonoverlapping ordered character ranges and nondecreasing start and end times; acoustic intervals may overlap. A mapped span reports the envelope of its intersecting whole tokens. It does not infer sub-token acoustic boundaries.

## Editing and invalidation

| Operation | Effect on existing citations |
| --- | --- |
| Insert before or exactly at a span's start | Move both boundaries right. |
| Insert exactly at or after its end | Keep its character range. |
| Insert inside a span | Invalidate with `cited_text_changed`. |
| Delete or replace before a span | Move both boundaries by the length delta. |
| Delete or replace overlapping a span | Invalidate with `cited_text_changed`. |
| Edit another part of a token cited by a span | Retire that token and invalidate with `cited_token_changed`. |
| Timestamp shift | Preserve character offsets and token IDs; recompute the acoustic envelope. |
| Finalize | Preserve valid citations and seal further text/timing changes. |

Replacement fragments require fresh token references even when their length matches the removed text. Use minimal edit ranges when the adapter has reliable unchanged-token lineage. A whole-segment replacement intentionally invalidates prior citations; the ledger cannot guess semantic equivalence without text. Reinserted content never revives an invalidated citation. Downstream processing must capture new evidence after reassessment.

`remap_span()` accepts only an exact reference registered in that ledger. Altering its revision, session, segment or source coordinates fails closed. Invalid results retain the explicitly labeled original reference for audit, but their current character and time coordinates are `None`. Always check `valid` before using them. Finalization means the revision is sealed, not clinically verified or safe for disclosure. Later corrections need a separately managed correction flow rather than mutation of sealed evidence.

## Bounds and privacy

A ledger accepts up to 65,536 code points, 4,096 distinct token identities over its lifetime, 4,096 operations and 4,096 registered spans. Citation anchors and operation history each have a separate limit of 65,536 total token references, preventing many broad spans or repeated full-segment shifts from multiplying storage without bound. Token times are bounded to one day. Malformed coordinates, reused IDs, stale updates, invalid timestamp ordering and resource limits raise a fixed value-free `RevisionLedgerError`. Validation happens before state mutation; rejected updates do not retire tokens, consume revisions or invalidate citations.

JSON version `openmed.transcript.revisions.v1` includes opaque identities, offsets, relative times, controlled operation names and invalidation reasons. There are no text, audio, name, path or arbitrary diagnostic fields, and the implementation performs no logging, file access or network calls. These records are still linkable provenance metadata; the application must apply its access and retention policies. The ledger does not authenticate adapters or offer persistence/recovery guarantees.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/streaming/transcript/test_revision_ledger.py -q
```

Synthetic controls cover all five operations, composition, Unicode offsets, boundary affinity, partial-token edits, timestamp bounds, stale references, atomic failures, identity reuse, finalization and bounded history.
