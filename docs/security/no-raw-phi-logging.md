# No-Raw-PHI Logging Policy

OpenMed code must not write raw protected health information (PHI), patient
text, source documents, or extracted cleartext spans to logs at any level.
Logs are operational telemetry only.

## Allowed log content

- Counts, durations, thresholds, model identifiers, backend names, and status
  transitions.
- Span metadata such as label, start offset, end offset, confidence bucket, and
  validity flags.
- A precomputed keyed `text_hash` when one already exists for the span or
  document. Do not create ad hoc hashes in log statements.
- Exception class names and high-level failure categories.

## Disallowed log content

- Input document text, cleaned text, truncated text snippets, prompts, sentences,
  or source surfaces that may contain clinical content.
- Entity text, original PII values, redacted-to-original mappings, or replacement
  mapping payloads.
- Exception messages that may include user input, source paths, request bodies,
  or downstream library payloads.
- File paths or item identifiers derived from patient, chart, or encounter data.

## Engineering requirements

- Prefer structured fields over formatted prose when adding logs.
- Use lengths, counts, labels, offsets, and safe identifiers instead of text.
- Keep request and response bodies out of service logs.
- When adding or changing a PII, de-identification, text-processing, or batch
  path, run the no-raw-PHI logging guard:

```bash
.venv/bin/python -m pytest tests/unit/test_no_raw_text_logging.py -q
```

The full suite must also pass before release or pull request review:

```bash
.venv/bin/python -m pytest tests/ -q
```

## Session-scoped voice embeddings

Voice embeddings are sensitive biometric identifiers. Python
`openmed.multimodal.voice_embeddings.VoiceEmbeddingSession` and OpenMedKit
`VoiceEmbeddingSession` own mutable embedding buffers behind opaque handles.
They perform no network calls and need no model assets or additional dependencies.
A handle exposes only same-session cosine similarity and opaque reference evidence.
It supplies no voice values, speaker identity, clinical role or cross-visit matching.
Persistence is always refused in this API; any future persistence or cross-visit
policy requires a separate explicit review and remains disabled here.

Python handles and session owners refuse pickling, copying and state serialization;
standard JSON encoding fails. `repr` and `to_evidence()` expose only an opaque
reference. Swift handles are not `Codable`, refuse `serialize()`, and implement
opaque description and reflection. Diagnostics contain only `handle_count` and
`handle_digests` (Swift: `handleCount` and `handleDigests`). Digests are computed
from fresh random reference bytes, never embedding values or encounter identifiers.
Similarity scores are results, not diagnostic or evidence fields. Results carry
`non_diagnostic_voice_similarity` and require explicit reviewer confirmation for
consequential downstream use; this API cannot authorize a clinical write.

```python
from openmed.multimodal.voice_embeddings import VoiceEmbeddingSession

session = VoiceEmbeddingSession()
left = session.add([1.0, 0.0])  # synthetic vectors only
right = session.add([0.0, 1.0])
assert left.similarity(right).score == 0.0
receipt = session.destroy("withdrawal")
assert receipt.destroyed_count == 2
```

The host **must** call `destroy` on pause, consent withdrawal, cancellation and
finalization (`destroy(.withdrawal)` in Swift). Each boundary overwrites and
releases all owned buffers, invalidates every handle, permanently closes the
owner and issues a value-free receipt: controlled reason code, destroyed count
and handle digests. Resumption needs a new owner and fresh vectors. Handles hold
weak owners, so retaining handles cannot retain biometric storage. Owner teardown
also erases storage as a fallback; explicit lifecycle calls provide receipts.

An optional injected `register(digest, erase)` hook registers each buffer with a
host retention controller. The idempotent callback erases that buffer and returns
an `embedding_destroyed` receipt; callbacks keep neither buffers nor owners alive.
The host still closes the owner at session boundaries. Registration failure erases
the just-created buffer and returns `embedding_registration_failed`, without the
underlying exception payload. This embedding-owned hook does not define or
implement the pending audio-retention, alias, diarization or consent contracts.

The injected allocator receives only a dimension count and transfers a fresh
zero-filled allocation to the owner (`array('d')` in Python;
`VoiceEmbeddingBuffer` in Swift). It must not retain memory views or mutate owned
storage. Tests use retained allocation references to inspect erasure and weak
references to verify release. Defaults bound a session to 256 handles of at most
4096 dimensions; explicit bounds can increase each limit up to 65536.

Stable refusal codes include `embedding_cross_session_refused`,
`embedding_destroyed`, `embedding_session_closed`,
`embedding_serialization_refused`, `embedding_persistence_refused`,
`embedding_vector_invalid`, `embedding_dimensions_mismatch`,
`embedding_capacity_exceeded`, `embedding_limits_invalid`,
`embedding_allocator_failed`, `embedding_handle_invalid` (Python-only direct
construction refusal) and `embedding_registration_failed`.

This is an API lifecycle boundary, not isolation from hostile in-process code,
heap inspection, swap, crash dumps or runtime copies. The caller/provider remains
responsible for its source vectors and other allocations; these cannot be erased
by the handle owner. Synthetic tests establish owned-buffer behavior only, not
provider qualification, clinical validation or legal compliance certification.
