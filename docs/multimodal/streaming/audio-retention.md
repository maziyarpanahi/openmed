# Local audio retention and destruction accounting

`AudioRetentionSession` owns bounded mutable audio buffers and closes them on finalization, cancellation, provider failure or expiry. It can also enroll completed application-owned audio files for verified removal. Every expected artifact must have a successful destruction receipt before the audit passes. Unknown external copies, missing enrollment, failed overwrites, changed files and unexplained directory entries prevent success.

The default path is local, in memory, and requires no model or network connection. The module never writes audio, temporary files, caches or checkpoints to disk. File enrollment is an explicit cleanup adapter for an application's existing artifacts; it does not authorize recording sensitive audio on disk. This Python lifecycle component does not add a Swift capture UI, clinical decisions or compliance certification.

## Ownership and the expected inventory

Declare a tuple of `ExpectedAudioArtifact` objects before capture, using opaque 32-character lowercase hexadecimal references. The immutable inventory, session reference and retention policy determine the receipt binding digest. References must never encode patient values. Three storage kinds are supported:

| Kind | Actual ownership and cleanup evidence |
| --- | --- |
| `BUFFER` | `capture` copies input into an owned `bytearray`. Cleanup overwrites that allocation, checks it contains only zeroes and releases the owner's reference. Existing views into that allocation see the overwrite. |
| `FILE` | `enroll_file` records the identity of a completed file in an explicitly supplied private directory. Cleanup verifies identity, quarantines the exact entry, verifies it again, unlinks it, syncs the directory and checks absence. |
| `EXTERNAL` | Declared caller, provider, device, cache or crash artifacts outside this component's ownership. They remain unaccounted and prevent audit success. A boolean callback or disappearing reference cannot certify their destruction. |

`capture` does not take ownership of the caller's original immutable bytes, device buffers or copies made by a provider. The application must inventory and manage those separately. Do not omit a known copy or mark it as a managed buffer just to obtain a passing audit. This component cannot discover arbitrary process memory, driver allocations, other directories, swap, core dumps or backups. A passing audit covers the declared managed inventory and the stated operations only; it does not prove that all audio everywhere was erased.

The [streaming ASR session](offline-asr-contract.md) tracks chunk counts and byte accounting; it does not own the original `AudioChunk.payload`. Draining its counters is not a destruction receipt. Use the retention owner at capture, route trusted local inference through `infer`, and coordinate both sessions' terminal events. Transcript [PHI withholding](phi-withholding-buffer.md) and [speaker-role review](ambient-speaker-roles.md) remain separate gates; audio cleanup does not de-identify a transcript or approve a note.

## Lifecycle, deadlines and inference

`capture` admits each expected buffer once. `advance` records actual host transitions such as buffering, inference or cache use without extending retention. `infer` passes a read-only view to a trusted local provider, records inference, releases the borrowed view on exit, and checks the deadline again before returning the provider's protected result. Provider exceptions are replaced with a fixed error outside the handler, without retaining their original exception context.

Cancellation or expiry overwrites owned storage even while a provider is using its view; that call's result is then discarded. Providers must tolerate cancellation and must not copy or retain audio without the application accounting for the extra artifact. This wrapper does not sandbox providers, prevent their side effects or redact their returned transcript. Keep callbacks local and never log protected input or output.

Every terminal reason uses the same cleanup path. The context manager finalizes on normal exit and uses failure cleanup when its body raises. Explicit `finish(SessionEnd.CANCELLED)` supports cancellation. `close()` cancels an open session, retries outstanding cleanup and releases the directory descriptor. Always use one of these lifecycle paths; garbage collection is not a retention policy.

The default limits are 64 expected artifacts, 16 MiB of managed live audio, a five-minute session lifetime and 4,096 operations. Configurable ceilings are 256 artifacts, 64 MiB, one hour and 16,384 operations. The deadline starts at construction, is monotonic and cannot be renewed by updates, retries, finalization or restart. Clock failure or rollback closes the session. A default daemon timer also enforces expiry during idle time while the process runs. Thread scheduling can delay delivery; this is not a real-time deadline guarantee.

For deterministic host event loops, inject a monotonic clock and explicitly select `automatic_expiry=False`. That choice transfers idle expiry scheduling to the host: it must call `expire()` at the deadline even without more audio events. Checks on later access alone do not enforce an idle deadline. Process termination also requires a host watchdog and restart cleanup; no in-process timer survives a crash. There is no post-session retention extension in this API.

## File cleanup boundary

File enrollment requires POSIX descriptor-relative operations. Windows and unsupported systems reject the file adapter; in-memory ownership remains available. The application supplies a dedicated directory owned by the current OS user with no group/other permissions. Its final path component cannot be a symlink. Filenames are derived exclusively from declared opaque references as `<artifact_ref>.audio`; arbitrary source paths are not accepted.

Only completed regular files owned by that user, with no group/other permissions and exactly one hard link, can be enrolled. Stop all file writers first. Enrollment records device, inode, size and modification time without reading audio content. Replacement, modification, symlinks, hard links, permission drift, missing entries or preexisting `<artifact_ref>.retiring` entries block deletion. Cleanup preserves unrelated files and never recursively deletes the directory. An unexpected file prevents a passing audit even if every enrolled file was removed.

The directory descriptor pins the original directory. The host must protect its ancestry and serialize access; no other process with the same OS identity may mutate it concurrently. Identity checks are conservative tamper detection, not a sandbox against a hostile same-user process racing an unlink. `owned_file_unlinked` means verified removal of that directory entry, not overwritten storage sectors or deletion of an already-open file descriptor, copy, snapshot or backup. Such surviving handles and copies belong in the external inventory.

## Receipts, retries and crash recovery

`audit().to_json()` contains the schema, opaque session reference, policy/inventory digest, terminal reason, controlled blocker codes and ordered destruction receipts. It contains no audio, source path, provider result, payload hash or free-form exception. `passed` is false while the session is open, if any expected artifact is unresolved, if an admission resource limit was exceeded, or if the enrolled directory has unexplained entries. A missing file is not accepted as proof that this owner removed it.

Failed overwrites retain the buffer for retry. Failed file removals retain identity and quarantine state. Calling `finish` again retries only unresolved artifacts and preserves previous receipts and the original terminal reason. Identical repeated failures do not grow the receipt history. Closing the directory while anything is unresolved permanently records that unresolved close; preserve the checkpoint before closing if later recovery is needed.

`checkpoint()` returns value-free JSON for the application's own integrity-protected durable journal. It writes nothing. Persist the initial expected inventory before capture and refreshed checkpoints after file enrollment and before inference. The host must coordinate checkpoint durability with its producers; a process crash before enrollment/checkpointing can leave an artifact unaccounted, and this implementation will not fabricate a successful receipt for it.

`AudioRetentionSession.recover(checkpoint, artifact_directory=...)` validates the complete bounded checkpoint and original directory identity before attempting cleanup. It creates an already-terminal crash-recovery session, never resumes inference or grants a new retention interval. Only original matching file entries, including checkpointed quarantine entries, can be removed. Pre-crash buffers and external artifacts remain unaccounted because restart cannot verify their overwrite. Files absent from the checkpoint or already missing remain unresolved. Malformed checkpoints do not trigger partial deletion. Checkpoints are not signatures; their integrity and access controls belong to the host.

## Synthetic example

```python
from openmed.multimodal.streaming.privacy import (
    ArtifactKind, AudioRetentionSession, AudioStage, ExpectedAudioArtifact,
)

artifact_ref = "2" * 32
with AudioRetentionSession(
    "1" * 32,
    (ExpectedAudioArtifact(artifact_ref, ArtifactKind.BUFFER),),
) as retention:
    retention.capture(artifact_ref, b"synthetic audio fixture")
    retention.advance(artifact_ref, AudioStage.BUFFERING)
    # A real local provider may return a protected transcript here.
    sample_count = retention.infer(artifact_ref, lambda audio: len(audio))
    assert sample_count > 0

report = retention.audit()
assert report.passed
assert report.receipts[0].action.value == "owned_buffer_overwritten"
assert "synthetic audio fixture" not in report.to_json()
```

The fixture's original bytes literal is deliberately outside the owner's cleanup boundary. Production applications must account for their real capture sources, provider copies and crash artifacts; the example demonstrates cleanup of the owned copy only.

## Validation

Run `.venv/bin/python -m pytest tests/unit/multimodal/streaming/privacy/test_audio_retention.py -q`. Synthetic tests inspect actual overwritten buffers and removed files, cancellation during inference, provider errors, deadlines and clock failures, resource limits, incomplete inventories, failed cleanup retries, file replacement and link attacks, unexpected entries, conservative restart recovery, malformed checkpoints and value-free offline reports. POSIX file tests are platform-scoped; portable buffer tests and unsupported-file-adapter rejection run on every supported Python platform.

## Mute identifiers in a retained utterance

Use `mute_retained_clip` from `openmed.multimodal.streaming.privacy.audio_muting` to create a second, verified buffer inside an existing retention session. Declare both source and destination as separate expected buffers before capture. The destination inherits the same deadline and cleanup inventory. No file or export is created. The source stays intact until its ordinary destruction.

Use one complete, already screened utterance per clip, in interleaved signed 16-bit little-endian PCM. WAV containers, compressed audio and unknown filter delay are not accepted as PCM lineage. The host must verify the codec, frame count, channel layout, stream origin and synchronization, bind the capture digest to that utterance, and establish that retention is permitted. Mislabeling a partial transcript as complete can leave identifiers outside its scope. Muting known spans does not prove that a detector found every identifier, and does not anonymize a speaker's voice.

At the trusted screening handoff, keep the `ReleasedSpeechSegment` from `WithholdingBuffer` and its `screened_segment_digest(release)` together with the capture's SHA-256. Pin those digests there, not from untrusted input later. Construct `RetainedClipAlignment` with them, the session reference, source and ASR rates, source frames/channels, and the stream-relative millisecond origin of target frame zero. Only declare `TimingEvidence.TOKENS` for reliable, synchronized timing. Each known identifier must be completely covered; a partial word uses the entire token. Mapping rounds outwards in the ASR and source frame domains, pads in source frames, and merges overlaps. Every channel in each reported half-open frame range is verified silent; samples outside those ranges are verified unchanged.

If only the bounds of the **complete utterance** are known reliably, select `TimingEvidence.SEGMENT` explicitly: a single identifier mutes the entire clip. `UNCERTAIN` refuses the whole operation. There is no automatic switch to a weaker timing source or partial output on a missing span.

```python
from openmed.multimodal.streaming.privacy.audio_muting import mute_retained_clip

# retention already owns source_ref; destination_ref was reserved in its inventory.
report = mute_retained_clip(
    retention, source_ref, destination_ref, alignment, release, padding_ms=30,
)
# Safe evidence contains frame ranges, counts and digests only.
evidence = report.to_json()
# Protected access uses retention.infer(destination_ref, trusted_local_consumer).
# retention.finish() accounts for both original and derived buffers.
```
