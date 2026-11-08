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

## On-device container and log sentinel sweep

The policy above covers code OpenMed writes. A sentinel sweep checks what an
OpenMedKit run actually leaves on a device, including dependency caches such as
tokenizer, runtime and OCR files. A test plants synthetic sentinel values in the
inputs, runs the public kit APIs, then sweeps app-writable storage and the
process's own log for those values.

The sweep is test support and never ships. On Apple platforms it is the
`OpenMedPrivacySweep` SwiftPM target, which no package product exposes. On Android
it is `android/openmedkit/src/sharedTest/kotlin`, compiled only into the `test`
and `androidTest` source sets, so the release AAR does not change.

| Surface | Swift (`PHISentinelSweep`) | Kotlin (`PhiSentinelSweep`) |
| --- | --- | --- |
| Storage roots | `AppContainerRoots.current()`: `documents`, `application_support`, `caches`, `preferences`, other `library` children, `tmp`, optional `app_group` | `AndroidAppStorageRoots.of(context)`: `files`, `cache`, `code_cache`, `no_backup`, `shared_prefs`, `databases`, other `data_other` children, `external_files`, `external_cache` |
| Process log | `OSLogStoreProcessLogReader` (`OSLogStore(scope: .currentProcessIdentifier)`) | `LogcatProcessLogReader` (`logcat -d -v tag --pid=<pid>`) |
| Test seams | `SweepRoot`, `AppContainerRoots.roots(containerHome:temporaryDirectory:)`, `ProcessLogReader`, `CapturedLogReader`, injected clock | `SweepRoot`, `AndroidAppStorageRoots.roots(dataDirectory:)`, `ProcessLogReader`, `CapturedLogReader`, logcat launcher, injected clock |

Each sentinel has a controlled identifier (`[a-z0-9][a-z0-9._-]{0,63}`) and a
synthetic value of 8 to 256 UTF-8 bytes; `synthetic(identifier:)` makes a random
value that cannot collide with other runs. File names, file contents and log
entries are matched against the value's exact UTF-8 bytes, UTF-16LE, UTF-16BE,
standard and URL-safe base64 at all three byte alignments, and RFC 3986
percent-encoding (upper- or lowercase hex, `%20` or `+` for a space). Symbolic
links are counted, not followed. Configuration errors are value-free codes such as
`sentinel_length_out_of_range` and `sweep_bounds_invalid`.

Reports contain only sentinel identifiers, storage classes, file kinds (`plist`,
`json`, `xml`, `sqlite`, `text`, `other`), log sources, the matching encoding and
counts of items and bytes. They never contain sentinel values, matched bytes, file
contents, file names or paths, and a reader's error is reported only as
`unreadable`. `clean` means nothing matched. `exhaustive` means no bound was
reached, nothing was unreadable and every log reader was read. Release lanes
require both wherever the host has its own container (see the scope table below).

Every bound is enforced and listed in `bounds_reached` when hit:

| Bound | Default | Ceiling | Effect when reached |
| --- | --- | --- | --- |
| `max_files` | 5,000 | 100,000 | Stop walking; regular files, allow-listed or not, count |
| `max_depth` | 12 | 64 | Do not enter deeper directories; `0` scans only direct children |
| `max_bytes_per_file` | 8 MiB (8,388,608) | 256 MiB | Scan only the first bytes of that file |
| `max_total_bytes` | 128 MiB (134,217,728) | 1 GiB | Stop reading file contents |
| `max_duration` | 30 s | 600 s | Stop files and skip remaining log readers |
| `max_log_entries` | 20,000 | 1,000,000 | Stop reading logs |
| `max_log_bytes` | 16 MiB (16,777,216) | 256 MiB | Stop reading logs |

Files are read in 1 MiB chunks, and matches that span chunks are still found. The
explicit model-artifact allow-list (`ArtifactAllowList.modelArtifacts` /
`ArtifactAllowList.MODEL_ARTIFACTS`) skips the contents, but not the names, of
`gguf`, `mlmodel`, `mlmodelc`, `mlpackage`, `onnx`, `ort`, `safetensors` and
`tflite` files plus OpenMedKit's own model and tokenizer caches (`caches`
`OpenMed/MLXModels`, `OpenMed/TokenizerAssets`, `OpenMed/PreparedTokenizerAssets`
on Apple platforms; `no_backup` `openmed-model-cache` on Android). The default is
no allow-list.

The sweep does not decode compressed, encrypted, XML-entity-escaped, JSON-escaped
or line-wrapped base64 data, so workload sentinels should avoid `&`, `<`, `>`,
quotes and backslashes. Unified-log arguments logged as private (including
`NSLog` arguments) are redacted by the OS, so only public or literal text is
detected, and `print` output on stdout or stderr is not part of the unified log.
The `NoPhiLoggingTest` source guard covers direct console output on Android.
Storage outside the app container (keychain, pasteboard, system crash reports) is
out of scope.

Run the offline tests with planted sentinels, fake readers and negative controls:

```bash
swift test --filter 'PHISentinelSweepTests|ContainerSentinelSweepTests|ExtensionSentinelSweepTests'
cd android && ./gradlew :openmedkit:testDebugUnitTest --tests 'com.openmed.openmedkit.privacy.*'
```

`ContainerSentinelSweepTests` and `ExtensionSentinelSweepTests` run the policy
de-identification and extension redaction entry points on sentinel inputs, then
call `PHISentinelSweep.currentProcess(sentinels:logSince:)`. It always sweeps
this process's unified log and sweeps storage according to `HostStorageScope`:

| Scope | Host | Storage swept | Gate (`passesGate(for:)`) |
| --- | --- | --- | --- |
| `app_container` | App or app-hosted test on a device or Simulator; sandboxed macOS app | The per-app container | `clean` and `exhaustive` |
| `simulator_device_home` | SwiftPM package tests on a Simulator, run by the `xctest` agent | The simulated device's shared data directory, within bounds | `clean`, nothing unreadable, every log read |
| `unsandboxed_host` | Unsandboxed macOS, such as `swift test` | None; the user's home is never read | `clean` and `exhaustive` |

Each test attaches the value-free report JSON to the test result. The Robolectric
`PublicApiSentinelSweepTest` runs analyze, extract, chunked extract and
de-identify for every bundled policy, then sweeps the sandbox roots,
`android.util.Log` output and the internal SafeLog diagnostics. A second test
plants leaks to prove the real roots and readers find them.

### Device lanes

An iOS Simulator `xcodebuild test` job that runs `-only-testing:OpenMedKitTests`
and `-only-testing:ExtensionTests` includes both sweep tests without extra wiring.
Package tests have no app container on a Simulator: their
home is the simulated device's shared data directory, which holds system files
beyond the default bounds, so that lane checks `clean` within bounds rather than
an exhaustive container. An exhaustive container sweep needs an app-hosted XCTest
bundle or a device run, which should follow the same pattern after its kit calls:

```swift
let start = Date()
let person = try PHISentinel.synthetic(identifier: "workload.person")
// ... run analyze, extract and de-identify with person.plantedValue in the input ...
let scope = HostStorageScope.current
let report = try PHISentinelSweep.currentProcess(
    sentinels: [person],
    logSince: start,
    scope: scope
).run()
XCTAssertTrue(report.passesGate(for: scope), report.description)
```

Android instrumented tests compile the shared sweep sources, because the
`androidTest` source set includes them. In an emulator lane, after the real ONNX
Runtime run, an instrumented test should sweep the target context with logcat for
the current process, and allow-list the directory the lane copies its test model
into by path prefix:

```kotlin
val report = PhiSentinelSweep(
    sentinels = listOf(person),
    roots = AndroidAppStorageRoots.of(context),
    logReaders = listOf(LogcatProcessLogReader.forCurrentProcess()),
    allowList = ArtifactAllowList(
        fileExtensions = ArtifactAllowList.MODEL_ARTIFACTS.fileExtensions,
        pathPrefixes = mapOf(
            StorageClass.NO_BACKUP to setOf("openmed-model-cache", "openmed-emulator-lane"),
        ),
    ),
).run()
assertTrue(report.clean && report.exhaustive, report.toString())
```

Upload `report.jsonData()` / `report.toJson()` as the lane artifact. It is
value-free by construction.

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
