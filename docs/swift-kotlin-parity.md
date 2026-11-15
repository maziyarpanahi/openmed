# Swift-Kotlin OpenMedKit Parity

OpenMedKit keeps one public mobile contract across Swift and Kotlin. This
checklist pins the required Android symbols against the Swift package symbols
so API drift is caught in tests instead of downstream apps.

## Public API Checklist

| Swift symbol | Required Kotlin counterpart | Status |
| --- | --- | --- |
| `OpenMed.analyzeText(_:confidenceThreshold:)` | `OpenMed.analyzeText(text, confidenceThreshold)` | Present and covered by `ApiParityTest` |
| `OpenMed.extractPII(_:confidenceThreshold:useSmartMerging:)` | `OpenMed.extractPII(text, confidenceThreshold, useSmartMerging)` | Present and covered by `ApiParityTest`, Kotlin `SpanEquivalenceTest`, and Swift `SpanEquivalenceParityTests` |
| `OpenMed.extractPIIChunked(_:confidenceThreshold:chunkTokenLimit:tokenOverlap:useSmartMerging:)` | `OpenMed.extractPIIChunked(text, confidenceThreshold, chunkTokenLimit, tokenOverlap, useSmartMerging)` | Present; delegates to the local Android scaffold path until Android windowed model inference lands |
| `EntityPrediction.label` | `EntityPrediction.label` | Present |
| `EntityPrediction.text` | `EntityPrediction.text` | Present |
| `EntityPrediction.confidence` | `EntityPrediction.confidence` | Present |
| `EntityPrediction.start` | `EntityPrediction.start` | Present |
| `EntityPrediction.end` | `EntityPrediction.end` | Present |
| `OpenMedModelStore.downloadMLXModel(repoID:revision:cacheDirectory:)` | `OpenMedModelStore.downloadMLXModel(repoID, revision, cacheDirectory)` | Public entry point present; Android download implementation is intentionally deferred |
| `OpenMedModelStore.cachedMLXModelDirectory(repoID:revision:cacheDirectory:)` | `OpenMedModelStore.cachedMLXModelDirectory(repoID, revision, cacheDirectory)` | Present |
| `OpenMedModelStore.isMLXModelCached(repoID:revision:cacheDirectory:)` | `OpenMedModelStore.isMLXModelCached(repoID, revision, cacheDirectory)` | Present |
| `OpenMedModelStore.mlxModelCacheState(repoID:revision:cacheDirectory:)` | `OpenMedModelStore.mlxModelCacheState(repoID, revision, cacheDirectory)` | Present |
| `OpenMedMLXModelCacheState.missing` | `OpenMedMLXModelCacheState.missing` | Present |
| `OpenMedMLXModelCacheState.partial` | `OpenMedMLXModelCacheState.partial` | Present |
| `OpenMedMLXModelCacheState.ready` | `OpenMedMLXModelCacheState.ready` | Present |

## Shared Span Fixture

The repository-root fixture in `fixtures/parity/` is the canonical synthetic
native-kit golden for its schema version. Swift and Kotlin read the same files;
neither kit keeps an embedded copy.

| File | Contents |
| --- | --- |
| `fixtures/parity/expected_spans.json` | Schema `openmed.native_span_parity`, version 2: synthetic cases, expected spans, policy actions, and expected de-identified text |
| `fixtures/parity/synthetic_clinical_note.txt` | The original ASCII clinical note, referenced by the `clinical_note_ascii` case through `text_file` |
| `fixtures/parity/span_parity_manifest.json` | SHA-256 digest of every fixture file, the update command, and every consumer with its CI gate |

All values are invented. Each case has an `id`, a bundled `policy` profile,
either inline `text` or a sibling `text_file`, the ordered `spans` that
`extractPII` must return, the ordered `actions` that policy de-identification
must return, and `expected_deidentified_text`. Spans and actions store only
labels, offsets, canonical labels, action names, and replacement tokens, so the
comparison never depends on confidence scores or model metadata.

The cases cover:

- `clinical_note_ascii`: the original note under `hipaa_safe_harbor`.
- `emoji_and_combining_marks`: an emoji ZWJ sequence with a skin-tone
  modifier, a regional-indicator flag, decomposed and stacked combining marks,
  CJK text, and supplementary emoji touching span boundaries, so scalar,
  UTF-8, and UTF-16 offsets all differ.
- `overlapping_identifiers`: two overlapping detections where extraction keeps
  both spans and de-identification keeps the longer span, under
  `clinical_minimal_redaction` so `mask` and `keep` actions are both pinned.
- `unicode_without_identifiers`: emoji and combining marks with no
  identifiers, which must produce no spans and unchanged bytes.

### Coordinate Contract

`start` and `end` are half-open Unicode scalar (code point) offsets and are the
only canonical coordinates. `utf8_start`, `utf8_end`, `utf16_start`, and
`utf16_end` are derived from them by the update command and exist so each
runtime can test its native conversions:

- Swift: `NSRegularExpression` reports UTF-16 `NSRange` values, which the
  recognizer converts once through `String.Index` to scalar offsets. The tests
  check `String.Index.utf16Offset(in:)`, the UTF-8 view distance, and the
  reverse conversions from UTF-16 and UTF-8 offsets back to scalar offsets.
- Kotlin: the tests check `UnicodeOffsetContract.scalarToUtf16Index`,
  `UnicodeOffsetContract.utf16ToScalarOffset`, and the UTF-8 byte length of
  the native UTF-16 prefix.

De-identified output is compared as UTF-8 bytes. A mismatch reports the case
identifier and the first differing byte offset, never the text.

### Consumers and CI Gates

| Surface | Test | CI gate |
| --- | --- | --- |
| Python validator | `tests/unit/parity/test_native_span_fixture.py` | the `pytest` run in `.github/workflows/ci.yml` |
| Swift | `swift/OpenMedKit/Tests/OpenMedKitTests/SpanEquivalenceParityTests.swift` | `swift test --filter SpanEquivalenceParityTests` in `.github/workflows/swift-test.yml` |
| Kotlin | `android/openmedkit/src/test/kotlin/com/openmed/openmedkit/parity/SpanEquivalenceTest.kt` | `--tests '*SpanEquivalenceTest'` in `.github/workflows/android-ci.yml` |

Both native workflows trigger on `fixtures/parity/**` and `scripts/parity/**`.
Every consumer validates the schema version and the manifest digests before it
uses a case, so a stale or hand-edited copy fails before any parity assertion
runs. Each consumer also carries a negative control: a shifted span end, a
changed label, a dropped span, changed fixture bytes, and a changed digest must
each fail.

Swift loads the files through `#filePath` relative to the repository root, the
same way `OffsetContractParityTests` loads `tests/fixtures/parity/`. Kotlin
registers `fixtures/parity` as a JVM test resource directory in
`android/openmedkit/build.gradle.kts` and runs under Robolectric, so grapheme
snapping uses the Android ICU segmenter.

### Deterministic Backends

Kotlin runs the public `OpenMed().extractPII(...)` scaffold, which uses the
bundled local recognizer, and then the policy `DeidentifyEngine`.

Swift runs the public `OpenMed.extractPII`, `OpenMed.extractPIIChunked`, and
`OpenMed.deidentify(_:policy:)` methods over an internal
`OpenMedSpanRecognizer` seam. The test recognizer mirrors the Kotlin
recognizer's patterns, capture groups, and ordering. The seam is internal and
reachable only through `@testable import`; it loads no model or tokenizer
assets and has no network path. The public backends and their defaults are
unchanged.

Both kits run with `useSmartMerging = false`. Swift's semantic merge pass is
Swift-only, and the Kotlin parameter is currently a no-op, so smart merging is
not part of this gate. Extending the gate to it needs a Kotlin semantic merger
first.

### Updating the Fixture

1. Edit `fixtures/parity/expected_spans.json` (or the referenced text file):
   case text, scalar `start` and `end`, labels, and policy actions.
2. Run `python scripts/parity/update_native_span_fixture.py --write`. It
   recomputes the UTF-8 and UTF-16 offsets, the expected de-identified text,
   and the manifest digests, and writes ASCII-escaped JSON so combining marks
   and emoji stay visible in review and cannot be normalized by an editor.
3. Run `python scripts/parity/update_native_span_fixture.py --check` and every
   consumer gate listed in the manifest.

Review rules:

- A fixture change lands only with all three consumers passing. A change that
  makes one kit fail is a parity regression to fix in that kit, not a reason to
  fork the fixture.
- Keep every value synthetic. Never add real patient text, identity-bearing
  examples, secrets, or reversible mappings.
- Changing the coordinate contract or the schema version needs a schema bump
  and matching loader updates in every consumer.
- Adding a consumer means adding it to `CONSUMERS` in the update script, to the
  table above, and to its workflow's path filters.
