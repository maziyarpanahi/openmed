# Swift-Kotlin OpenMedKit Parity

OpenMedKit keeps one public mobile contract across Swift and Kotlin. This
checklist pins the required Android symbols against the Swift package symbols
so API drift is caught in tests instead of downstream apps.

## Public API Checklist

| Swift symbol | Required Kotlin counterpart | Status |
| --- | --- | --- |
| `OpenMed.analyzeText(_:confidenceThreshold:)` | `OpenMed.analyzeText(text, confidenceThreshold)` | Present and covered by `ApiParityTest` |
| `OpenMed.extractPII(_:confidenceThreshold:useSmartMerging:)` | `OpenMed.extractPII(text, confidenceThreshold, useSmartMerging)` | Present and covered by `ApiParityTest` and `SpanEquivalenceTest` |
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

Both mobile kits use the same synthetic fixture files:

- `fixtures/parity/synthetic_clinical_note.txt`
- `fixtures/parity/expected_spans.json`

The fixture is synthetic and contains invented names, dates, contact details,
and record identifiers. `expected_spans.json` stores only `label`, `start`, and
`end` so platform tests compare offsets without depending on confidence scores
or model-specific metadata.

The Kotlin tests register `fixtures/parity` as a JVM test resource directory in
`android/openmedkit/build.gradle.kts`. `SpanEquivalenceTest` reads the note and
expected spans from the test classpath, runs `OpenMed.extractPII(...)`, and
asserts exact span equivalence.

Swift tests should consume the same files from the repository root instead of
copying fixture data into the Swift package. A Swift test can resolve the repo
fixture directory relative to the package checkout and decode the expected
spans with `JSONDecoder`, then compare `EntityPrediction.label`,
`EntityPrediction.start`, and `EntityPrediction.end` against the shared JSON.
Keeping both tests on these root-level files makes fixture changes explicit in
one diff.
