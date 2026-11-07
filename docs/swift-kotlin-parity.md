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

## Kit-Parity Result Profile

`tests/fixtures/parity/kit_parity_vectors.json` (schema
`openmed.kit_parity_vectors`, version 1) freezes the kit-parity result profile
`openmed.kit_parity_result`, version 1. Every OpenMedKit surface projects its
native de-identification result into the same small record and serializes it
to the same canonical JSON bytes. The record carries no source text, span
surface text, source hashes, or reversible mappings. The profile is scoped to
OpenMedKit result parity; it does not replace the broader v4 contract registry.

The vectors are registered in `fixtures/parity/span_parity_manifest.json` under
`kit_parity`, which pins their SHA-256 digest and names every consumer. The
same `python scripts/parity/update_native_span_fixture.py --check` and
`--write` commands validate and refresh them alongside the span fixture. The
Python reference implementation is `openmed.core.kit_parity`.

### Record

| Field | Type | Rule |
| --- | --- | --- |
| `profile` | string | Always `openmed.kit_parity_result`. |
| `profile_version` | integer | Always `1`. Any other version is rejected before the record is read further. |
| `offset_unit` | string | Always `unicode_scalar`. |
| `deidentified_text` | string | The exact de-identified output, compared as UTF-8 bytes. |
| `spans` | array | Applied spans ordered by `(start, end)`, non-empty, non-overlapping. May be empty. |
| `spans[].label` | string | Detector label, preserved byte for byte; ASCII `^[A-Za-z][A-Za-z0-9_:-]{0,79}$`. |
| `spans[].canonical_label` | string | One of the portable policy taxonomy labels (the full OpenMed canonical label set; see Portable Policy Subset). |
| `spans[].action` | string | One of `hash`, `keep`, `mask`, `redact`, `remove`, `replace`. |
| `spans[].start`, `spans[].end` | integer | Half-open Unicode scalar offsets into the source text. |
| `spans[].confidence_ppm` | integer, optional | `floor(confidence * 1_000_000 + 0.5)` in double precision, in `[0, 1000000]`. Omitted when the surface has no score. |
| `spans[].replacement` | string, optional | Required for every action except `keep`, where it is omitted. `remove` uses the empty string. |

Absent optional fields are omitted; `null` is never canonical. Canonical JSON
has lexicographically sorted keys, no insignificant whitespace, integers only,
and RFC 8785 string escaping: `"` and `\`, the short escapes `\b`, `\f`, `\n`,
`\r`, `\t`, `\u00xx` (lowercase hex) for the remaining C0 controls, and raw
UTF-8 for everything else, including `/`, DEL, and U+2028. Swift `String`
equality treats canonically equivalent text as equal, so every surface compares
canonical output and de-identified text as UTF-8 bytes.

### Offsets and Conversions

Unicode scalar offsets are the only stored coordinates. Each vector span also
carries derived `utf8_*` and `utf16_*` offsets so every surface tests its
native conversions in both directions. An offset that splits a surrogate pair
or a UTF-8 sequence fails with `offset_not_scalar_boundary`; an offset past the
text fails with `offset_out_of_bounds`.

| Surface | Native index | Conversions |
| --- | --- | --- |
| Python | `str` index (scalar) | `scalar_to_utf8_offset`, `scalar_to_utf16_offset`, `utf8_to_scalar_offset`, `utf16_to_scalar_offset` in `openmed.core.kit_parity` |
| Swift | `String.Index` | `KitParity.utf8Offset(in:scalarOffset:)`, `KitParity.utf16Offset(in:scalarOffset:)`, `KitParity.scalarOffset(in:utf8Offset:)`, `KitParity.scalarOffset(in:utf16Offset:)` |
| Kotlin | UTF-16 index | `KitParity.scalarToUtf16Offset`, `KitParity.utf16ToScalarOffset`, `KitParity.scalarToUtf8Offset`, `KitParity.utf8ToScalarOffset`, backed by `UnicodeOffsetContract` |
| Web | UTF-16 index | `scalarToUtf16Offset`, `utf16ToScalarOffset`, `scalarToUtf8Offset`, `utf8ToScalarOffset` exported by `openmed` |

The web package keeps JavaScript UTF-16 indices on `OpenMedSpan`;
`toKitParityResult` converts them against the result's source text. The React
Native bridge receives scalar offsets from the native modules and projects them
unchanged. Electron renderer spans carry UTF-16 indices, which
`toRendererKitParityResult` converts.

### Validation Order and Reason Codes

Parsers check, in this order: JSON syntax (`json_invalid`), object shape
(`result_not_object`), `profile` (`profile_unknown`), `profile_version`
(`profile_version_unsupported`), unknown top-level keys (`field_unknown`),
`offset_unit` (`offset_unit_unsupported`), `deidentified_text`, `spans`, then
each span's `label` (`label_invalid`), `canonical_label` (`label_unknown`),
`action` (`action_unknown`), `start` and `end` (`offset_invalid`),
`confidence_ppm` (`confidence_invalid`), and `replacement`
(`replacement_unexpected`), then ordering (`span_order_invalid`,
`span_overlap`), and finally that the input bytes are the canonical bytes
(`result_not_canonical`). A missing field is `field_missing`, an explicit
`null` is `field_null`, and a wrong JSON type is `field_type_invalid`.
Verifying a parsed record against its source text adds `offset_out_of_bounds`
and `deidentified_text_mismatch`.

Each failure carries the reason code and a JSON field path such as
`$.spans[2].end`, and its message is exactly `<code> at <path>`. Unknown keys
are reported at the enclosing object path, so a key name is never echoed. The
vectors embed a privacy sentinel in rejected values, and every surface asserts
that no failure message contains it.

### Bundle Compatibility

The profile accepts offset contract version `1` (the version of
`tests/fixtures/parity/offset_contract.json`) and decoder contract version `1`
(the BIO span decoder emitting span schema version 1). The compatibility gate
(`check_compatibility` / `accept_result` in Python, `KitParity.accept` in Swift
and Kotlin, `acceptKitParityResult` on the web) rejects any other version with
`offset_contract_version_unsupported` or `decoder_contract_version_unsupported`
before a result produced under that bundle is parsed. The signed bundle
manifest that will supply these versions is tracked separately; until then
callers pass the versions their bundle declares.

### Vectors

The 13 positive vectors cover ASCII text with spans at both text edges,
emoji ZWJ sequences with skin-tone modifiers and flags, decomposed and stacked
combining marks, CJK text after a supplementary-plane ideograph, Indic conjunct
clusters, CRLF, tab, and C0 control escapes, empty input, Unicode text without
identifiers, overlapping detections resolved to the longer span with a kept
span, adjacent spans, every profile action, confidence extremes with an absent
score, and renderer-safe redaction. Ten vectors also carry the raw `engine`
detections and the bundled policy, so Swift and Kotlin run their real policy
de-identification engines and compare the result bytes.

Negative controls cover malformed JSON, unknown profiles and versions, unknown
fields, labels, and actions, missing, `null`, and mistyped fields, fractional
and out-of-range confidence, replacements on `keep`, out-of-order,
duplicate, and overlapping spans, non-canonical whitespace, key order, and
ASCII escaping, offsets past the source, changed bytes, UTF-16 offsets passed
off as scalar offsets, and NFC output for NFD input. Every consumer also alters
an offset, a label, an action, the span order, and a de-identified byte and
asserts that the gate fails with a value-free report.

### Kit-Parity Consumers and CI Gates

| Surface | Test | CI gate |
| --- | --- | --- |
| Python | `tests/unit/parity/test_kit_parity_vectors.py` | the `pytest` run in `.github/workflows/ci.yml` |
| Swift | `swift/OpenMedKit/Tests/OpenMedKitTests/KitParityVectorTests.swift` | `swift test --filter KitParityVectorTests` in `.github/workflows/swift-test.yml` |
| Kotlin | `android/openmedkit/src/test/kotlin/com/openmed/openmedkit/parity/KitParityVectorTest.kt` | `--tests '*KitParityVectorTest'` in `.github/workflows/android-ci.yml` |
| Web | `tests/web/test_kit_parity.spec.ts` | `npm --prefix js/openmedkit-web test` in the `openmedkit-kit-parity` job of `.github/workflows/ci.yml` |
| React Native | `tests/mobile/test_rn_kit_parity.ts` | `npm --prefix js/openmedkit-react-native test` in the same job |
| Electron | `tests/web/test_electron_kit_parity.spec.ts` | `npm --prefix js/openmedkit-electron test` in the same job |

Projection notes:

- Python projects `DeidentificationResult` through
  `from_deidentification_result`. Methods outside the profile, such as
  `shift_dates` and `format_preserve`, are rejected rather than mapped.
  Python mask placeholders name the canonical label (`[PERSON]`), matching the
  kits; the portable policy vectors below compare the shared decision engine
  byte for byte.
- Swift and Kotlin project `PolicyDeidentificationResult` through
  `kitParityResult()` and `toKitParityResult()`. Both load their label taxonomy,
  including `DIET_TYPE`, `FEEDING_ROUTE`, `NUTRITIONAL_STATUS`, and
  `NUTRITION_TARGET`, from the generated portable taxonomy.
- React Native rejects native spans with missing or non-integer offsets or an
  unknown action name with `OpenMedKitParityError` instead of coercing them.
- Electron renderer spans carry no action or replacement and always redact to
  `[canonical_label]`, so the Electron projection claims the vectors whose every
  span is a `redact` to that token.
- A vector a surface cannot represent natively (an absent score on Swift,
  Kotlin, and Python) still passes that surface's strict parse and canonical
  round trip.

To change the vectors, edit the inputs (`text`, the span inputs, `engine`, and
the negative controls), run
`python scripts/parity/update_native_span_fixture.py --write`, and run every
consumer gate above.

## Portable Policy Subset

`openmed.core.portable_policy` declares the bounded policy subset that every
kit executes identically and is its Python reference implementation. The
Python policy runtime stays richer; anything outside the subset is excluded by
name rather than approximated.

### Profiles and Resources

The portable profiles are `australia_privacy_act`, `canada_pipeda`,
`clinical_minimal_redaction`, `gdpr_art9_health`, `gdpr_pseudonymization`,
`hipaa_expert_review_assist`, `hipaa_safe_harbor` (the default),
`research_limited_dataset`, and `strict_no_leak`. Every other file in
`openmed/core/policies/` is listed in the manifest's `excluded_profiles` with a
value-free reason (`python_only_profile`, or `structured_schema_policy` for the
FHIR and OMOP schema policies). Loading one on any kit fails with
`profile_not_portable`.

`openmed/core/policies/`, `openmed/core/labels.py`, and
`openmed/core/thresholds.json` are the only sources.
`python scripts/parity/generate_portable_policies.py --write` lowers them into
versioned canonical JSON (`openmed.portable_policy`, IR version 1) and writes
byte-identical copies to:

| Kit | Location |
| --- | --- |
| Swift | `swift/OpenMedKit/Sources/OpenMedKit/Resources/policies/` |
| Kotlin | `android/openmedkit/src/main/assets/policies/` |
| Web | `js/openmedkit-web/src/portable-policy-resources.ts` (the same documents embedded as strings) |
| React Native | `js/openmedkit-react-native/src/portable-taxonomy-labels.ts` (taxonomy labels only) |

Each kit ships one file per profile, `portable_policy_taxonomy.json`, and
`portable_policy_manifest.json`. The manifest pins the `sha256:` digest of every
profile and of the taxonomy, the profile aliases, and the exclusions. A profile
declares its source digest, actions for every taxonomy label, the policy-class
actions, the default action, the threshold profile resolved to integer parts per
million, the arbitration mode, `strict_no_leak`, the merge and surrogate
algorithm versions, its required capabilities, and the Python-only features it
omits (`excluded_features`, for example `safety_sweep`,
`forced_cascade_execution`, `language_specific_thresholds`, and
`realistic_surrogates`).

`--check` (the default) fails when any kit copy is missing, stale, or
unexpected, or when a vector expectation drifts. It runs in the `lint` job of
`.github/workflows/ci.yml` and inside `tests/unit/parity/test_portable_policy.py`.

### Label Taxonomy

The portable taxonomy is the full Python canonical label set, which is also the
kit-parity result vocabulary. Normalization strips whitespace and one BIOES
prefix, lowercases, keeps `a-z0-9`, and looks the key up in the generated alias
table. It equals `openmed.core.labels.normalize_label` for ASCII labels; the
Chinese CMeEE mapping is language-conditional and stays Python-only. Unknown
labels normalize to `OTHER`, and every kit loads the same table instead of a
hand-maintained one, so role-qualified names such as `patient_name` and
`doctor_name` resolve to `PERSON` everywhere.

### Loading and Fail-Closed Outcomes

Loading checks, in order: the manifest schema and IR version, the profile name
(aliases resolve first), an optional caller-pinned digest, the profile resource
digest, the taxonomy resource digest, and then the profile document (IR schema,
IR version, name, taxonomy binding, required capabilities, merge algorithm,
surrogate algorithm, arbitration mode, `strict_no_leak`, actions, default
action, thresholds). Every failure is a reason code and a JSON field path:

| Code | Meaning |
| --- | --- |
| `profile_name_invalid`, `profile_unknown`, `profile_not_portable` | The name is blank, unknown, or names an excluded profile. |
| `digest_mismatch` | The caller pinned a different profile digest. |
| `resource_digest_mismatch`, `taxonomy_digest_mismatch` | A bundled resource is stale against the manifest, or a profile binds a different taxonomy. |
| `manifest_invalid`, `manifest_version_unsupported`, `ir_version_unsupported`, `profile_invalid`, `taxonomy_invalid` | Schema or version mismatch. |
| `capability_missing` | The profile requires a capability the runtime does not implement. A strict no-leak profile on a runtime without `strict_no_leak.v1` is refused, never downgraded. |
| `merge_algorithm_unsupported`, `surrogate_algorithm_unsupported` | The profile names an algorithm version the runtime does not implement. |

### Decisions

`decide` (Python), `PolicyEngine.decide` (Swift and Kotlin), and
`decidePortablePolicy` (web) take the source text and normalized detections
(label, scalar offsets, `confidence_ppm`, and `source` of `model` or
`semantic`) and run these stages:

1. Validate detections, overrides, and the surrogate key
   (`detection_label_invalid`, `detection_offset_invalid`,
   `detection_offset_out_of_bounds`, `detection_confidence_invalid`,
   `detection_source_invalid`, `override_label_unknown`,
   `override_action_unknown`, `surrogate_key_invalid`).
2. Normalize labels through the taxonomy.
3. Merge each semantic unit with the model spans it overlaps: the dominant
   model label wins when the unit's label is among the model labels or less
   specific; the extent is the union; the confidence is
   `(6 * model_average + 4 * unit + 5) // 10` for a validated unit and
   `(9 * model_average + 1 * unit + 5) // 10` otherwise, in integer parts per
   million. A unit with no overlapping model span is a `semantic_only`
   candidate.
4. Keep a candidate when `confidence_ppm >= keep_floor_ppm` for its label and
   flag it `escalate` when it is below `escalate_below_ppm` (review metadata
   only; no cascade runs portably).
5. Resolve the action: a caller override, else the profile action. Under
   `strict_no_leak`, `keep` becomes `mask` and the decision records
   `strict_upgrade`.
6. Collapse exact duplicates (same offsets and label), then cluster transitive
   overlaps. Each cluster becomes one span over the union of its members, so no
   overlapping text is left unredacted. The winner order is: a protective
   (non-`keep`) action, higher specificity, longer span, higher confidence,
   earlier start, then ASCII label order. Under `high_recall_union`, only the
   members carrying the strongest action
   (`keep < replace < hash < mask < redact < remove`) compete.
7. Render right to left at scalar offsets: `mask` and `redact` give
   `[LABEL]`, `remove` gives the empty string, `hash` gives
   `LABEL_` plus the first 12 bytes of the SHA-256 of the span as hex, and
   `keep` leaves the text in place.

The output is a kit-parity result plus one value-free decision record per
candidate (`inputs`, `kind`, labels, offsets, `confidence_ppm`,
`keep_floor_ppm`, `escalate`, `action`, `strict_upgrade`, `outcome`, and the
output `span`). Grapheme snapping stays a platform pre-step in Swift and
Kotlin; the vectors use grapheme-aligned offsets. The Swift and Kotlin policy
de-identification paths (`OpenMed.deidentify(_:policy:)` and
`DeidentifyEngine`) now run through this engine.

### Surrogates

Portable `replace` is a deterministic keyed surrogate
(`openmed.portable_surrogate.hmac_sha256`, version 1). With a caller-supplied
key of 16 to 64 bytes, a span renders `[LABEL_token]`, where `token` is the
first 8 bytes, as hex, of HMAC-SHA256 over
`openmed.portable_surrogate.v1` 0x00 label 0x00 span text. The same key, label,
and text always give the same token, within a document and across documents;
a different key gives a different token. Without a key, `replace` renders the
value-free placeholder `[LABEL_REPLACED]`. Keys and reversible mappings stay in
caller-controlled storage and never enter a profile, a resource, a decision, or
an error. Python's realistic, locale-aware surrogates are Python-only and listed
in `excluded_features`.

### Portable Vectors and Consumers

`tests/fixtures/parity/portable_policy_vectors.json` (schema
`openmed.portable_policy_vectors`, version 1) holds synthetic decision cases
(threshold edges, label normalization, keep and mask overrides, overlap and
tie-breaking, duplicates, semantic merging, semantic-only matches,
`strict_no_leak`, `high_recall_union`, and surrogate reuse), error cases, and
load cases (every profile digest, aliases, refusals, stale digests, schema and
capability mismatches). The surrogate keys are named `synthetic_test_key_*` and
are for tests only. The expected blocks are derived from the Python reference by
`generate_portable_policies.py --write`.

| Surface | Test | CI gate |
| --- | --- | --- |
| Python | `tests/unit/parity/test_portable_policy.py` | the `pytest` run and the generator `--check` step in `.github/workflows/ci.yml` |
| Swift | `swift/OpenMedKit/Tests/OpenMedKitTests/PortablePolicyVectorTests.swift` | `swift test --filter PortablePolicyVectorTests` in `.github/workflows/swift-test.yml` |
| Kotlin | `android/openmedkit/src/test/kotlin/com/openmed/openmedkit/policy/PortablePolicyVectorTest.kt` | `--tests '*PortablePolicyVectorTest'` in `.github/workflows/android-ci.yml` |
| Web | `tests/web/test_portable_policy.spec.ts` | `npm --prefix js/openmedkit-web test` in the `openmedkit-kit-parity` job |

To change a profile, edit `openmed/core/policies/`, run
`python scripts/parity/generate_portable_policies.py --write`, review the
regenerated resources and vectors, and run every consumer above.
