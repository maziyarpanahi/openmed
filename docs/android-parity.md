# Android Span Parity Protocol

Android ONNX Runtime Mobile exports must preserve the same tokenization and
span decoding behavior as the Python reference. The parity fixture in
`android/openmedkit/src/test/resources/parity/android_span_parity.json` pins
that contract for synthetic clinical text.

## Generate Fixtures

Export a token-classification checkpoint with the Android ONNX profile, then
generate the parity JSON:

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model dslim/bert-base-NER \
  --output dist/example-android-onnx \
  --profile android

.venv/bin/python scripts/android/generate_parity_fixtures.py \
  --export-dir dist/example-android-onnx \
  --output android/openmedkit/src/test/resources/parity/android_span_parity.json
```

The generator validates the ONNX graph with the Android profile, loads the
exported tokenizer and `id2label.json`, runs the ONNX `logits` output through a
deterministic argmax decoder, and writes token IDs, character offsets, predicted
labels, and decoded spans.

## Fixture Contract

Android parity tests must compare these fields:

- `cases[].text`: exact synthetic input text.
- `cases[].tokens[].id`: exact tokenizer ID sequence.
- `cases[].tokens[].offset`: exact `[start, end)` character offsets.
- `cases[].spans[].canonical_label`: exact canonical label.
- `cases[].spans[].start` and `cases[].spans[].end`: exact span boundaries.

The committed tolerance contract is strict:

```json
{
  "token_ids": "exact",
  "char_offsets": "exact",
  "span_labels": "exact",
  "span_boundaries": {
    "mode": "exact",
    "tolerance_chars": 0
  },
  "logit_ties": "lowest_label_id"
}
```

If an Android decoder produces the same label but shifts a boundary by one
character, the parity test should fail. Boundary tolerance is documented in the
fixture so a future change can be reviewed explicitly instead of drifting
silently.

## Privacy Rules

Parity inputs must remain synthetic. The committed fixture marks each case with
`synthetic: true` and `phi_free: true`, uses `SYNTH_` placeholders, and rejects
common PHI-shaped patterns such as emails, phone numbers, and SSNs during
Python validation.

Span records do not include surface text. Android tests should slice
`cases[].text` with the expected offsets when they need the span surface, and
use `text_hash` as a deterministic integrity check.

## Android Resource Layout

The Android module loads fixtures from:

```text
android/openmedkit/src/test/resources/parity/android_span_parity.json
```

The resource is plain JSON and has no Android-specific binary encoding. A JVM
unit test can read it with the class loader, parse `cases`, run the Android
tokenizer and ONNX session for each `text`, and compare tokens and spans against
the fields listed above.

## Experimental multimodal metadata contracts

`com.openmed.openmedkit.multimodal` mirrors the committed Python metadata
contracts. This Android surface is **experimental** and is limited to contract
parsing and serialization; it does not run inference, decode assets, capture
media, align intake, or render UI.

| Kotlin type | Python reference | Accepted version |
| --- | --- | --- |
| `AssetManifest` | `AssetManifest` | integer `version: 1` (defaults to 1 if omitted) |
| `ManifestProfile` | `PreflightReport.metadata_profile` | string `version: "1.0"` |
| `PreflightFinding` | `PreflightFinding.to_dict()` | inherited from the containing preflight report v1; no standalone version field |
| `AbstentionRecord` | `AbstentionRecord` | integer `schema_version: 1` |
| `ProviderResultEnvelope` | `ProviderResultEnvelope` | `openmed.multimodal.provider_result.v1` |

Call `Type.fromJson(payload)` to construct a validated instance, and `toJson()`
to serialize it. Profile references carry only modality and version; their
required, optional, and inapplicable field sets come from the existing Python
v1.0 profiles. No new profile wire schema or preflight orchestration is added.
Asset and provider JSON use sorted keys; finding, profile-reference, and
abstention JSON preserve the Python field order. Numeric JSON preserves the
integer/float distinction, binary64 shortest-roundtrip spelling, and signed
zero where the Python reference preserves them.

Parsing rejects unknown fields (including nested count keys), duplicate keys,
unknown schema versions, unknown reason codes, invalid stage/reason and
outcome/digest combinations, wrong numeric types, non-finite values, and values
outside the reference bounds. Each input is limited to 64 KiB of UTF-8 and 16
nesting levels. Errors have a fixed category without submitted values or parser
causes. Version upgrades require a deliberate contract and vector update;
there is no permissive fallback.

The types carry only controlled codes, bounded numbers, opaque asset/provider/
model references, and digests. They have no path, URL, transcript, arbitrary
message, source payload, credentials, or clinical output field. Opaque IDs
must be caller-generated references, never identifiers derived from patient
content. Passing structural validation is not evidence of clinical validity or
provider qualification. No assets or model weights are bundled.

These contracts are non-diagnostic and do not authorize clinical action.
Consumers must bind a non-diagnostic notice and explicit reviewer confirmation
to any consequential output in their own integration. A successful provider
envelope alone must never trigger a clinical decision. All processing remains
local; these types add no transport, telemetry, or cloud fallback.

### Offline shared conformance vectors

`tests/fixtures/parity/multimodal_contracts_v1.json` is the single versioned,
synthetic fixture loaded by Python and Kotlin. Android already includes that
directory as JVM test resources. The vectors cover all mirrored types, all
abstention reasons and valid stage combinations, provider outcomes and counters,
manifest bounds, all finding reasons, arbitrary-precision pixel counts, and
binary64 edge cases. Both suites compare canonical UTF-8 bytes exactly.
Negative Kotlin controls reject sensitive fields, unsafe references, malformed
JSON, unknown versions/codes, numeric overflows, and invalid combinations.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/test_android_contract_parity.py -q
ANDROID_HOME="$HOME/Library/Android/sdk" ./android/gradlew -p android --offline \
  :openmedkit:testDebugUnitTest \
  --tests 'com.openmed.openmedkit.parity.MultimodalContractParityTest'
```

These tests need no model, network, microphone, camera, or Android device.
The Gradle toolchain, SDK, and build dependencies must already be installed
and cached for `--offline` execution.
