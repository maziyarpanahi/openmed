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
