# OpenMed Medical Reasoning Demo

A separate SwiftUI example for local, document-grounded clinical conversation
with LiquidAI LFM2.5 through OpenMedKit.

The app deliberately keeps generative inference outside `OpenMedScanDemo`.
Its flow is:

1. Download the exact pinned `LiquidAI/LFM2.5-2.6B-MLX-4bit` artifact.
2. Review or replace a synthetic, de-identified clinical context.
3. Start a multi-turn evidence chat with separately streamed reasoning and
   final-answer text.

## Run on iPhone or iPad

1. Open `OpenMedMedicalReasoningDemo.xcodeproj` in Xcode.
2. Select the `OpenMedMedicalReasoningDemo` scheme.
3. Run on a recent physical iPhone or iPad with roughly 1.6 GB of free storage
   for the model plus working memory for inference.
4. Download the model, keep or edit the synthetic case, and start chatting.

The artifact is pinned by `OpenMedLFM.pinnedRevision`. The downloader requires
all seven official repository files, validates the LFM2 architecture, verifies
the exact 4-bit weight size, and uses OpenMedKit's shared local model cache.
There is no cloud fallback.

## Clinical Boundary

- Supply only synthetic or already de-identified context.
- The model is asked to ground answers in that context and distinguish evidence
  from inference.
- The UI exposes the model-generated reasoning trace separately and collapses
  it once the response completes or stops. It can be reopened for review.
- The demo does not diagnose or recommend treatment. Verify consequential
  details against the source and with a clinician.

## Develop and test on an Apple-silicon Mac

Select **OpenMedMedicalReasoningMac → My Mac** in Xcode. This native macOS
target compiles the **same SwiftUI screens, downloader, conversation store and
OpenMedKit runtime** as the iOS target. It runs real MLX inference, not a mock or
remote service, and uses ad-hoc signing without an iOS provisioning profile.
Download the model in the app, or set the Debug-only scheme environment variable
`OPENMED_LFM_MODEL_DIRECTORY` to an existing pinned artifact directory.

For the repeatable integration gate, from the repository root:

```bash
bash scripts/test_medical_reasoning_mac.sh /absolute/path/to/pinned/model
```

The script runs the package's LFM tests and the app's tests using Xcode (which
bundles MLX's Metal library), retains logs and `.xcresult` bundles, and fails on
any skipped test, failed test, or missing suite. The real-model cases cover:

- Official tokenizer and chat template, model loading and separate reasoning/
  answer streams.
- Clinical question → “thanks” → follow-up with completed turn history.
- Stop during generation, exclusion of failed/stopped turns from later prompts,
  producer synchronization before unload, and successful reload/restart.
- Rejection of concurrent generation and use after unload.

To acquire the exact test artifact separately, use the revision printed by
`OpenMedLFM.pinnedRevision` (currently below):

```bash
hf download LiquidAI/LFM2.5-2.6B-MLX-4bit \
  chat_template.jinja config.json generation_config.json model.safetensors \
  model.safetensors.index.json tokenizer.json tokenizer_config.json \
  --revision 04efa23776ce61ec34ec95ec34c859854c89542b \
  --local-dir /absolute/path/to/pinned/model
hf cache verify LiquidAI/LFM2.5-2.6B-MLX-4bit \
  --revision 04efa23776ce61ec34ec95ec34c859854c89542b \
  --local-dir /absolute/path/to/pinned/model
```

No clinical prompt leaves the machine. Test fixtures are synthetic. The Mac
Debug artifact override does not exist in iOS or Release builds.

Use iOS Simulator for layout/build checks, not as proof of MLX execution.
Mac tests remove the routine tokenizer/chat/lifecycle debugging loop on a phone;
iPhone memory pressure, thermal behaviour and background transitions still need
target-device QA. Plain `swift test` is not a substitute for the Xcode Metal
integration gate.

The project is generated with `xcodegen generate` using `project.yml`.
