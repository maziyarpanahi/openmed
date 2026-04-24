# OpenMed v1.1.0 Release Notes

OpenMed v1.1.0 expands the Apple/on-device story from the 1.0 launch into a broader local inference platform: Python MLX, Swift OpenMedKit, Privacy Filter, experimental GLiNER-family tasks, and a much stronger iPhone scan demo workflow.

## Highlights

- Native Python MLX support now covers OpenMed token-classification artifacts, OpenAI Privacy Filter artifacts, experimental GLiNER span NER, GLiClass zero-shot classification, and GLiNER-Relex relation extraction.
- OpenMedKit now includes native Swift MLX runtimes for DeBERTa-v2/v3-backed custom tasks and OpenAI Privacy Filter, plus public zero-shot APIs for NER, classification, and relation extraction.
- `OpenMed/privacy-filter-mlx` and `OpenMed/privacy-filter-mlx-8bit` can run locally through OpenMed without external inference services.
- The iPhone scan demo has been redesigned into a guided clinical workflow: input, OCR review, de-identification, clinical extraction, and summary/export.
- Model downloads are cleaner: public artifacts are cached once, token UI was removed, and offline reuse is the default story after preparation.

## Python

- Added MLX custom-task dispatch for `openmed-mlx.json` task/family pairs.
- Added `PrivacyFilterMLXPipeline` with tiktoken-style tokenization, byte-offset reconstruction, BIOES/Viterbi decoding, and model-led span repair.
- Added experimental pipelines for GLiNER span NER, GLiClass, and GLiNER-Relex.
- Added artifact handling for `weights.safetensors` with `weights.npz` fallback.
- Added tests for Privacy Filter decoding, MLX artifact loading, custom task dispatch, and PII post-processing.

Install:

```bash
pip install -U "openmed[mlx]"
```

Privacy Filter example:

```python
from huggingface_hub import snapshot_download
from openmed.mlx.inference import create_mlx_pipeline

path = snapshot_download("OpenMed/privacy-filter-mlx-8bit")
pipe = create_mlx_pipeline(path)

entities = pipe("Alice Smith emailed alice@example.com and called 415-555-0101.")
print(entities)
```

## Swift And Apple Apps

- Added native Swift MLX DeBERTa support for the converted GLiNER-family artifacts.
- Added native Swift OpenAI Privacy Filter model, tokenizer, and post-processing.
- Added public OpenMedKit APIs:
  - `OpenMedZeroShotNER`
  - `OpenMedZeroShotClassifier`
  - `OpenMedRelationExtractor`
- Added first-run model download/cache support through `OpenMedModelStore`.
- Added Swift tests for artifact validation, zero-shot task setup, Privacy Filter decoding, sample assets, and post-processing.

Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/maziyarpanahi/openmed.git", from: "1.1.0"),
]
```

Privacy Filter example:

```swift
import OpenMedKit

let modelURL = try await OpenMedModelStore.downloadMLXModel(
    repoID: "OpenMed/privacy-filter-mlx-8bit"
)

let openmed = try OpenMed(backend: .mlx(modelDirectoryURL: modelURL))
let entities = try openmed.extractPII(
    "Alice Smith emailed alice@example.com and called 415-555-0101."
)
```

GLiNER-Relex example:

```swift
import OpenMedKit

let modelURL = try await OpenMedModelStore.downloadMLXModel(
    repoID: "OpenMed/gliner-relex-base-v1.0-mlx"
)

let extractor = try OpenMedRelationExtractor(modelDirectoryURL: modelURL)

let result = try extractor.extract(
    "Aspirin was prescribed for headache after migraine symptoms.",
    entityLabels: ["medication", "condition", "symptom"],
    relationLabels: ["treats", "associated with"],
    threshold: 0.5,
    relationThreshold: 0.9
)
```

## Demo Apps

- `swift/OpenMedDemo` can test OpenMed PII and the public 8-bit Privacy Filter MLX artifact on macOS/iOS.
- `swift/OpenMedScanDemo` is now a guided iPhone-first demo suitable for product videos and App Store prep.
- The scan demo includes a generated clinical sample image, OCR review, de-identification model selection, clinical extraction presets, model preparation UI, and summary/export flow.
- App privacy assets were added for local document scanning.

## Upgrade Notes

- No intentional Python API breaking changes from v1.0.0.
- For Swift packages, update the dependency requirement to `from: "1.1.0"` after the release tag is published.
- MLX inference should be validated on Apple Silicon macOS or a physical iPhone/iPad. iOS Simulator remains outside the MLX acceptance path.
- GLiNER-family MLX support is still experimental while parity testing continues.
- Conversion/export internals remain active platform work; public users should consume the published OpenMed MLX artifacts.

## Validation

Release-prep validation completed on April 24, 2026:

```bash
python -m pytest tests/unit/mlx/test_mlx_inference.py tests/unit/mlx/test_privacy_filter_mlx.py tests/unit/test_pii.py tests/unit/test_pii_entity_merger.py
cd swift/OpenMedKit && swift test
xcodebuild -project swift/OpenMedDemo/OpenMedDemo.xcodeproj -scheme OpenMedDemo -destination 'generic/platform=iOS' build
xcodebuild -project swift/OpenMedScanDemo/OpenMedScanDemo.xcodeproj -scheme OpenMedScanDemo -destination 'generic/platform=iOS' build
```

Results:

- Python targeted tests: `143 passed, 1 skipped`
- Swift OpenMedKit tests: `47 passed, 12 skipped`
- OpenMedDemo generic iOS build: passed
- OpenMedScanDemo generic iOS build: passed

The Swift skips are the existing SwiftPM CLI guard for real MLX runtime resources; physical-device MLX smoke testing remains the acceptance path for those gated artifact tests.
