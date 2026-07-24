# OpenMedKit

OpenMedKit is the Swift package for running OpenMed models in iOS, iPadOS, and
macOS apps with MLX or CoreML backends.

## Local-First Guarantees

- Inference is on-device after model and tokenizer assets are loaded.
- `analyzeText(...)`, `extractPII(...)`, and `extractPIIChunked(...)` do not
  perform network I/O.
- Telemetry is off by default. The package does not send usage, prompts,
  clinical text, entities, or identifiers to OpenMed or third parties.
- PHI should stay on device. Public download helpers may fetch model artifacts,
  but they are explicit model-preparation APIs and are not required for
  inference when assets are bundled or cached locally.

## No-PHI Logging

OpenMedKit routes internal inference diagnostics through `SafeLog`, a structured
logging shim that records operation names, counts, labels, offsets, confidence,
and span hashes rather than raw matched text. `EntityPrediction.description` is
also PHI-safe, so printing an entity emits offsets and a SHA-256 span hash
instead of the original identifier text.

## Clinical Safety

OpenMedKit is not a medical device. It is a developer library for local clinical
NLP and PII detection. Applications using it remain responsible for clinical
validation, user-facing safety controls, review workflows, and regulatory
classification.
