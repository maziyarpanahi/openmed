# Clinical NLI checkpoint preparation

This optional research recipe prepares #3236 without a library-release milestone.
Training and publishing a new clinical NLI checkpoint are not prerequisites for
feature APIs or SDK releases. Existing independently qualified local models or
caller-supplied calibrated providers can satisfy the runtime contract. Missing
or unqualified providers still fail closed. No checkpoint, registry alias, or
quality result is released by this document.

The configuration at `openmed/training/configs/clinical_nli_small.yaml` starts from the Apache-2.0 ClinicalE5-Small encoder at immutable revision `2ea947130eb28d610ceb96472af0b694fce9c344`. Its token-classification head must be discarded and replaced with a three-class sequence head. The configured seed, split, optimizer, and export formats are proposed starting values, not evidence of a training run.

Before any training, record a public-data manifest with dataset URLs, immutable revisions, license review, content hashes, duplicate checks, and source-grouped splits. Synthetic examples must be clearly tagged. MedNLI is user-supplied and evaluation-only; no MedNLI data or derived examples may enter training, the repository, or a published artifact. Patient data is excluded.

A candidate must report public and synthetic three-class accuracy separately, negation false-entailment rate, error slices, and calibrated abstention. The PyTorch, ONNX int8, and MLX exports need pinned checksums, offline load tests, and prediction parity. `BenchmarkReport` and gate results must contain only counts, rates, hashes, model IDs, and revision IDs. The model card must explain the three-way-plus-abstain contract and evidence limits. A MedNLI number, if the owner has lawful local access, is one evaluation-only aggregate line.

`openmed.eval.nli_gate.build_nli_preparation_report` composes the existing error-slice, calibration, and negation reports with public and synthetic accuracy counts. Its `BenchmarkReport` retains digests and aggregate metrics only; its publication `GateCheck` always fails. The synthetic unit test exercises composition and privacy without presenting mock results as model evidence.

The model-promotion gate remains closed until artifacts, license, provenance,
independent evaluation and the immutable public revision are verified. This
does not block the library/SDK release. Adding a `models.jsonl` default alias
before those checks would make an unvalidated checkpoint appear shipped.

Paid compute and public model publication require a separate owner decision. The cost plan must specify hardware, provider rate, estimated training and export hours, storage and egress, a spending ceiling, and evidence from a small local pilot. No paid work is configured or started here.
