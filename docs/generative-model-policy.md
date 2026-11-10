# Generative Model Policy

OpenMed is small-extractor-first and generative-second. Deterministic detectors,
trained span models, checksum validators, and release gates do the privacy-
critical work. A generative model may assist only around those controls.

## Encouraged Uses

- Synthetic data augmentation for locale examples, social-history sections, and
  OCR/DICOM text overlays.
- Teacher or weak labeling when labels are accepted through inter-model
  agreement or human adjudication.
- Active-learning triage from gate failures, low-agreement spans, and hard
  negatives.
- Error analysis that clusters missed spans and explains recurring leakage
  patterns.
- Post-de-identification summarization after the de-identification and leakage
  checks have already passed.
- Policy drafting support for profile text, gate reports, and review checklists.

## Prohibited Uses

- Sending raw PHI to a remote generative service for redaction, extraction, or
  review.
- Using a generative model as the sole de-identification detector.
- Accepting schema-less extraction output without canonical labels, offsets, and
  provenance.
- Training only on synthetic examples without a real held-out evaluation gate.
- Running a summarizer before de-identification.
- Letting a model approve a release. Gates decide whether a candidate is
  `RELEASABLE`.

## Release Authority

The release gate is the authority for shipping. Owners are roles, not named
people:

- Model Lead: recipe, candidate checkpoints, distillation, and hard negatives.
- Privacy Eval Lead: leakage fixtures, benchmark reports, and signed gate
  evidence.
- Release Engineer: CI wiring, manifest state, channel promotion, and rollback.
- Device/Export Lead: tier benchmarks, quantization deltas, and packaged formats.

No model artifact or library increment moves to stable unless the applicable
gate report passes and the responsible role has reviewed the evidence.
