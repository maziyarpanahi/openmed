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
- Routing PHI to Private Cloud Compute or to any other cloud model, including as
  a fallback for an on-device request.
- Letting a model approve a release. Gates decide whether a candidate is
  `RELEASABLE`.

## Apple Foundation Models

Apple Foundation Models are a borderline path, not a trusted control. They may
assist an OpenMedKit workflow only as a local, assistive step behind the
deterministic controls described above, under these rules:

- **On-device only.** Run the on-device model. Do not send PHI to Private Cloud
  Compute or to any other cloud model, and do not use a cloud model as a
  fallback when an on-device request is slow, unavailable, or rejected.
- **No tools.** Do not expose tool or function calling to the model in a PHI
  workflow. The host app performs retrieval, writes, and exports itself, after
  its own checks.
- **Borderline-tier output.** Treat generated text as a draft, not as evidence.
  Return it with the usual disclaimers, validate every consequential field
  against the source document, and never auto-trigger a diagnosis, treatment, or
  other clinical decision from the result.
- **Human review before use.** A reviewer must confirm the exact generated
  output before it leaves the workflow. The confirmation covers that output
  only; a new revision needs a new review.
- **Apple's acceptable-use requirements.** Review them before shipping an
  integration; Apple publishes them alongside the
  [Foundation Models framework documentation](https://developer.apple.com/documentation/foundationmodels).

The rest of the workflow is unchanged. De-identification and leakage checks run
before the model sees any text, the model never replaces a deterministic
control, and an OpenMed gate still decides whether an artifact is `RELEASABLE`.

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
