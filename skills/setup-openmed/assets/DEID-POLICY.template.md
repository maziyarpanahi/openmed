<!--
Template: deid-policy
Template version: 1.0
Policy schema: 1

Replace only the five decision placeholders. Keep this header and do not add
source data, timestamps, machine paths, user identities, or model output.
-->

# De-identification policy

**Status:** `DRAFT — HUMAN APPROVAL REQUIRED`

This document records deterministic de-identification configuration choices for
human review. It is not a compliance certification, legal opinion, or guarantee
of de-identification, privacy, or regulatory compliance.

## Decision record

| Decision | Selected value |
| --- | --- |
| Jurisdiction or operating context | `{{ jurisdiction }}` |
| Recall floor | `{{ recall_floor }}` |
| Surrogate strategy | `{{ surrogate_strategy }}` |
| Model policy | `{{ model_policy }}` |
| Audit location | `{{ audit_location }}` |

The jurisdiction is a context label only. The recall floor is a release target,
not a measured result. No claim about a law, certification, model quality, or
release safety follows from any selected value.

## Operational requirements

- Process data only after a human has approved this draft through the project's
  review process.
- Keep inference local. Do not make a network call, download a model, or send
  source content to a hosted service as an implicit part of this policy.
- If the selected local model is unavailable, stop and request an explicit
  installation or user-supplied model; do not fall back to a remote service.
- Keep raw source values out of logs, exceptions, reports, fixtures, caches,
  and this policy. Audit evidence may contain offsets, hashes, provenance,
  counts, and risk summaries only.
- Treat mappings, keys, and linkage hashes as sensitive material and store
  them separately from de-identified output when the selected strategy needs
  them.
- Block release when the selected recall target has not been evaluated or when
  the required audit handling is unavailable. This is a project control, not a
  compliance verdict.

## Review checklist

- [ ] A human reviewed all five selected values.
- [ ] The local model or rules path is available and has no unapproved network
      fallback.
- [ ] The recall floor is backed by the project's approved evaluation evidence.
- [ ] The surrogate strategy and any mapping or linkage handling are understood.
- [ ] The audit location is available and contains no raw sensitive values.
- [ ] The reviewer explicitly approved this policy for the intended workflow.

**Human approval:** `PENDING`

**Approved reviewer role:** ____________________

**Approval reference:** ____________________

Approval metadata should identify the project's review record without adding
source values or other sensitive content to this file.
