---
name: setup-openmed
description: "Collect a bounded set of de-identification policy decisions and write a deterministic, reviewable DEID-POLICY.md from the versioned local template. Use when a project needs explicit jurisdiction, recall floor, surrogate strategy, model policy, audit location, and human approval before privacy work begins."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: adjacent
  version: "1.0"
---

# Set up an OpenMed de-identification policy

Use this skill before a de-identification pipeline when the privacy decisions
are not already recorded. It creates a small policy document; it does not
inspect, transform, upload, or retain clinical data. The setup is local-first,
deterministic, and has **no mandatory network call**.

The output is a draft configuration contract, not a compliance certification,
legal opinion, or guarantee of de-identification. A human must review and
approve the draft before it controls a run.

## When to use this skill

Use it when a project needs to turn implicit privacy choices into a reviewable
`DEID-POLICY.md` file. Use the focused de-identification skills afterward to
apply or audit the selected policy.

Do not use this skill to collect a note, dataset row, identifier, model output,
secret, credential, or any other source payload. The setup questions accept
policy choices only. Keep logs, exceptions, reports, and fixtures free of raw
sensitive values.

## Bounded decision contract

Collect these five decisions, one at a time. Do not invent a sixth field or
silently choose a value. The value written to the artifact must be the
lowercase canonical value in the right-hand column.

| Decision | Bounded choices (canonical value) |
| --- | --- |
| Jurisdiction or operating context | `us`, `eu`, `canada`, `research`, `organization-defined` |
| Recall floor | `0.90`, `0.95`, `0.99` |
| Surrogate strategy | `mask`, `remove`, `replace`, `hash` |
| Model policy | `local-preinstalled`, `local-user-supplied`, `rules-only` |
| Audit location | `separate-local-directory`, `controlled-artifact-store`, `no-retention` |

The jurisdiction choices are context labels, not findings that a law applies.
For example, `us` may be used for a US/HIPAA-context workflow and `eu` for an
EU/GDPR-context workflow, but the generated document must not claim that the
workflow is compliant. `research` and `organization-defined` require the
project's own governance review.

The recall floor is a release target, not an observed score. Do not write a
metric, benchmark result, dataset name, or model claim into this artifact.
`replace` must use deterministic, synthetic surrogates when it is later
implemented; any re-identification mapping remains a separately protected
secret. `hash` is one-way for this policy document, but hashes can still be
sensitive linkage material.

## Setup workflow

Follow this order exactly:

1. Explain that the result is a draft configuration and that human approval is
   required. Ask only for the five decisions in the table.
2. Normalize each answer for comparison by trimming surrounding whitespace,
   folding case, and treating spaces or underscores as hyphens. Match the
   normalized answer against the bounded choices exactly. Do not accept a
   free-form value, and do not infer a jurisdiction from prose.
3. If an answer is missing or invalid, stop before writing the artifact. Report
   only the field name and the allowed canonical choices; never echo the
   answer, a source value, or exception text.
4. Read the local [DEID-POLICY.template.md](assets/DEID-POLICY.template.md)
   and replace only its five decision placeholders:

   - `{{ jurisdiction }}`
   - `{{ recall_floor }}`
   - `{{ surrogate_strategy }}`
   - `{{ model_policy }}`
   - `{{ audit_location }}`

   Preserve the template version, section order, checkboxes, and line endings.
   Do not add a timestamp, random identifier, machine path, user identity,
   source text, detected span, model output, or free-form rationale.
5. Write the result as `DEID-POLICY.md` in the requested project directory.
   If a file already exists, ask for explicit permission before replacing it;
   never overwrite it implicitly.
6. Report only that the draft path was written and that review is pending.
   Do not print the collected answers or any source payload.

The same five canonical choices and the same template version must produce
byte-for-byte identical output. Do not use the current time, environment
variables, network responses, or machine-specific paths in the artifact.

## Local-only model rule

This setup does not download a model or call a hosted service. With
`local-preinstalled`, stop and ask the project owner to install or provide the
approved local model if it is absent. With `local-user-supplied`, record no
secret or personal path in `DEID-POLICY.md`; the caller supplies the model
outside the artifact. `rules-only` must remain deterministic and local. A
later pipeline may have its own explicitly approved setup step, but it is not
part of this skill and must never be mandatory here.

## Human approval gate

The template always writes `DRAFT — HUMAN APPROVAL REQUIRED`. Stop after the
draft is written. A human reviewer must inspect the five choices, verify that
the intended local model and audit handling exist, and explicitly change the
status to approved through the project's review process. The setup workflow
must not self-approve, sign, certify, or claim a regulatory outcome.

## Handoff

- Apply the selected profile with
  [`configuring-privacy-policies`](../configuring-privacy-policies/SKILL.md).
- Transform text with
  [`deidentifying-clinical-text`](../deidentifying-clinical-text/SKILL.md).
- Check coverage with
  [`auditing-deidentification-runs`](../auditing-deidentification-runs/SKILL.md).

Those skills inherit this artifact's guardrails. Keep any audit report limited
to offsets, hashes, provenance, counts, and risk summaries; never put raw
identifiers into logs, exceptions, reports, fixtures, or the policy file.
