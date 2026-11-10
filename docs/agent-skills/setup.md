# De-identification policy setup

The [`setup-openmed` skill`](https://github.com/maziyarpanahi/openmed/blob/master/skills/setup-openmed/SKILL.md) turns five
privacy decisions into a reviewable `DEID-POLICY.md` draft. It is useful at the
start of a project, before a de-identification pipeline begins, when otherwise
implicit choices would make downstream work inconsistent.

The skill asks for a bounded choice for each of these fields:

| Field | Example choices |
| --- | --- |
| Jurisdiction or operating context | `us`, `eu`, `canada`, `research`, `organization-defined` |
| Recall floor | `0.90`, `0.95`, `0.99` |
| Surrogate strategy | `mask`, `remove`, `replace`, `hash` |
| Model policy | `local-preinstalled`, `local-user-supplied`, `rules-only` |
| Audit location | `separate-local-directory`, `controlled-artifact-store`, `no-retention` |

Answers are normalized to canonical values and inserted into the local
[`DEID-POLICY.template.md`](https://github.com/maziyarpanahi/openmed/blob/master/skills/setup-openmed/assets/DEID-POLICY.template.md).
The same choices and template version produce the same artifact. The setup
workflow does not inspect source text, accept a free-form payload, add
timestamps, or make a network call. It has no mandatory network call. Keep
logs, exceptions, reports, fixtures, and the policy artifact free of raw
sensitive values. The template schema and exact placeholder set are validated
before writing. Output is synced to a same-directory temporary file and
atomically replaced only after the target is rechecked; status output names
only `DEID-POLICY.md`, never its absolute or parent path.

The output always starts as `DRAFT — HUMAN APPROVAL REQUIRED`. A human must
review the choices, local model or rules path, recall evidence, surrogate or
linkage handling, and audit location before changing the status through the
project's review process. The document records configuration; it is not a
compliance certification, legal opinion, or guarantee.

After approval, hand the policy to the focused
[`configuring-privacy-policies`](https://github.com/maziyarpanahi/openmed/blob/master/skills/configuring-privacy-policies/SKILL.md)
and de-identification skills. Keep any later audit evidence limited to offsets,
hashes, provenance, counts, and risk summaries.
