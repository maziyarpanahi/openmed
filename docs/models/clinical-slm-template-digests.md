# Clinical SLM prompt-template digests

`openmed.models.clinical_slm_templates` binds the three prompt surfaces of a
clinical small-language-model run: the system instruction, task instruction,
and output-format contract. Each template is canonicalized locally and hashed
with SHA-256 before runtime values are rendered.

The module is standard-library-only and performs no network I/O. The template
set keeps canonical text only for the immediate local render. Its provenance
report contains the schema version, three template digests, the aggregate
template-set digest, and declared placeholder names; it does not contain
template text, runtime substitution values, model output, or clinical input.

## Build a template contract

Use simple named placeholders. Literal braces in an output-format example must
be doubled because rendering uses Python's format-string rules.

```python
from openmed.models.clinical_slm_templates import ClinicalSLMTemplateSet

templates = ClinicalSLMTemplateSet(
    system="You are a local clinical review assistant.",
    task="Summarize the de-identified note: {clinical_note}.",
    output_format='Return JSON with keys {{"summary", "review_required"}}.',
)

provenance = templates.provenance_report
print(provenance["template_digests"]["task"])
print(provenance["template_set_digest"])
```

Only the canonical text is hashed. NFC Unicode normalization and CRLF/CR to LF
line-ending normalization make equivalent local files produce the same digest;
meaningful whitespace remains part of the template contract. A change to any
system, task, or output-format template changes its individual digest and the
aggregate digest.

## Render with a fail-closed substitution boundary

Runtime substitutions must exactly match the placeholders declared by the
templates. Missing and extra values are rejected before rendering. Rejected
errors use stable reason codes and do not echo the rejected value.

```python
rendered = templates.render({"clinical_note": "synthetic de-identified note"})

# Pass these values to the local runtime only. Keep them out of logs and
# serialized evidence; persist the value-free report instead.
local_messages = rendered.messages()
run_evidence = {"prompt_templates": rendered.provenance_report}
```

An undeclared value fails with
`UndeclaredTemplateSubstitutionError`. A caller cannot silently add a new
runtime prompt field without changing the template contract and its digest.
The optional `declared_substitutions` constructor argument can make the
placeholder contract explicit; its names must match the placeholders exactly.

## Verify replay provenance

Persist the value-free report with the guarded run and verify it before replay
or comparison:

```python
from openmed.models.clinical_slm_templates import verify_template_provenance

verify_template_provenance(templates, run_evidence["prompt_templates"])
```

This check compares all three individual digests, the aggregate digest, and
the declared substitutions. A stale or altered record raises
`ClinicalSLMTemplateDigestMismatchError`. The digest is reproducibility
metadata, not a compliance certification or an autonomous clinical decision
guarantee; human review requirements for the surrounding clinical workflow
still apply.

Collections and placeholder declarations are bounded to 4,096 entries. Each
template or runtime value is limited to 1 MiB UTF-8; each rendered prompt is
limited to 4 MiB, checked before concatenation. Invalid Unicode, conflicting
factory aliases, and conflicting provenance wrappers fail closed. Typed records
are rebuilt at render/replay boundaries, so changed text cannot retain an old
cached digest. The aggregate digest is checked against its individual digests
and declarations even when provenance is parsed without template text.

Flat declaration lists name the union of all placeholders; per-template maps
must match each template separately. Placeholder names are public schema
metadata: use generic names, never patient names or identifiers. Hashes are not
anonymization or signatures, and template digests do not bind runtime values.
Rendered prompts remain sensitive in-memory data; the template set retains its
canonical strings for its lifetime, and this API does not promise secure memory
erasure. Upstream exception context is discarded.
