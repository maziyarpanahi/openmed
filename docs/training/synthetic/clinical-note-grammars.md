# Composable synthetic clinical note grammars

`openmed.training.synthetic.note_grammar` renders encounter, discharge, referral,
operative and nursing note grammars using explicitly synthetic facts. Each
occurrence has exact gold spans, canonical labels, stable fact references, roles
(identifier, clinical fact, distractor), source provenance and section bounds.
Repeated facts reuse the same identity and value rather than drifting between
sections. These are structure/provenance fixtures, not clinically validated notes.

`synthetic_note_facts(seed=...)` reuses the existing English locale surrogate
provider for identity fields and adds repository-authored fictional clinical
placeholders. `render_synthetic_note(type, facts, omitted_sections=...)` composes
only supplied facts. Omitted or empty sections stay absent; no missing diagnosis
or medication is invented. A dose cannot exist without a medication, unknown
fact slots reject, and duplicate facts cannot disagree silently.

Select the grammar's owned fact slots explicitly. The helper returns a shared
superset, including medication and dose; an operative grammar rejects slots it
does not own rather than silently discarding them.

```python
from openmed.training.synthetic.note_grammar import (
    GRAMMARS, render_synthetic_note, synthetic_note_facts, validate_synthetic_note,
)

kind = "operative"
allowed = {key for _section, keys in GRAMMARS[kind] for key in keys}
facts = tuple(fact for fact in synthetic_note_facts(seed=42) if fact.key in allowed)
note = render_synthetic_note(kind, facts)
assert validate_synthetic_note(note) == ()
```

Use `validate_synthetic_note` independently to detect source offset, canonical
label, fact identity, role, source or section-binding corruption. Data carries
explicit synthetic markers. All source text is synthetic; do not supply private
patient facts or restricted corpora. Note generation requires no model or network
and writes no files. Rendering is bounded to 128 facts/omission entries and each
fact value to 512 characters. The templates make no medical plausibility,
clinical effectiveness, de-identification release or diagnostic assurance claim.
