# Local extensions to the status vocabulary

`openmed.clinical.status_vocab` ships deterministic cue tables for three SDOH
domains: substance use, employment, and living situation. Adding a fourth
domain does not require bundling new terminology into the package or changing
its defaults. `load_status_vocab(path)` loads and validates any vocabulary
from an explicit local path, so a contributor-owned domain can live entirely
outside the repository.

This page walks through
[`examples/custom_status_vocabulary.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/custom_status_vocabulary.py),
which adds a synthetic "mobility" domain and demonstrates the fail-closed
checks a local vocabulary should pass before it is trusted.

## Load a local vocabulary

```python
from openmed.clinical import load_status_vocab

payload = load_status_vocab("custom_status_vocab.yaml")
mobility = payload["vocabularies"]["mobility"]
```

`load_status_vocab` enforces the same structure as the bundled tables:
a `schema_version`, `provenance.source` and an advisory `provenance.disclaimer`
mentioning "clinical decision", a `defaults.unknown_status`, and for each
vocabulary a `priority` order, a `statuses` mapping with `cues`, a
`current_statuses` list, and `axis_overrides` for `negated` and
`historical_current`. A vocabulary missing its advisory disclaimer is
rejected:

```python
try:
    load_status_vocab("invalid_provenance.yaml")
except ValueError as error:
    print(error)  # "... requires an advisory disclaimer"
```

## Guard against duplicate cues

Matching resolves ties by `priority` order, so a cue accidentally listed
under two statuses would be ambiguous. `load_status_vocab` rejects cues that
collide after the same Unicode, case, and whitespace normalization used for
matching. The error identifies the conflicting statuses and cue positions
without echoing the cue text:

```python
load_status_vocab("duplicate_cues.yaml")
# ValueError: mobility vocabulary lists one normalized cue under two statuses
```

## Normalize against the local domain

The bundled `normalize_substance_status`, `normalize_employment_status`, and
`normalize_living_status` helpers always load the packaged vocabulary, so
they cannot target a locally extended domain. The example re-implements the
same deterministic, case-insensitive substring match documented on
`openmed.clinical.status_vocab` as `normalize_mobility_status()`, so the
pattern is copy-pasteable without depending on that module's private
helpers:

```python
from examples.custom_status_vocabulary import normalize_mobility_status

normalize_mobility_status("uses a cane", mobility)  # "assisted"
normalize_mobility_status("uses a cane", mobility, negated=True)  # "never"
normalize_mobility_status(
    "walks independently", mobility, temporality="historical"
)  # "former"
```

## Scope

This example is offline and synthetic: every cue is a generic English phrase
authored for this walkthrough, not a bundled or restricted terminology. It
demonstrates local validation and normalization only; it does not recommend
a clinical decision from a normalized status, and it does not change any
package default. See
[`tests/unit/examples/test_custom_status_vocabulary.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/examples/test_custom_status_vocabulary.py)
for the runnable checks behind this page.
