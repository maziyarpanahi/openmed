# Synthetic governed-agent trace fixtures

`openmed.eval.agent_trace_fixtures` provides a shared offline fixture pack for
testing governed-agent outcomes without storing or reconstructing clinical
content. The bundled JSONL file covers:

- read-only allow and minimum-data projection success;
- missing-capability abstention;
- purpose-mismatch and expired-consent policy denial;
- human-review handoff; and
- a bounded timeout failure.

Each line uses schema `openmed.eval.agent_trace_fixture.v1` and contains only a
stable synthetic case/scenario identifier, opaque run and action identifiers,
a SHA-256 trace digest, and an expected outcome/reason pair from the closed
agent outcome vocabulary. The digest binds omitted synthetic trace content; it
does not expose that content or prove that a runtime action was authorized.

```python
from openmed.eval.agent_trace_fixtures import (
    load_governed_agent_trace_fixtures,
)

fixtures = load_governed_agent_trace_fixtures()
for fixture in fixtures:
    assert fixture.trace_digest.startswith("sha256:")
```

The loader preserves JSONL order, rejects duplicate case and action IDs, and
fails closed on unknown or missing fields, duplicate JSON fields, malformed
identifiers or digests, unknown scenarios, and undeclared outcome/reason
combinations. Errors contain stable codes, public field names, and line numbers
only; rejected values are not echoed.

This fixture pack does not run an agent or model, call an EHR, authorize a tool,
or represent production traces. Do not add names, chart text, credentials,
tool arguments, prompts, outputs, or other clinical content. Add a new scenario
to the closed enum, the JSONL pack, and the focused coverage assertion together.

Run the offline checks with:

```text
.venv/bin/python -m pytest tests/unit/eval/test_agent_trace_fixtures.py -q
```
