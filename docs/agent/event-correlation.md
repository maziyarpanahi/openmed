# Agent Event Correlation

`openmed.agent.correlation` provides opaque identifiers for correlating local
agent runs and actions without embedding patient, user, workflow, or tool
content.

## Identifier Format

- Run identifiers use `run_` followed by 32 lowercase hexadecimal characters.
- Action identifiers use `act_` followed by 32 lowercase hexadecimal characters.
- The hexadecimal token is exactly 16 bytes (128 bits).
- Parsing is canonical: uppercase, shortened, extended, malformed, and
  wrong-kind identifiers are rejected instead of normalized.

Generate identifiers without supplying source content:

```python
from openmed.agent import ActionCorrelation, ActionId, RunId

run_id = RunId.generate()
parent_id = ActionId.generate()
action_id = ActionId.generate()

correlation = ActionCorrelation(
    run_id=run_id,
    action_id=action_id,
    parent_action_id=parent_id,
)

payload = correlation.to_json()
```

`ActionCorrelation` requires a `RunId` for `run_id` and `ActionId` values for
both action fields. An action cannot identify itself as its parent. Its compact
JSON representation has deterministic key ordering and contains only the schema
version and opaque identifiers.

## Privacy Boundary

Runtime generation uses `secrets.token_bytes`; the optional `token_source`
argument exists only for deterministic offline tests. Do not derive an
identifier by hashing or encoding prompts, clinical text, filenames, user IDs,
tool arguments, model output, or other workflow content. Parsing is for
previously generated opaque identifiers, not for turning business data into an
identifier.

Malformed values fail with a public field name and stable error code. Rejected
values are not included in exception messages or object representations. The
module does not persist events, implement distributed tracing, or encode patient
or operator identity.
