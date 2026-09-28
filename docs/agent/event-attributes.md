# Agent Event Attributes

`validate_event_attributes` checks one event's attributes against a closed
allowlist. It is a shared contract for event producers, not an event store, a
policy engine, or an observability backend.

```python
from openmed.agent.event_attributes import EventAttributes

attributes = EventAttributes.from_mapping(
    {
        "run_id": "run_0123456789abcdef0123456789abcdef",
        "execution_stage": "tool_call",
        "outcome_class": "success",
        "outcome_reason": "completed",
        "input_digest": "sha256:" + "0" * 64,
        "duration_ms": 12.5,
        "retryable": False,
    }
)
print(attributes.to_json())
```

## The allowlist

Every accepted name has one declared kind, and there is no free-text field at
all — so a prompt, tool argument, model output, path, or clinical note has
nowhere to go.

| Kind | Attributes | Accepted value |
| --- | --- | --- |
| Correlation | `run_id`, `action_id`, `parent_action_id` | Opaque `run_`/`act_` identifiers from [event correlation](event-correlation.md); the kinds are not interchangeable. |
| Governance | `capability_id`, `policy_id`, `purpose_id`, `tool_id`, `workflow_id` | Canonical [governance identifiers](governance-identifiers.md) of that exact kind. |
| Stage | `execution_stage` | One of `planned`, `authorized`, `started`, `tool_call`, `completed`, `failed`, `aborted`. |
| Outcome | `outcome_class`, `outcome_reason` | An [outcome class](outcome-reasons.md) and a reason code allowed for *that* class. |
| Digest | `input_digest`, `output_digest`, `artifact_digest` | Lowercase `sha256:<64 hex>`. |
| Count | `sequence_number`, `attempt_number`, `retry_count`, `tool_call_count`, `artifact_count` | Integer in `0..MAX_COUNT_VALUE`. |
| Duration | `duration_ms` | Finite integer or float in `0..MAX_DURATION_MS`. |
| Flag | `retryable`, `redacted` | A real `bool`. |

`outcome_reason` is cross-checked against the `outcome_class` in the same
mapping and fails closed when that class is absent. Counts reject booleans
because `type(value) is int` is checked rather than `isinstance`, and
`duration_ms` rejects NaN and both infinities.

A `schema_version` key equal to `EVENT_ATTRIBUTES_SCHEMA_VERSION` is accepted
so a serialized payload round trips; it is not stored as an attribute.

## Refusals

`EventAttributeError` carries a stable `.code` and a `.field_name` that is
only ever an allowlisted name. An unknown or sensitive-looking key is reported
with `field_name=None`, because naming it would echo the submitted key:

- `unknown_attribute` — the key is not on the allowlist.
- `sensitive_attribute_key` — the key is not on the allowlist *and* looks like
  a credential, prompt, path, URL, message, or direct identifier. It is a
  separate code so a producer sees why the contract exists, not just that it
  failed.

Other codes are `not_a_mapping`, `invalid_key_type`, `too_many_attributes`,
`nested_value_not_allowed`, `invalid_identifier`, `unknown_execution_stage`,
`unknown_outcome_class`, `unknown_outcome_reason`, `outcome_class_required`,
`invalid_digest`, `invalid_count`, `count_out_of_range`, `invalid_duration`,
`non_finite_number`, `duration_out_of_range`, `invalid_flag`,
`invalid_schema_version`, `duplicate_attribute`, `malformed_json` and
`payload_too_large`.

Mappings, lists, tuples, sets and byte strings are rejected outright, so no
nested object can smuggle a payload under an allowlisted name. A rejected
identifier is not chained onto the raised error.

## Serialization

`EventAttributes` is frozen and its `values` mapping is read-only.
`to_dict()` puts `schema_version` first and the attributes in sorted key
order; `to_json()` sorts keys for byte-identical payloads. `from_json()`
bounds the payload at `MAX_ATTRIBUTE_JSON_BYTES` before parsing and rejects
duplicate keys, which JSON permits but the contract does not.

Persisting events, evaluating policy, and storing hashed copies of arbitrary
clinical payloads are out of scope. Valid attributes are a shape statement,
not a clinical or security approval.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/agent/test_event_attributes.py -q
```

Fixtures are synthetic. Tests validate every allowlisted attribute together,
every execution stage, every outcome class against its own reason codes, and
cover unknown, sensitive, duplicate, nested, oversized and non-finite cases,
including sentinel prompts, bearer values and paths that must never appear in
output or exceptions.
