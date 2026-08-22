# Machine-readable CLI contract

`--json` emits one document with stable command, exit-status, and error-code
fields.

## Contract fixtures

The offline gate covers the outcomes below. Fixtures omit inputs, paths,
exceptions, and record values.

| Outcome | Command path | Exit code | Error code |
| --- | --- | ---: | --- |
| Success | `models list` | `0` | — |
| Validation failure | `risk discover` | `2` | `invalid_discovery_config` |
| Offline failure | `models pull` | `1` | `offline_unavailable` |
| Privacy-policy failure | `risk assess` | `1` | `release_policy_failed` |

## JSON shape

Successful commands use the following top-level keys:

```json
{
  "ok": true,
  "command": "models list",
  "data": {"count": 0, "models": []}
}
```

Failures use the same command field and a stable error code/message pair:

```json
{
  "ok": false,
  "command": "models pull",
  "error": {
    "code": "offline_unavailable",
    "message": "The requested operation requires a local model cache."
  }
}
```

`message` never echoes input, identifiers, paths, exceptions, or model
responses. Change a fixture only for an intentional public-contract change.

## Exit codes

- `0` means the command completed successfully.
- `1` means a runtime, offline, or privacy-policy gate failed.
- `2` means command validation or usage failed.

Run the deterministic offline gate with
`.venv/bin/python -m pytest tests/unit/cli/test_contract.py -q`.

These fixtures are a scripting contract, not a compliance or clinical
guarantee.
