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
| Brief needs human review | `brief` | `0` | — |
| Brief refused | `brief` | `1` | — |
| Brief request/output failure | `brief` | `1` | `brief_failed` |
| Diagnostic check failed | `doctor` | `1` | — |
| FHIR validation check failed | `fhir validate` | `1` | — |

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

### Completed commands with negative results

`brief`, `doctor`, and `fhir validate` add a top-level `outcome` without changing
the existing `ok`, `command`, `data`, or `error` keys. `ok: true` means a command
produced its structured result, not that a clinical or validation gate passed.
Consumers should check `outcome` and the exit code:

| `outcome` | `ok` | Exit | Meaning |
| --- | --- | ---: | --- |
| `completed` | `true` | `0` | Command completed; a brief still requires human review |
| `refused` | `true` | `1` | Brief safely refused; no summary was produced |
| `check_failed` | `true` | `1` | Diagnostic or validation result includes failed checks |
| `failed` | `false` | `1` or `2` | The command could not produce a normal result |

Successful envelopes for other commands remain unchanged. A refusal is not an
exception or a clinical decision. Brief stdout contains only controlled status,
reason, character count and digest, never the source or generated summary.

## Exit codes

- `0` means the command completed successfully.
- `1` means a runtime, offline, or privacy-policy gate failed.
- `2` means command validation or usage failed.

Run the deterministic offline gate with
`.venv/bin/python -m pytest tests/unit/cli/test_contract.py -q`.

These fixtures are a scripting contract, not a compliance or clinical
guarantee.
