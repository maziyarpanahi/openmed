# Machine-readable CLI contract

OpenMed's scriptable CLI returns one JSON document when a command is run with
`--json`. The envelope and process status are intended for automation, so the
command path, exit code, and error code must remain stable across releases.

## Contract fixtures

The local fixture gate covers the representative outcomes below. The fixtures
contain aggregate metadata only; they do not include command input, file paths,
exception details, or record-level values.

| Outcome | Command path | Exit code | Error code |
| --- | --- | ---: | --- |
| Success | `models list` | `0` | — |
| Validation failure | `risk discover` | `2` | `invalid_discovery_config` |
| Offline failure | `models pull` | `1` | `offline_unavailable` |
| Privacy-policy failure | `risk assess` | `1` | `release_policy_failed` |

The canonical definitions live in
`openmed/cli/contract.py`. They are rendered through the shared output helpers,
not through a model loader, filesystem input, or network service.

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

`message` is safe for logs and automation. It must not echo command input, raw
identifiers, local paths, exception text, or model responses. A contract
fixture should be changed only when the public automation contract is
intentionally changed.

## Exit codes

- `0` means the command completed successfully.
- `1` means a runtime, offline, or privacy-policy gate failed.
- `2` means command validation or usage failed.

The gate is deterministic and offline. Run it with:

```shell
.venv/bin/python -m pytest tests/unit/cli/test_contract.py -q
```

These fixtures are a scripting contract, not a compliance certification or a
clinical decision guarantee.
