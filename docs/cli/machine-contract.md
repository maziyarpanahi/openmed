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

## Guarded summary and NLI commands

`summarize` and `nli verify` use the same success/error envelopes above. Their
fixed error message is `Local clinical request could not complete.`; it never
includes clinical text, paths, backend names supplied by a caller or exception
details. Use `--json` on the leaf command for stable error codes.

| Outcome | Exit | Stable codes |
| --- | ---: | --- |
| Successful summary files or all claims entailed | `0` | None |
| Valid NLI contradiction, neutral or abstention | `1` | No error; `ok: true` with per-claim evidence |
| Backend/cache/dependency unavailable | `1` | `summary_backend_unavailable`, `nli_backend_unavailable` |
| Privacy or ordering gate refusal | `1` | `summary_leakage_rejected`, `summary_deidentification_rejected` |
| Output reservation/write failure | `1` | `summary_output_failed` |
| Invalid backend result or processing failure | `1` | `summary_result_invalid`, `summary_failed`, `nli_result_invalid`, `nli_failed` |
| Invalid command arguments | `2` | `summary_arguments_invalid`, `nli_arguments_invalid` |
| Unsupported mode or local alias | `2` | `summary_mode_unsupported`, `summary_model_invalid`, `nli_backend_invalid`, `nli_backend_factory_invalid` |
| Invalid, unavailable, special or oversized local input | `2` | `summary_input_invalid`, `summary_input_unavailable`, `summary_input_not_regular`, `summary_input_too_large`; corresponding `nli_input_*` codes |
| Invalid claim array or per-claim bounds | `2` | `nli_claims_invalid` |

The summary console/file metadata contains only `mode`, `backend_id`,
`template_digest`, `leakage_check`, `summary_characters` and
`human_review_required`. The protected summary goes to its separate explicit
new destination. NLI data contains only a `claims` array with each row's
`claim_index`, `label`, `score`, `backend_id`, `contradicted` and
`review_required`. Plain console output also omits clinical values.

Summary/source inputs are limited to 16 KiB, protected summary output to
8 KiB, and the claims file to 64 KiB with 1–128 claims of at most 4 KiB each.
These commands use existing native APIs with local-only defaults and never
download models. Their guards do not qualify a model artifact or validate a
clinical workflow. See [summarization](../clinical/summarization.md#guarded-local-cli)
and [NLI verification](../clinical/nli-verification.md#local-cli-verification)
for provisioning and trusted-code boundaries.
