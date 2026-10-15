# Sealed evaluation CLI

`openmed eval sealed` runs a sealed evaluation of a signed candidate model
against a local holdout manifest and writes **only** the signed aggregate result
envelope to `--output`. It runs offline and delegates evaluation to an injected
sealed-evaluation service; case-level artifacts stay in the service's site-local
directory and are never printed.

```bash
openmed eval sealed --model <candidate-ref> \
  --holdout-manifest holdout.json --output result.json [--json]
```

## Exit codes and error codes

| Exit | Meaning | Error codes |
|---|---|---|
| 0 | Aggregate envelope written | – |
| 1 | Refused or failed, before evaluation where applicable | `candidate_unsigned`, `candidate_digest_mismatch`, `network_required`, `sealed_eval_preflight_failed`, `sealed_eval_failed`, `sealed_eval_service_unavailable`, `output_unwritable` |
| 2 | Usage error | `holdout_manifest_missing` |

Standard output and errors contain only controlled codes and messages: no case
text, spans, per-case scores, or paths. With `--json` the standard envelope is
used (`{"ok", "command", "data" | "error"}`).
