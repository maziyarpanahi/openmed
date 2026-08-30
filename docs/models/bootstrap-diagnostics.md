# Offline bootstrap diagnostics

`openmed.models.bootstrap_check` gives a local command a small, deterministic
readiness report before it starts model work. It is intended for cache
preflight and troubleshooting, not for clinical or compliance decisions.

The check performs no downloads and does not open sockets. It reads cache
directory metadata and, when a local integrity manifest is present, verifies
the manifest's listed files. A caller-controlled cache path can still refer to
a mounted or remote filesystem, so use a known-local path for an offline
preflight. The report never emits cache paths, model identifiers, checksums,
environment values, credentials, or model contents.

## Run the check

Run it directly from an environment with OpenMed installed:

```bash
python -m openmed.models.bootstrap_check --cache-dir "$HF_HUB_CACHE" --json
```

To check a particular local model or make a prerequisite mandatory, add the
corresponding option:

```bash
python -m openmed.models.bootstrap_check \
  --model-id OpenMed/example-model \
  --extra hf \
  --require-checksum \
  --require-offline
```

The command only returns non-zero for a failed readiness requirement or invalid
usage. The stable exit codes are:

| Code | Meaning |
| ---: | --- |
| `0` | All required checks are ready. |
| `1` | At least one required check is not ready. |
| `2` | Diagnostic input is invalid. |

## Report shape

The JSON report has a versioned top level and exactly four categories:

```json
{
  "schema_version": "openmed.bootstrap_diagnostics.v1",
  "ready": true,
  "exit_code": 0,
  "categories": {
    "cache": {"status": "pass", "reason": "snapshot_available"},
    "checksum": {"status": "warn", "reason": "checksum_unavailable"},
    "optional_extras": {"status": "pass", "reason": "no_required_extras"},
    "offline_policy": {"status": "pass", "reason": "offline_not_requested"}
  }
}
```

Category facts are limited to booleans, counts, and fixed labels. An absent
checksum manifest is a warning by default; `--require-checksum` turns it into a
readiness failure. Optional extras are informational unless one or more
`--extra` options make them required. `--require-offline` requires the OpenMed
and dependency-specific offline flags to be enabled. This proves that the
offline policy is configured; it does not claim that a socket guard is active
during the diagnostic itself. Run the protected model operation through
OpenMed's normal offline guard.

The result is a preflight signal. It does not prove that a model is suitable
for a clinical task, that a host is compliant, or that an application will
finish successfully after startup.
