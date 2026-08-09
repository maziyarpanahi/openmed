# Offline install smoke check

`smoke_check.py` verifies a completed OpenMed installation without downloading
packages, models, or other runtime data. Run it from a clean environment after
installing the package:

```bash
python scripts/install/smoke_check.py > install-smoke.json
```

The command uses the selected Python environment's `openmed` entry point and
checks three things:

- `openmed --version` starts successfully and matches the installed package.
- `openmed models validate --json` can read the bundled model manifest.
- A synthetic, in-memory redaction preview is deterministic and privacy-safe.

The child processes receive `OPENMED_OFFLINE=1` together with the Hugging Face
and Transformers offline flags. They run with a temporary home, cache, and
configuration directory, and their stdout/stderr are never copied into the
report. The report contains only stable statuses, counts, the package version,
and SHA-256 hashes of synthetic surfaces. A successful run exits `0`; any
failed check exits `1`.

Use another installed environment explicitly when needed:

```bash
python scripts/install/smoke_check.py --python /path/to/venv/bin/python
```

This is an install/runtime evidence check, not a compliance certification or a
clinical decision guarantee.
