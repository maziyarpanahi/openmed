# Offline install smoke check

`smoke_check.py` verifies a completed OpenMed installation without downloading
packages, models, or other runtime data. Run it from a clean environment after
installing the package:

```bash
python scripts/install/smoke_check.py > install-smoke.json
```

The command uses only the `openmed` entry point installed beside the selected
Python interpreter (it does not fall back to another `openmed` on `PATH`) and
checks three things:

- `openmed --version` starts successfully and matches the installed package
  metadata.
- `openmed models validate --json` can read the bundled model manifest.
- A synthetic, in-memory redaction preview is deterministic and privacy-safe.

The child processes receive `OPENMED_OFFLINE=1` together with the Hugging Face
and Transformers offline flags. They run with a temporary home, cache, and
configuration directory and a search path restricted to the selected Python
environment. Their stdout/stderr are never copied into the report. The report
contains only stable statuses, counts, a metadata-confirmed package version,
and SHA-256 hashes of synthetic surfaces. A successful run exits `0`; any
failed check exits `1`.

Use another installed environment explicitly when needed:

```bash
python scripts/install/smoke_check.py --python /path/to/venv/bin/python
```

This is an install/runtime evidence check, not a compliance certification or a
clinical decision guarantee.

## Clinical brief installation profiles

Add `--brief-profile core` for a **minimal, non-editable** installation, or
`--brief-profile adapters` for an installation with exactly the declared
`[cli,service,mcp]` extras. Neither profile installs MLX, PyTorch, Transformers,
model weights or training data. The profiles intentionally reject environments
containing those runtimes: they verify the missing-runtime contract separately
from model qualification.

Both profiles exercise the Python brief composer, reviewed synthetic provider
injection, the installed CLI entry point, private summary/audit separation,
exclusive output creation, typed missing-runtime errors and full-chain privacy
sentinels. The adapter profile additionally exercises REST, the Python REST
client and the read-only MCP adapter through in-memory transports. Test scores
are synthetic fixtures, not calibrated model evidence or clinical validation.

Probes run under `python -I` in a temporary directory outside the checkout.
They reject editable installs, check imported module locations, and require
the repository root to be absent from `sys.path`. Python outbound connection
and DNS calls are blocked during application operations, with an empty cache
and offline flags. Windows event-loop self-pipe setup happens before the guard;
no application listener is started. These guards are test controls, not an
OS-level security sandbox for untrusted native code.

Resource checks load the packaged review-packet schema, thresholds and synthetic
grounding vocabulary using `importlib.resources`. Unix outputs must have mode
`0600`; Windows checks readable/writable regular files and exclusive-create
behavior, **not** Unix permission bits or unverified Windows ACL guarantees.

To reproduce the complete CI lane from the checkout:

```bash
uv build --out-dir /path/to/artifacts
uv export --frozen --no-hashes --no-emit-project --extra cli --extra service --extra mcp --output-file /path/to/constraints.txt
python scripts/install/smoke_check.py --artifacts /path/to/artifacts --constraints /path/to/constraints.txt --report /path/to/smoke.json
```

The artifact directory must contain exactly one wheel and one sdist. This
**explicit artifact mode** uses `uv` to install package/build dependencies in
fresh disposable environments; dependency preparation may use the network.
Runtime probes do not. Each artifact is tested with core and adapter installs,
then a disposable copy with the required schema deliberately omitted is
installed. That negative control must fail specifically at resource resolution.
The original artifacts and existing environments are never modified. Installer
output, traceback contents and local paths are not relayed to reports; failures
use fixed codes. A successful lane has six passing rows, including two controls.
