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

The adapter profile then checks every service transport against one shared
service runtime whose model loader is a synthetic stand-in. The stand-in tags
only a synthetic marker and never touches weights, tokenizers, caches or the
network:

| Check | What it proves |
| --- | --- |
| `rest_pii_extract` | `POST /pii/extract` returns the expected label and offsets, and an unserved model is refused with `model_not_served` without echoing input. |
| `graphql_query` | One `deidentify` query on `/graphql` returns the masked synthetic text with no errors. |
| `grpc_stubs` | The generated `openmed.service.proto.generated` stubs are packaged files, import from the installed tree and describe `openmed.service.v1.OpenMedService`. |
| `grpc_loopback` | A gRPC server bound explicitly to `127.0.0.1` on an ephemeral port serves one `Extract` RPC that matches the REST offsets, and refuses an unserved model with `INVALID_ARGUMENT`. |

Probes run under `python -I` in a temporary directory outside the checkout.
They reject editable installs, check imported module locations, and require
the repository root to be absent from `sys.path`. Python outbound connection
and DNS calls are blocked during application operations, with an empty cache
and offline flags. Windows event-loop self-pipe setup happens before the guard.
REST and GraphQL run in memory; the only listener is the adapter profile's
loopback gRPC server, which is stopped before the probe ends. These guards are
test controls, not an OS-level security sandbox for untrusted native code.

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
then with two negative controls, each installed from a disposable damaged copy:

| Control | Omitted from the copy | Must fail only at |
| --- | --- | --- |
| `missing_resource` | the review-packet schema (core install) | `packaged_resources` |
| `missing_proto_stubs` | every generated protobuf stub (adapter install) | `grpc_stubs` |

A control passes only when exactly that check fails, and it refuses to run
against an artifact that already lacked the resource. The original artifacts
and existing environments are never modified. Each row records the artifact
type, profile, the original artifact's SHA-256 digest and fixed pass or fail
codes; install rows also list each check, including the metadata-confirmed
package version and synthetic-surface hashes. Installer output, request and
response bodies, traceback contents and local paths are not relayed to
reports. A successful lane has eight passing rows, including four controls.
