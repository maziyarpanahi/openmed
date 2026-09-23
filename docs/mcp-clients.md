# Connect MCP clients to OpenMed

OpenMed exposes its clinical NLP, PII extraction, and de-identification tools
through the Model Context Protocol (MCP). Use the `openmed-mcp` command for
local stdio clients or the Streamable HTTP transport for clients that connect
to a URL.

All examples on this page use synthetic text. Only send real PHI to an OpenMed
runtime that you operate and trust.

## Install and verify

Install OpenMed with the optional MCP dependency:

```bash
python -m pip install "openmed[mcp]"
openmed-mcp --version
```

For a checkout managed with `uv`:

```bash
uv sync --extra mcp
uv run openmed-mcp --version
```

## Run in a container

Build the dedicated MCP image from a checkout:

```bash
docker build -f Dockerfile.mcp -t openmed-mcp:local .
```

The image defaults to stdio. Keep stdin attached with `-i` and do not allocate
a TTY, because MCP protocol messages use stdin and stdout:

```bash
docker run -i --rm \
  -v openmed-hf-cache:/root/.cache/huggingface \
  openmed-mcp:local
```

A command-map client can launch that container directly:

```json
{
  "mcpServers": {
    "openmed-container": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v",
        "openmed-hf-cache:/root/.cache/huggingface",
        "openmed-mcp:local"
      ]
    }
  }
}
```

For a local Streamable HTTP endpoint, override the image's default command.
The server must listen on all container interfaces, while the published port
remains limited to host loopback:

```bash
docker run --rm \
  -p 127.0.0.1:8081:8081 \
  -v openmed-hf-cache:/root/.cache/huggingface \
  openmed-mcp:local \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8081 \
  --streamable-http-path /mcp
```

The equivalent Compose service includes the same loopback port binding, a
listener health check, and a persistent model cache:

```bash
docker compose up --build mcp
```

Connect remote-style local clients to `http://127.0.0.1:8081/mcp`. Set
`OPENMED_MCP_PORT` before starting Compose to select a different host port;
the container endpoint remains on port `8081`.

## Local stdio connections

Stdio is the recommended transport when the MCP client and OpenMed run on the
same machine. The client launches the server as a child process, so no TCP port
is exposed.

### JSON command-map clients

Use this shape for desktop and IDE clients that accept an `mcpServers` command
map:

```json
{
  "mcpServers": {
    "openmed": {
      "command": "uvx",
      "args": [
        "--from",
        "openmed[mcp]",
        "openmed-mcp",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

If OpenMed is already installed in a virtual environment, set `command` to the
absolute path of that environment's `openmed-mcp` executable and omit the
`uvx`, `--from`, and package arguments.

### TOML server-table clients

Terminal coding clients commonly use a TOML server table:

```toml
[mcp_servers.openmed]
command = "uvx"
args = ["--from", "openmed[mcp]", "openmed-mcp", "--transport", "stdio"]
```

### Typed local blocks

Clients that distinguish local and remote servers with a `type` field usually
accept the equivalent typed block:

```json
{
  "name": "openmed",
  "type": "stdio",
  "command": "uvx",
  "args": ["--from", "openmed[mcp]", "openmed-mcp", "--transport", "stdio"]
}
```

Client field names vary, but the command and argument sequence is the same.

## Streamable HTTP connections

Start a loopback-only server for a local URL-based client:

```bash
openmed-mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8081 \
  --streamable-http-path /mcp
```

The MCP endpoint is `http://127.0.0.1:8081/mcp`. A typed remote configuration
typically looks like this:

```json
{
  "name": "openmed",
  "type": "http",
  "url": "http://127.0.0.1:8081/mcp"
}
```

Hosted-assistant developer connectors use the same URL. They cannot reach a
loopback address on your workstation; deploy through a private network or an
authenticated HTTPS gateway instead of exposing the OpenMed process directly.

## Journey workflow tools and generated clients

Six fixed-resource, read-only tools expose journey, cohort, dataset, registry,
measure, and trial-review workflows. Their schemas are derived from the same
versioned Journey page contract used by REST, GraphQL, and read-only SQL:

- `openmed_read_journey`
- `openmed_read_cohort`
- `openmed_read_dataset`
- `openmed_read_registry`
- `openmed_read_measure`
- `openmed_read_trial_review`

Every result carries evidence identifiers, schema and immutable snapshot
versions, the access-policy decision, controlled warnings, and explicit review
metadata. The result contract excludes raw source text. All six tools advertise
`readOnlyHint=true`, `destructiveHint=false`, and closed-world execution.

The machine-readable generation contract is available at
`openmed://journey-workflows`. Regenerate the Python and TypeScript client
surfaces after changing a Journey workflow schema:

```bash
python scripts/generate_journey_workflow_clients.py
python scripts/generate_journey_workflow_clients.py --check
```

Python clients expose `journey()`, `cohort()`, `dataset()`, `registry()`,
`measure()`, and `trial_review()`. The TypeScript client exposes the same names,
with `trialReview()` using normal TypeScript casing.

The registry document marks every tool as either state-changing or read-only.
State-changing tools continue through the signed, single-use consent-receipt
verification path; the read-only Journey tools never accept a receipt as a
substitute for access policy.

## Fixed-option decisions

`openmed_decide` scores bounded caller-supplied options, preserves caller
ordering, applies the selected calibration profile, and returns typed
abstention, denial, conflict, unsupported, timeout, and failure states. It is a
read-only, non-destructive, idempotent, closed-world tool. The result always
requires human review and never authorizes a clinical action. Its request and
result schemas are identical to the Python and REST contracts described in
[Fixed-option decision API](api/fixed-option-decisions.md).

## Canonical clinical agent workflow

MCP clients can discover the `openmed-clinical-workflow` prompt, the
`openmed://clinical-workflow` guidance resource, and the
`openmed://clinical-workflow/golden-agent-run` synthetic fixture through the
standard MCP prompt and resource listings. The `openmed://tool-registry`
document also publishes a `workflows` entry with the prompt, resources, tool
names, stage order, and registered artifact schema identifiers, so clients do
not need to hardcode the workflow contract.

The workflow de-identifies first, then executes the canonical pipeline order:
`detect`, `context`, `sections`, `relations`, `ground`, `export`, and `risk`.
Each stage produces a registered artifact:

| Artifact | Contents allowed after the de-identification boundary |
|---|---|
| `deidentify` | De-identified text and a source-text hash |
| `detect` | Canonical text-free spans, model identifier, and count |
| `context` | Canonical spans with clinical context metadata |
| `sections` | Canonical spans and surface-free section offsets |
| `relations` | Canonical spans and surface-free relation endpoints |
| `ground` | Terminology codes, scores, hashes, and provenance |
| `export` | Identifier-free FHIR Bundle |
| `risk` | Aggregate residual-risk summary |

Execution is local and has zero network egress by default. Keep source text,
direct identifiers, entity surfaces, and reversible mappings inside the
trusted local runtime. Anything after the de-identification boundary may carry
only de-identified text, offsets, hashes, canonical spans, terminology
provenance, identifier-free FHIR resources, and aggregate risk values. Do not
write source text or identifiers to logs or agent traces.

External-LLM-capable stages remain disabled unless an operator explicitly opts
in, and they must route through the OpenMed privacy gateway. Raw source text
must never cross that boundary. Grounded codes and exported resources require
human review and must not automatically trigger clinical, treatment, billing,
or medical-device decisions.

## Remote authentication and protocol headers

The built-in MCP server does not validate API keys or bearer tokens. Never bind
it to a public interface without a TLS-terminating reverse proxy or gateway
that authenticates every request. Configure the client to send the gateway's
bearer token, for example:

```json
{
  "name": "openmed",
  "type": "http",
  "url": "https://openmed.example.org/mcp",
  "headers": {
    "Authorization": "Bearer ${OPENMED_MCP_TOKEN}"
  }
}
```

Keep the token in the client's secret or environment-variable store rather
than committing it to a configuration file.

Streamable HTTP clients also send the `MCP-Protocol-Version` header on requests
after initialization. SDK-based clients set it automatically to the version
negotiated during the `initialize` exchange. If a gateway allowlists headers,
forward `MCP-Protocol-Version`, `Mcp-Session-Id`, `Content-Type`, `Accept`, and
`Authorization`; do not replace the negotiated protocol version with a fixed
value at the proxy.

## Structured tool errors

Expected OpenMed failures return structured content with `is_error: true` and
the same stable code exposed by the Python API and REST service:

```json
{
  "error": {
    "code": "input_error",
    "message": "The request input is malformed. Correct the documented field and retry.",
    "details": {"argument": "text"}
  },
  "is_error": true
}
```

Branch on `error.code`, not the human-readable message. Error payloads never
echo tool arguments, clinical text, mappings, credentials, or upstream
exception text. Security, authorization, consent, and tool-schema errors retain
their existing specialized codes. See
[Structured public errors](api/errors.md) for the complete Python/REST/MCP
mapping.

## Environment-variable defaults

The command-line flags can also be configured with environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `OPENMED_MCP_TRANSPORT` | `stdio` | `stdio` or `streamable-http` |
| `OPENMED_MCP_HOST` | `127.0.0.1` | HTTP bind address |
| `OPENMED_MCP_PORT` | `8081` | HTTP port |
| `OPENMED_MCP_PATH` | `/mcp` | Streamable HTTP endpoint path |

Command-line flags override these values. Keep `OPENMED_MCP_HOST=127.0.0.1`
unless an authenticated network boundary is already in place.

## Troubleshooting

- **Command not found:** install the `mcp` extra and ensure the selected Python
  environment's executable directory is on `PATH`.
- **Client starts and immediately disconnects:** keep stdio reserved for MCP
  protocol messages; avoid wrapper scripts that print banners to stdout.
- **HTTP client receives 404:** confirm the URL includes the configured
  `--streamable-http-path`, which defaults to `/mcp`.
- **Remote client cannot connect:** loopback is intentionally local-only. Use a
  private route or authenticated HTTPS gateway rather than changing the bind
  address without access controls.
