---
name: deploying-openmed-mcp
description: "Run OpenMed's Model Context Protocol (MCP) server so coding agents (Claude Code, Codex) and chat clients can call clinical NER, PII extraction, and de-identification as tools, on-device. Use when the user wants to add OpenMed to an agent's MCP config, expose de-id/NER as MCP tools, run an MCP server over stdio or Streamable HTTP, give Claude/Codex access to OpenMed, or containerize the MCP server. Covers the mcp extra, create_mcp_server, the 7 tools (openmed_analyze_text, openmed_extract_pii, openmed_deidentify, openmed_list_models, openmed_list_pii_languages, openmed_loaded_models, openmed_unload_model), the resources and prompts, stdio vs streamable-http transports, ServiceRuntime env config, and MCP client config snippets."
license: Apache-2.0
metadata:
  project: OpenMed
  category: deployment-ops
  pairs: adjacent
  version: "1.0"
---

# Deploying the OpenMed MCP server

`openmed.mcp.server` exposes OpenMed's clinical NLP as **Model Context Protocol**
tools, so coding agents (Claude Code, Codex) and chat clients can de-identify and
analyze clinical text by calling tools instead of writing glue code. It runs
**on-device** — models are local, no telemetry — and the server instructs
clients to send real PHI only to instances the user operates.

## When to use this skill

When an agent or LLM client should be able to *invoke* OpenMed: add it to a
coding agent's MCP config, give a chat client de-id/NER tools, or run a shared
MCP endpoint for a team. For programmatic HTTP from your own services, prefer
`serving-openmed-rest-api`; for corpora, `batch-processing-clinical-text`.

## Quick start

```bash
pip install "openmed[mcp]"                 # FastMCP / MCP SDK

# stdio transport (what coding agents spawn): default
python -m openmed.mcp.server

# Streamable HTTP transport (network-reachable):
python -m openmed.mcp.server --transport streamable-http --host 127.0.0.1 --port 8081
```

```python
# Or embed it:
from openmed.mcp.server import create_mcp_server
server = create_mcp_server()              # FastMCP("OpenMed", ...) with tools+resources+prompts
server.run(transport="stdio")             # or "streamable-http"
```

CLI flags (`build_arg_parser`): `--transport {stdio,streamable-http,http}`,
`--host`, `--port`, `--streamable-http-path` (default `/mcp`), `--version`.
Env equivalents: `OPENMED_MCP_TRANSPORT`, `OPENMED_MCP_HOST`,
`OPENMED_MCP_PORT` (8081), `OPENMED_MCP_PATH`.

## The 7 tools (confirmed in `openmed/mcp/server.py`)

| Tool | What it does | Key args |
| --- | --- | --- |
| `openmed_analyze_text` | clinical NER | `text`, `model_name` (`disease_detection_superclinical`), `confidence_threshold`, `group_entities`, `aggregation_strategy`, `sentence_*`, `keep_alive` |
| `openmed_extract_pii` | detect PII/PHI spans | `text`, `model_name` (default PII model), `confidence_threshold` (0.5), `use_smart_merging`, `lang`, `normalize_accents` |
| `openmed_deidentify` | mask/remove/replace/hash/shift dates | `text`, `method` (`mask`), `confidence_threshold` (0.7), `keep_year`, `shift_dates`, `date_shift_days`, `keep_mapping`, `lang` |
| `openmed_list_models` | list registry models | `category`, `pii_language`, `limit` |
| `openmed_list_pii_languages` | supported PII languages + default models | — |
| `openmed_loaded_models` | resident-model status of the MCP runtime | — |
| `openmed_unload_model` | free one model or all inactive models | `model_name`, `all_models` |

It also registers **resources** — `openmed://models`, `openmed://pii-languages`,
`openmed://examples` (synthetic) — and **prompts** `openmed-clinical-ner` and
`openmed-pii-deidentify` that nudge the agent toward safe, correct calls.

## Adding it to a coding agent

```json
// Claude Code: .mcp.json (or ~/.claude.json) — stdio transport
{
  "mcpServers": {
    "openmed": {
      "command": "python",
      "args": ["-m", "openmed.mcp.server"],
      "env": { "OPENMED_PROFILE": "prod" }
    }
  }
}
```

For a shared HTTP deployment, run `--transport streamable-http` and point the
client at `http://<host>:8081/mcp`. The agent then sees the 7 tools and can call
e.g. `openmed_deidentify` on a snippet before sending it elsewhere.

## Runtime config

The MCP server shares OpenMed's `ServiceRuntime` (`ServiceRuntime.from_env()`),
so the same env vars as the REST service apply: `OPENMED_PROFILE`,
`OPENMED_SERVICE_PRELOAD_MODELS`, `OPENMED_SERVICE_KEEP_ALIVE`,
`OPENMED_SERVICE_MAX_RESIDENT_MODELS`. Preload to avoid first-call latency;
`openmed_unload_model`/`openmed_loaded_models` let an agent manage memory.

## Running in Docker

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir "openmed[mcp]"
ENV OPENMED_MCP_TRANSPORT=streamable-http \
    OPENMED_MCP_HOST=0.0.0.0 OPENMED_MCP_PORT=8081 \
    OPENMED_SERVICE_PRELOAD_MODELS="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
EXPOSE 8081
CMD ["python", "-m", "openmed.mcp.server"]
```

stdio servers are spawned by the client and don't need a port; use HTTP only for
shared/remote access, behind your own auth proxy. Mount the model cache so the
container starts offline.

## Workflow

1. **Install + launch.** `pip install "openmed[mcp]"`, then
   `python -m openmed.mcp.server` (stdio) or `--transport streamable-http`
   for a shared endpoint.
2. **Configure the runtime** via the `ServiceRuntime` env vars (profile,
   preload, keep-alive, max resident) so first calls aren't cold.
3. **Register with the client.** Add the `mcpServers` entry (stdio command, or
   HTTP URL) to the agent's config; the 7 tools, resources, and prompts appear.
4. **Front HTTP with auth/TLS** if remote — the server has none built in; keep
   stdio/local for untrusted-network scenarios.
5. **Let the agent call tools** (`openmed_deidentify` before sharing a snippet,
   `openmed_analyze_text` for NER), and discover models via
   `openmed_list_models` rather than hardcoding.
6. **Manage memory** with `openmed_loaded_models` / `openmed_unload_model`.

## Hand-off to / from OpenMed

- **Same engine:** each tool calls `openmed.analyze_text` / `extract_pii` /
  `deidentify` through the shared runtime — identical results to the library and
  the REST service.
- **REST sibling:** `serving-openmed-rest-api` exposes the same operations as
  HTTP routes for non-agent callers.
- **Discovery:** `openmed_list_models` / `openmed_list_pii_languages` mirror the
  library's `list_*` functions — agents should query, not hardcode.

## Edge cases & gotchas

- **stdio vs HTTP.** Coding agents spawn the server over **stdio** (default) and
  manage its lifecycle; use **streamable-http** only for a shared endpoint, and
  put auth/TLS in front of it (the server has none built in).
- **PHI trust boundary.** The server's instructions tell clients to send real
  PHI only to instances the user controls. Keep it local/self-hosted; don't
  point agents at an OpenMed MCP you don't operate.
- **`keep_mapping=True` returns a re-identification map** in the
  `openmed_deidentify` response — only enable for trusted agents, treat the
  mapping as PHI, never log it.
- **No raw PHI in logs.** Don't add transcript/body logging around the server.
- **Use synthetic examples** in docs/tests/prompts — the bundled
  `openmed://examples` resource is synthetic on purpose.
- **`--transport http`** is accepted as an alias for `streamable-http`.

## Standards & references

- Model Context Protocol specification: https://modelcontextprotocol.io/
- MCP transports (stdio, Streamable HTTP):
  https://modelcontextprotocol.io/docs/concepts/transports
- Claude Code MCP configuration:
  https://docs.anthropic.com/en/docs/claude-code/mcp
- OpenMed source: `openmed/mcp/server.py` (`create_mcp_server`, the 7 tools,
  resources, prompts, `main`/`build_arg_parser`).
