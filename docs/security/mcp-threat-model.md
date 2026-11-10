# MCP clinical gateway threat model and deployment guide

This document describes the security boundary for the OpenMed MCP gateway. It
is a design and deployment aid, not a substitute for a local risk assessment,
an identity-provider review, or clinical governance.

## Security objectives

The gateway treats every MCP document, resource, prompt, and tool argument as
untrusted input. Authorization is decided separately from prompt-injection
screening. A token is accepted only when its resource indicator and audience
identify this exact MCP resource and its scopes authorize the selected tool.
Inbound MCP bearer credentials are never forwarded to terminology, FHIR,
model, or other upstream services.

The gateway also keeps credentials and raw tool payloads out of errors and
logs. Operational evidence is limited to request identifiers, route/status
metadata, bounded timing, authorization decision codes, token key identifiers
when intentionally provided by the deployment, and hashes or offsets produced
by an approved audit workflow.

## Trust boundaries and threats

| Boundary | Assumption | Main threats | Required control |
| --- | --- | --- | --- |
| Local stdio client to process | The operator controls both ends | Malformed JSON, oversized arguments, prompt injection | Registry schema validation, payload bounds, untrusted-content policy |
| Remote MCP client to gateway | The network and client are untrusted | Token substitution, wrong audience, missing scope, replay, credential leakage | HTTPS, protected-resource metadata, RFC 8707 binding, per-tool scopes, safe errors |
| Gateway to an upstream service | The upstream is a separate trust domain | Bearer-token passthrough, PHI or tool-payload leakage | Allowlisted headers, separate upstream credentials, no inbound token forwarding |
| Authorization server to gateway | Metadata and tokens are independently operated | Downgrade, issuer mix-up, weak OAuth client | HTTPS metadata, issuer/resource equality, S256 PKCE, exact redirect URI checks |

Prompt-injection detection is a defense-in-depth signal. It does not grant
access, override a scope decision, or make a state-changing tool safe by
itself.

## Local stdio mode

Stdio is the default and stays offline after model assets are available. Do not
set the MCP authorization environment variables for a local stdio process.
The process still applies registered JSON schemas, bounded payloads, and the
untrusted-content policy. A local operator controls the process boundary, so
state-changing tools are permitted only inside that local process.

Example:

```bash
openmed-mcp --transport stdio
```

Do not put access tokens, client secrets, or document text in command-line
arguments, environment snapshots, shell history, or diagnostic output.

## Authenticated remote mode

Remote mode is opt-in. Configure a resource URL representing the exact MCP
endpoint and an authorization-server issuer. Non-local URLs must use HTTPS;
HTTP is available only when `OPENMED_MCP_ALLOW_INSECURE_LOCALHOST=true` is
explicitly set for a loopback development test.

```bash
export OPENMED_MCP_AUTH_ENABLED=true
export OPENMED_MCP_RESOURCE_URL=https://mcp.example.test/mcp
export OPENMED_MCP_AUTHORIZATION_SERVER_URL=https://login.example.test
export OPENMED_MCP_REQUIRED_SCOPES=mcp:read
openmed-mcp --transport streamable-http --host 0.0.0.0 --port 8081
```

The authorization server must publish RFC 8414 metadata with S256 PKCE
support. The protected resource must publish RFC 9728 metadata that names the
same resource URL and the authorization server issuer. Authorization-code
requests include the RFC 8707 `resource` parameter and an S256 code challenge.

The global scope is a server-level grant. The policy also requires the
per-tool scope `mcp:tool:<tool-name>` unless an embedding application supplies
an explicit per-tool scope map. The state-changing tools additionally require
the configured state-change scope (default `mcp:state:write`).

Remote mode must use a verified token resolver or an identity-provider-backed
MCP provider. An opaque inbound bearer string is never trusted merely because
it is present. Tokens must have an exact `resource` claim or an exact `aud`
claim for the configured MCP resource; a token for another resource is
rejected. Scope checks are exact, with only explicitly configured namespace
wildcards supported.

## PHI and credential boundaries

- Keep PHI on the local side of the gateway unless a separately reviewed
  upstream integration explicitly authorizes the transfer.
- Use separate upstream credentials. Never copy the MCP `Authorization`
  header, access token, refresh token, client assertion, or client secret to an
  upstream request.
- Keep token stores in a protected process-owned secret store. Do not commit,
  persist in a general-purpose cache, or include tokens in audit records.
- Do not log tool arguments, document/resource text, request bodies, response
  bodies, authorization headers, or exception strings. Use the structured
  security decision code and a correlation identifier instead.
- Local synthetic fixtures may contain recognizable markers for tests; those
  markers must not be used as production evidence or copied into logs.

## Token rotation and incident evidence

Use short-lived access tokens, rotate refresh tokens, and revoke the affected
credential set after a suspected leak. Rotate signing keys with overlapping
verification windows and remove old keys after all issued tokens expire. A
rotation must not require sending PHI to a cloud service.

For an incident, retain only the minimum evidence needed to reconstruct the
decision: UTC timestamp, request/correlation ID, route, status, safe decision
code, configured resource identifier, token key ID or issuer subject when
approved by policy, and hashes/offsets from a local audit artifact. Never
retain bearer values, client secrets, assertions, raw authorization headers,
or document/tool payloads. Treat a prompt-injection finding as an input event,
not as proof that authorization was bypassed.

## Deployment checklist

- [ ] Stdio is used for local/offline workflows unless remote access is needed.
- [ ] Every non-local resource and authorization URL uses HTTPS.
- [ ] Protected-resource metadata and authorization-server metadata agree on
      the resource and issuer.
- [ ] The token verifier validates expiry, exact audience/resource, and every
      selected tool scope.
- [ ] Authorization-code clients require S256 PKCE and exact registered
      redirect URIs.
- [ ] Upstream adapters use an allowlist of headers and independent credentials.
- [ ] State-changing tools require a separate explicit permission scope.
- [ ] Logs, errors, caches, metrics, and audit evidence contain no credentials,
      authorization headers, or raw tool/document payloads.
- [ ] Synthetic offline tests cover wrong audience, missing scope, token
      passthrough, insecure URLs, redirect/PKCE failures, prompt injection,
      payload bounds, and log redaction.
