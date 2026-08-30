# MCP Upstream Endpoint Policy

The MCP gateway must validate an operator-configured upstream endpoint before
constructing a request that could carry credentials, terminology, or clinical
content. The endpoint policy is intentionally independent of an HTTP client:
it performs URL parsing, exact-origin approval, DNS resolution, and address
classification without sending a request.

## Remote upstreams

Remote endpoints must use HTTPS and match an exact configured origin. An
origin is a scheme, host, and optional non-default port; paths, query strings,
fragments, wildcards, and user information do not extend the allowlist.

```python
from openmed.mcp.upstream_endpoint_policy import UpstreamEndpointPolicy


def offline_resolver(host: str, port: int) -> list[str]:
    # Production code may use the default resolver. Tests should inject
    # deterministic synthetic answers like this one.
    return {"upstream.example.test": ["93.184.216.34"]}[host]


policy = UpstreamEndpointPolicy(
    allowed_origins={"https://upstream.example.test"},
    resolver=offline_resolver,
)

# Validate immediately before invoking the requester. The returned URL is
# request data and must not be written to logs.
response = policy.call(
    "https://upstream.example.test/fhir",
    requester,
)
```

Every address returned by the resolver is checked. The policy fails closed if
any answer is private, link-local, multicast, unspecified, loopback, reserved,
non-global, or a recognized cloud-metadata address. A mixed public/prohibited
answer is rejected as a whole; callers must not select the first DNS answer.
This re-check occurs on every validation call, which protects redirect hops
and repeated requests from DNS rebinding between lookups.

## Development loopback

Loopback is never enabled by the default policy. A development-only policy can
be opted into explicitly:

```python
policy = UpstreamEndpointPolicy(
    resolver=lambda host, port: ["127.0.0.1"] if host == "localhost" else [],
    allow_loopback=True,
)

policy.authorize("http://localhost:8081/mcp")
```

Only `localhost` and literal loopback addresses qualify, and every resolved
answer must be loopback. A hostname that returns both loopback and public (or
private) answers is rejected. Loopback and remote endpoints cannot be mixed in
one redirect chain.

## Redirect handling and safe errors

Automatic HTTP-client redirects must be disabled. Validate each `Location`
value before following it; relative targets may be resolved against the
previously validated URL:

```python
first = policy.validate("https://upstream.example.test/fhir")
next_endpoint = policy.validate_redirect("/next", base_url=first.url)
```

`UpstreamEndpointPolicyError` and its typed subclasses expose a stable
`reason_code`, `code`, static message, and `to_dict()` payload. They never
include the input URL, user information, query values, resolver exception
text, credentials, or clinical terms. Use `ValidatedUpstreamEndpoint.to_safe_dict()`
for operational metadata. Do not log `.url`; it is retained only so the
requester can use the exact validated request target.

The policy protects network reachability only. It does not authorize the
clinical purpose of an otherwise safe request, inspect request bodies, or
operate a proxy. Tests for this module use synthetic resolver answers and do
not probe real networks or metadata services.
