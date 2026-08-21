# Mutual TLS Client Authentication

Mutual TLS (mTLS) can require every REST caller to present a client
certificate issued by a configured private CA. OpenMed validates the client
chain again at the application boundary, exposes the verified certificate
identity to handlers, and can map an exact subject or subject alternative name
(SAN) to the principal and scopes used by route authorization.

mTLS is disabled by default. Enable it without changing application code:

```bash
OPENMED_SERVICE_MTLS_ENABLED=true \
OPENMED_SERVICE_MTLS_CA_BUNDLE=/etc/openmed/client-ca.pem \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

`OPENMED_SERVICE_MTLS_CA_BUNDLE` must contain at least one trusted CA
certificate. It must not contain a production private key. Online CRL and OCSP
checking are not performed; certificate issuance and revocation infrastructure
remain the deployer's responsibility.

## Terminate TLS at the application

For direct TLS, use an ASGI server that implements the ASGI TLS extension and
places the leaf-first PEM client chain in
`scope["extensions"]["tls"]["client_cert_chain"]`. Configure the server to
require client certificates against the same CA bundle. For example, the
corresponding Uvicorn transport flags are:

```bash
uvicorn openmed.service.app:app \
  --host 0.0.0.0 \
  --port 8443 \
  --ssl-keyfile /etc/openmed/server-key.pem \
  --ssl-certfile /etc/openmed/server-cert.pem \
  --ssl-cert-reqs 2 \
  --ssl-ca-certs /etc/openmed/client-ca.pem
```

The transport must both enforce the TLS handshake and expose the verified
chain through the ASGI TLS extension. If the server does not expose that
extension, OpenMed cannot recover the peer certificate from an HTTP request and
returns `mtls_certificate_required`; use a compatible server or the trusted
proxy pattern below.

## Terminate TLS at a sidecar or ingress

A sidecar or ingress may terminate mTLS and forward a URL-escaped PEM client
chain in a dedicated header. Configure both the header name and the exact proxy
addresses or CIDR networks allowed to supply it:

```bash
OPENMED_SERVICE_MTLS_ENABLED=true \
OPENMED_SERVICE_MTLS_CA_BUNDLE=/etc/openmed/client-ca.pem \
OPENMED_SERVICE_MTLS_CLIENT_CERT_HEADER=X-OpenMed-Client-Cert \
OPENMED_SERVICE_MTLS_TRUSTED_PROXIES=127.0.0.1/32,10.42.0.0/16 \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080 --no-proxy-headers
```

The proxy must:

1. remove the client-supplied `X-OpenMed-Client-Cert` header;
2. require and verify a client certificate at its TLS listener;
3. URL-escape the leaf-first PEM chain into the configured header; and
4. connect to OpenMed only from an address in the trusted-proxy list.

OpenMed rejects the header from any other peer and independently verifies the
forwarded chain against `OPENMED_SERVICE_MTLS_CA_BUNDLE`. Do not configure a
public or shared proxy network as trusted. The ASGI `scope["client"]` value must
remain the transport peer address used for this check. Disable generic proxy
header rewriting as shown above, or configure the server so untrusted
`X-Forwarded-For` input cannot replace that peer address.

## Map certificates to principals and scopes

Verified certificate details are available to handlers as
`request.state.mtls_identity`:

- `subject`: the RFC 4514 distinguished name;
- `sans`: typed values such as `uri:spiffe://openmed.test/clinic-api`,
  `dns:clinic-api.internal`, or `ip:10.42.0.8`; and
- `fingerprint_sha256`: the lowercase leaf-certificate fingerprint.

Use exact subject or SAN values to assign a stable, non-PHI service principal
and route scopes:

```bash
OPENMED_SERVICE_MTLS_PRINCIPALS='[
  {
    "identities": ["uri:spiffe://clinic.example/workload/openmed-client"],
    "principal": "clinic-api",
    "scopes": ["analyze:write", "pii:read", "pii:write", "models:read"]
  }
]'
```

The mapped `AuthPrincipal` is available as `request.state.auth_principal` and
`request.scope["openmed.auth"]`. When no mapping matches, OpenMed uses
`mtls:<sha256-fingerprint>` with no scopes. This authenticates the certificate
without silently granting application permissions.

## Combine mTLS with JWT or API keys

mTLS protects the connection and supplies a workload identity. REST
authentication remains independently configurable with
`OPENMED_SERVICE_AUTH_ENABLED=true`. An explicit JWT or API key takes
precedence for route authorization, while `request.state.mtls_identity`
continues to identify the certificate-bearing workload. Without an explicit
credential, the mapped mTLS principal is used, so only scopes granted by
`OPENMED_SERVICE_MTLS_PRINCIPALS` can authorize scoped routes.

To require JWT on normal application routes, leave the mTLS principal without
route scopes and configure JWT keys and route scopes as described in
[REST Service Authentication](authentication.md). The TLS certificate is still
required before the JWT is evaluated.

## Logging and PHI safety

Client certificate PEM, subject, SANs, and fingerprint are never written to
access logs. A successful mTLS request adds only the configured principal (or
the fallback fingerprint-based service identifier) as `identity`, plus
`credential_type=mtls`. Use machine and workload identities in certificates;
do not put patient names, record identifiers, or other PHI in certificate
subjects, SANs, or principal mappings.
