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
python -m openmed.service.logging --host 127.0.0.1 --port 8080
```

`OPENMED_SERVICE_MTLS_CA_BUNDLE` must contain at least one trusted CA
certificate. It must not contain a production private key. OpenMed can reject
revoked client certificates from operator-supplied local certificate revocation
lists (CRLs); see [Reject revoked client certificates](#reject-revoked-client-certificates).
Online CRL fetching and OCSP checking are not performed; certificate issuance,
CRL publication and distribution remain the deployer's responsibility.

## Terminate TLS at the application

For direct TLS, use an ASGI server that implements the ASGI TLS extension and
places the leaf-first PEM client chain in
`scope["extensions"]["tls"]["client_cert_chain"]`. Configure the server to
require client certificates against the same CA bundle. For example, the
corresponding Uvicorn transport flags are:

```bash
python -m openmed.service.logging \
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
python -m openmed.service.logging --host 127.0.0.1 --port 8080 --no-proxy-headers
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

## Reject revoked client certificates

When a client key is compromised, revoke its certificate at your CA and give
OpenMed the resulting CRL. OpenMed reads local CRL files only. It never fetches
CRLs from distribution points and never contacts an OCSP responder.

| Setting | Default | Meaning |
| --- | --- | --- |
| `OPENMED_SERVICE_MTLS_CRL_FILES` | unset | Comma-separated paths to PEM or DER CRL files, one CRL per file. |
| `OPENMED_SERVICE_MTLS_CRL_GRACE_SECONDS` | `0` | Seconds a CRL is still accepted after its `nextUpdate` time. |
| `OPENMED_SERVICE_MTLS_CRL_FAIL_CLOSED` | `true`, or `false` for the `dev` and `test` profiles | Whether invalid or stale CRLs reject requests. It cannot be set to `false` outside the `dev` and `test` profiles. |

```bash
OPENMED_SERVICE_MTLS_ENABLED=true \
OPENMED_SERVICE_MTLS_CA_BUNDLE=/etc/openmed/client-ca.pem \
OPENMED_SERVICE_MTLS_CRL_FILES=/etc/openmed/client-ca.crl.pem \
OPENMED_SERVICE_MTLS_CRL_GRACE_SECONDS=3600 \
python -m openmed.service.logging --host 127.0.0.1 --port 8080
```

How the checks work:

1. At startup, OpenMed reads every configured CRL file and verifies its
   signature against a CA certificate in `OPENMED_SERVICE_MTLS_CA_BUNDLE`
   with a matching subject. The CA must be allowed to sign CRLs (the
   `cRLSign` key usage, when key usage is present). A CRL that cannot be read
   or parsed, has a bad signature, is signed by a CA outside the bundle, has
   no `nextUpdate` time, is a delta CRL, or holds more than one PEM CRL block
   is invalid. Verified CRLs are cached in memory. Changing a file needs a
   restart.
2. On every request, after the client chain is verified and before any
   principal mapping runs, each certificate in the verified chain is looked up
   by serial number in the CRLs signed by its issuer. A revoked leaf or
   intermediate CA certificate is rejected with HTTP 401 and
   `client_certificate_revoked`, in every profile.
3. On the same request, each CRL used for that chain must still be fresh: the
   current time must not be later than its `nextUpdate` time plus the grace
   period. This runs per request, so a CRL that goes stale while the service
   runs is caught without a restart.

A certificate whose issuer has no configured CRL is not checked for
revocation. Configure one CRL per issuing CA that you need to cover. A CRL
signed by an intermediate CA can be verified only when that intermediate CA
certificate is also in the CA bundle.

### Profile behavior and error codes

`OPENMED_PROFILE` selects how CRL problems are handled. The `dev` and `test`
profiles report problems; every other profile, including the default `prod`
and custom profiles, fails closed.

| Condition | Code | Fail closed (`prod` and other profiles) | Report only (`dev`, `test`) |
| --- | --- | --- | --- |
| Certificate is listed in a CRL | `client_certificate_revoked` | HTTP 401 | HTTP 401 |
| CRL is past `nextUpdate` plus grace | `crl_stale` | HTTP 503 | One warning, request proceeds |
| CRL file is invalid | `crl_invalid` | Service refuses to start | One warning per file, file is skipped |

In report-only mode, "reported" means one `WARNING` record on the
`openmed.service.mtls` logger. It holds only the code, and for an invalid file
its 1-based position in `OPENMED_SERVICE_MTLS_CRL_FILES`. The record also
carries the code in its `openmed_code` attribute. Revoked certificates are
always rejected, even when the CRL that lists them is stale.

Responses, exceptions and log records never contain certificate subjects,
SANs, serial numbers, CRL file paths, or underlying parser or OS error text.
The startup error for an invalid CRL names only the file position.

### Produce and rotate a CRL with OpenSSL

The commands below are an example for a small OpenSSL-managed CA. The paths
are placeholders; keep the CA key off the OpenMed host.

```bash
# Revoke a compromised client certificate in the CA database.
openssl ca -config /path/to/ca.cnf -revoke /path/to/compromised-client.pem

# Issue a new CRL that is valid for 7 days.
openssl ca -config /path/to/ca.cnf -gencrl -crldays 7 \
  -out /path/to/client-ca.crl.pem

# Check the CRL signature and dates before deploying it.
openssl crl -in /path/to/client-ca.crl.pem -CAfile /path/to/client-ca.pem \
  -noout -verify -lastupdate -nextupdate
```

To rotate, issue a new CRL before the current `nextUpdate` time, replace the
file atomically, and restart the service. Set the grace period to cover the
time your rotation job may lag. Alert on `crl_stale` responses, because in a
fail-closed profile every client of that CA is rejected until a fresh CRL is
loaded.

If your sidecar or ingress can also check CRLs, you may enable that too. The
application-level check still runs, so the result is the same whether TLS ends
at OpenMed or at a trusted proxy.

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

## Rotate the client CA without a restart

OpenMed holds the CA bundle as a versioned credential generation. Startup
validates it and refuses to serve when it is unreadable, is not valid PEM,
contains a certificate that is not a CA or is outside its validity window, or
when every CA in it expires within five minutes. A reload validates a new
bundle the same way before it replaces the active one:

- every request is verified against the generation active when it arrives;
- a request already admitted keeps the identity it was verified with, so
  long-running requests, streaming responses and jobs they started are not
  interrupted;
- an invalid replacement is refused with a value-free reason code (such as
  `tls_malformed` or `tls_certificate_expired`) and the previous bundle keeps
  serving. mTLS is never switched off by a failed reload.

When the bundle changes, certificates issued by the outgoing CAs stay accepted
for a bounded trust overlap, then only the new bundle is trusted:

```bash
OPENMED_SERVICE_MTLS_TRUST_OVERLAP_SECONDS=300 \
OPENMED_SERVICE_MTLS_RELOAD_INTERVAL_SECONDS=10
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENMED_SERVICE_MTLS_TRUST_OVERLAP_SECONDS` | `300` | How long the previous CA bundle stays trusted after a change; `0` disables the overlap; at most `86400` |
| `OPENMED_SERVICE_MTLS_RELOAD_INTERVAL_SECONDS` | unset | Poll the bundle's file metadata every N seconds (1 to 3600) and reload when it changes; unset disables polling |

Reloads are local only: the poller above, `app.state.mtls.reload()` from code
that embeds the app, or
`openmed.service.tls_reload.install_reload_signal_handler(app.state.mtls)` to
reload on `SIGHUP` from your own launcher. Replace the bundle atomically
(write a temporary file, then rename it, or use a secret volume). Handlers can
read the opaque generation that verified a request from
`request.state.mtls_generation_id`. Certificate-to-principal mappings are
configuration, not part of the credential generation, and do not change on a
reload.

If no CA in the trusted bundle is currently valid, every request, including
`/readyz`, gets `503 not_ready` with `details.reason` set to
`tls_client_trust_unenforceable` rather than being served without client
verification.

The REST server certificate itself is not reloaded in-process. Uvicorn builds
its TLS context from file paths once at startup, and Python can only load a
certificate chain from a file, so an in-process swap would need key material
in a temporary file. Rotate the REST server certificate by restarting behind
the shutdown drain, or terminate TLS at a sidecar or ingress that reloads its
own certificate. With direct Uvicorn TLS, `--ssl-ca-certs` is also fixed at
startup: stage a bundle that contains both the old and new CAs there, and let
OpenMed's reloadable bundle and trust overlap decide which clients are
accepted.

## gRPC transport

The gRPC service terminates TLS and mutual TLS itself through grpcio rather
than through these REST settings. See
[gRPC Service: Transport Security](grpc.md#transport-security) for its
loopback default, `GrpcTLSConfig`, and startup reason codes, and
[Certificate rotation](grpc.md#certificate-rotation) for reloading its server
certificate, key and client CA without a restart. gRPC maps verified client
certificates through the same `OPENMED_SERVICE_MTLS_PRINCIPALS` mappings, so
one certificate resolves to the same principal and scopes on both transports.

## Logging and PHI safety

Client certificate PEM, subject, SANs, and fingerprint are never written to
access logs. A successful mTLS request adds only the configured principal (or
the fallback fingerprint-based service identifier) as `identity`, plus
`credential_type=mtls`. Use machine and workload identities in certificates;
do not put patient names, record identifiers, or other PHI in certificate
subjects, SANs, or principal mappings.
