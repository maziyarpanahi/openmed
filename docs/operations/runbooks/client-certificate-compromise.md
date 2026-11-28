# Client certificate compromise runbook

Use this runbook when the private key of an mTLS client certificate may have
been exposed. For example, it may have been copied off a host, included in an
image or backup, or used by an unknown workload. Read the
[shared rules](service-runbooks.md#rules-for-tickets-logs-and-chat) first.

## Compromised client certificate

When `OPENMED_SERVICE_MTLS_ENABLED` is `true`, every request needs a client
chain that verifies against `OPENMED_SERVICE_MTLS_CA_BUNDLE`. The chain comes
from the ASGI TLS extension (`client_cert_chain`). It can also come from
`OPENMED_SERVICE_MTLS_CLIENT_CERT_HEADER`, but only when the request arrives
from an address in `OPENMED_SERVICE_MTLS_TRUSTED_PROXIES`.
`OPENMED_SERVICE_MTLS_PRINCIPALS` maps an exact subject or SAN value to a
principal and scopes. A verified certificate with no mapping gets the principal
`mtls:<sha256 fingerprint>` and no scopes.

The service does not check CRLs or OCSP. A stolen certificate keeps passing
chain verification until it expires, or until its issuing CA is no longer in the
CA bundle. Reloading the CA bundle or the server certificate without a restart
is pending in #3580.

### Detection signals

- Access logs record `identity` and `credential_type` for mTLS requests.
  `identity` is the mapped principal or `mtls:<fingerprint>`. Look for the
  compromised principal at unexpected times or volumes. Look for unknown
  `mtls:` fingerprints.
- More `401` responses with `mtls_certificate_invalid` or
  `mtls_certificate_required` can mean probing with a stolen or tampered
  chain. Watch `openmed_service_request_total`.
- More `403` responses with `forbidden` can mean an unmapped certificate
  reaching scoped routes.
- Your PKI or host monitoring reports a key exposure.

### Containment

Use the strongest step you can take now, then continue with rotation.

1. Remove the certificate's subject or SAN from `OPENMED_SERVICE_MTLS_PRINCIPALS`
   and restart. The certificate then maps to an `mtls:` principal with no
   scopes. Scoped routes reject it with `forbidden`. Unscoped probe routes such
   as `GET /readyz` stay reachable.
2. If TLS ends at a reverse proxy that forwards the chain in
   `OPENMED_SERVICE_MTLS_CLIENT_CERT_HEADER`, deny the certificate's
   fingerprint at the proxy. Check that `OPENMED_SERVICE_MTLS_TRUSTED_PROXIES`
   lists only that proxy. This stops other hosts from injecting the header.
3. If API keys or JWTs are also enabled, rotate any credentials that the same
   workload held. Use the [credential compromise runbook](credential-compromise.md).

Removing the mapping does not stop chain verification. It only removes the
certificate's scopes. If the same subject or SAN is reused for the replacement
certificate, the stolen certificate gets the same principal again when you
re-add the mapping.

### Rotation

1. Issue the replacement certificate with a new SAN value, for example a new
   workload instance name. Map only that new value in
   `OPENMED_SERVICE_MTLS_PRINCIPALS`. Mapped values must be unique; a value
   listed twice fails startup.
2. To fully revoke the stolen certificate, move client issuance to a new CA
   hierarchy. Then replace `OPENMED_SERVICE_MTLS_CA_BUNDLE` with a bundle that
   contains only the new CA. A root CA in the bundle trusts every intermediate
   it ever signed. Keeping the old root keeps the stolen certificate valid.
   Use a new root that never signed the old issuer.
3. Update the transport that enforces the TLS handshake to use the same bundle.
   This is the ASGI server's client CA setting or the reverse proxy's. Then
   restart both.
4. Re-issue certificates to every other client of the old issuer before you
   remove it. Otherwise they lose access at the restart.

### Recovery

- Review which scopes the mapping granted and which routes those scopes
  reach. Start the [breach notification runbook](../../compliance/breach-notification-runbook.md)
  if a PHI-returning route may have been used.
- Revoke the certificate in your own PKI and record the revocation there, even
  though OpenMed does not check revocation.

### Verification

1. With a synthetic client that holds the stolen certificate, confirm
   `forbidden` on a scoped route after step 1 of containment. After the CA
   change, confirm `mtls_certificate_invalid` or a failed TLS handshake.
2. With the replacement certificate, confirm the mapped principal appears in
   the access log `identity` field and a scoped route succeeds.
3. Confirm `GET /readyz` and `GET /livez` still answer for your health checks.

### Keep out of tickets and logs

- Private keys, PEM certificates and chains, and CA bundles. Record the SHA-256
  fingerprint of the compromised certificate and of the new issuer.
- Subject and SAN values that contain a person's name or a host name you
  consider sensitive. Record the mapped principal label instead.
- The forwarded certificate header value, and proxy debug logs that print it.
