# Synthetic mTLS test material

`tests/unit/service/test_mtls_auth.py` creates a synthetic root CA, a trusted
client certificate, and a client certificate signed by a different CA in the
pytest temporary directory. It uses short-lived RSA keys and certificates with
synthetic workload-only names:

- subject: `CN=synthetic-clinic-client,O=OpenMed Test`
- URI SAN: `spiffe://openmed.test/clinic-api`
- DNS SAN: `clinic-api.test`
- extended key usage: TLS client authentication

The generated client chain exercises successful authentication, principal and
scope mapping, handler-visible subject/SAN fields, and rejection of an
untrusted issuer. The material is regenerated for each test session. No private
keys, production certificates, patient identifiers, or other PHI are committed
to the repository.
