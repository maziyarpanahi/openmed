# Local Secret Detection

`openmed.core.secrets` provides a deterministic local pass for credentials and
secret tokens that may appear in agent traces, HTTP headers, configuration
payloads, or environment-style text. It complements PII detection; it does not
replace credential rotation, provider validation, or a compliance review.

## Use the detector

```python
from openmed.core.secrets import detect_secrets

trace = "Authorization: Bearer " + "A" * 24
findings = detect_secrets(trace)

for finding in findings:
    print(finding.to_dict())
```

Each finding contains only:

```text
category     Stable category such as authorization_header or private_key
offset       Half-open character offsets into the input: [start, end]
fingerprint  One-way sha256:<digest> value for local deduplication
```

The matched value is not retained in `SecretFinding`, `to_dict()`, or its
representation. The detector does not log input text, make a network request,
or call a provider. Offsets refer to Python character positions in the exact
string passed to the scan.

## Covered high-confidence shapes

- `authorization`, `proxy-authorization`, and common API/auth headers;
- JWT-like values and recognizable GitHub, Slack, Hugging Face, npm, PyPI,
  Google, Stripe, and cloud access-key prefixes;
- PEM and PGP private-key blocks, including an unmatched private-key header;
- secret-looking values assigned to names such as `API_KEY`, `ACCESS_TOKEN`,
  `PASSWORD`, `DATABASE_URL`, `PRIVATE_KEY`, and `CLIENT_SECRET`.

Explicit placeholders such as `<TOKEN>`, `your-api-key`, `replace-me`, and
runtime references such as `${TOKEN}` are ignored to keep documentation and
configuration examples from becoming findings. High-entropy text without a
recognized shape or secret-name context is also ignored deliberately.

## Safe handling

Use the category, offsets, and fingerprint in reports or deduplication records.
Do not reconstruct or log the sensitive slice from the source text. Findings are
not proof that a credential is active, and a missed or malformed credential must
still be handled through the deployment's normal secret-management process.
